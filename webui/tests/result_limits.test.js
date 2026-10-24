'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os');
const {Workspace,noLinks}=require('../automation/workspace'),storage=require('../automation/storage');
const {AnalysisJobs}=require('../automation/jobs');
const {Transfers}=require('../automation/transfers');
async function fixture(t){const base=await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()),'omfdm-results-'));t.after(async()=>{await noLinks(base);assert.ok(path.basename(base).startsWith('omfdm-results-'));await fs.rm(base,{recursive:true,force:true});});return base;}
test('result count and byte caps cover writes and preserve smaller replacements above a lowered limit',async t=>{
    const base=await fixture(t);storage.configure(base,{fileLimits:{defaults:{results:1,'result-mib':1}}});
    const ws=new Workspace(base,'wa');await ws.init();
    const first=await ws.write('results/run/Az/one.tiff',Buffer.alloc(1024*1024),null);
    await assert.rejects(ws.write('results/other/one.txt',Buffer.from('x'),null),{code:'QUOTA_EXCEEDED'});
    await assert.rejects(ws.write('results/run/two.txt',Buffer.from('x'),null),{code:'QUOTA_EXCEEDED'});
    storage.configure(base,{fileLimits:{defaults:{results:0,'result-mib':0}}});
    const smaller=await ws.write(first.id,Buffer.from('smaller'),first.revision);assert.equal(smaller.size,7);
    await ws.remove('results/run',(await ws.describe('results/run')).revision);
    assert.equal((await ws.list({category:'results'})).total,0);
});
test('a disappearing nested result fails quota measurement instead of undercounting',async t=>{
    const base=await fixture(t),ws=new Workspace(base,'wa');await ws.init();
    await ws.write('results/run/data.bin',Buffer.from('data'),null);
    const original=fs.readdir;
    fs.readdir=async(directory,...args)=>{
        if(path.basename(directory)==='run') {const error=Error('Disappeared during scan');error.code='ENOENT';throw error;}
        return original(directory,...args);
    };
    try {await assert.rejects(require('../automation/limits').resultUsage(base,'wa'),{code:'ENOENT'});}
    finally {fs.readdir=original;}
});
test('backup upload admission counts declared bytes and other pending transfers',async t=>{
    const base=await fixture(t);storage.configure(base,{fileLimits:{defaults:{'result-mib':1}}});
    const ws=new Workspace(base,'wa');await ws.init();
    await ws.write('results/run/Az/one.csv',Buffer.alloc(700*1024),null);
    const transfers=new Transfers();t.after(()=>transfers.close());
    const actor={id:'agent',user:'wa',guard(){}};
    await assert.rejects(transfers.begin(actor,ws,'backup',400*1024),{code:'QUOTA_EXCEEDED'});
    const first=await transfers.begin(actor,ws,'backup',200*1024);
    await assert.rejects(ws.write('results/run/extra.bin',Buffer.alloc(200*1024),null),{code:'QUOTA_EXCEEDED'});
    await assert.rejects(require('../automation/limits').checkResults(base,'wa',{bytes:200*1024,starting:true}),{code:'QUOTA_EXCEEDED'});
    await assert.rejects(transfers.begin(actor,ws,'backup',200*1024),{code:'QUOTA_EXCEEDED'});
    await transfers.append(actor,{transferId:first.transferId,offset:0,base64:Buffer.alloc(100*1024).toString('base64')});
    await assert.rejects(transfers.begin(actor,ws,'backup',200*1024),{code:'QUOTA_EXCEEDED'});
    await transfers.release(actor,first.transferId);
    const extra=await ws.write('results/run/extra.bin',Buffer.alloc(200*1024),null);
    await ws.remove(extra.id,extra.revision);
    const second=await transfers.begin(actor,ws,'backup',200*1024);
    await transfers.release(actor,second.transferId);
});
test('queued snapshots reserve a result slot and all submitters share admission limits',async t=>{
    const base=await fixture(t);storage.configure(base,{fileLimits:{defaults:{results:1}}});
    const directory=path.join(base,'private');await fs.mkdir(directory);
    const backend={jobs:new Map(),mergeLibraryIntoConfig:a=>a},jobs=new AnalysisJobs(base,{directory},backend);await jobs.init();jobs.stopping=true;
    const ws=new Workspace(base,'wa');await ws.init();
    const config=await ws.write('configs/a.yaml',Buffer.from('variables: {}'),null),image=await ws.write('images/a.png',Buffer.from('image'),null);
    const actor={id:'agent',name:'MCP',user:'wa',guard(){}},input={config:{id:config.id,revision:config.revision},image:{id:image.id,revision:image.revision},idempotencyKey:'one'};
    const accepted=await jobs.start(actor,input);assert.equal(accepted.status,'queued');
    assert.equal((await jobs.start(actor,input)).jobId,accepted.jobId,'quota must not break idempotent retries');
    await assert.rejects(jobs.start({...actor,id:'browser'},{...input,idempotencyKey:'two'}),{code:'QUOTA_EXCEEDED'});
    assert.equal((await ws.list({category:'results'})).total,1);
});
test('snapshot admission includes referenced files and metadata without leaving rejected output',async t=>{
    const base=await fixture(t),ws=new Workspace(base,'wa');await ws.init();
    const jobs=new AnalysisJobs(base,{directory:base},{jobs:new Map()});await jobs.init();jobs.stopping=true;
    await ws.write('results/existing/reference.bin',Buffer.alloc(600*1024),null);
    let config=await ws.write('configs/model.yaml',Buffer.from('initial_az_path: results/existing/reference.bin\n'),null);
    let image=await ws.write('images/model.png',Buffer.from('image'),null);
    storage.configure(base,{fileLimits:{defaults:{'result-mib':1}}});
    const actor={id:'agent',user:'wa',name:'QA',guard(){}};
    const input=key=>({config:{id:config.id,revision:config.revision},image:{id:image.id,revision:image.revision},idempotencyKey:key});
    await assert.rejects(jobs.start(actor,input('reference')),{code:'QUOTA_EXCEEDED'});
    assert.equal((await ws.list({category:'results'})).total,1);
    await ws.remove('results/existing',(await ws.describe('results/existing')).revision);
    config=await ws.write(config.id,Buffer.from('a: 1\n'),config.revision);
    image=await ws.write(image.id,Buffer.alloc(1048500),image.revision);
    await assert.rejects(jobs.start(actor,input('metadata')),{code:'QUOTA_EXCEEDED'});
    assert.equal((await ws.list({category:'results'})).total,0);
});
test('result moves enforce folder count and protect the running destination',async t=>{
    const base=await fixture(t),ws=new Workspace(base,'wa');await ws.init();
    await ws.write('results/one/nested/data.txt',Buffer.from('data'),null);
    storage.configure(base,{fileLimits:{defaults:{results:1}}});
    let nested=await ws.describe('results/one/nested');
    await assert.rejects(ws.rename(nested.id,'results/two',nested.revision),{code:'QUOTA_EXCEEDED'});
    const one=await ws.describe('results/one');await ws.rename(one.id,'results/renamed',one.revision);
    storage.configure(base,{fileLimits:{defaults:{results:2}}});
    nested=await ws.describe('results/renamed/nested');await ws.rename(nested.id,'results/two',nested.revision);
    assert.equal((await require('../automation/limits').resultUsage(base,'wa')).count,2);
    const guarded=new Workspace(base,'wa',{busy:id=>id.startsWith('results/renamed/')});
    const file=await ws.describe('results/two/data.txt');
    await assert.rejects(guarded.rename(file.id,'results/renamed/injected.txt',file.revision),{code:'BUSY'});
});
test('a connected agent or a different running job does not lock completed results',async t=>{
    const base=await fixture(t),ws=new Workspace(base,'wa');await ws.init();
    const jobs=new AnalysisJobs(base,{directory:base},{jobs:new Map(),userHasActiveSolver:()=>true});
    jobs.records=[{jobId:'active',user:'wa',resultId:'results/live',status:'running'},{jobId:'old',user:'wa',resultId:'results/done',status:'completed',completed:true,converged:true}];
    const guarded=new Workspace(base,'wa',{busy:id=>jobs.busy('wa',id)});
    await ws.write('results/done/file.txt',Buffer.from('saved'),null);
    await guarded.read('results/done/file.txt',{text:true});
    await guarded.remove('results/done',(await guarded.describe('results/done')).revision);
    assert.equal(jobs.busy('wa','results/live/Az/one.tiff'),true);
    assert.equal(jobs.busy('wa','results/other'),false);
    assert.equal((await ws.list({category:'results'})).total,0);
});
test('job pages are workspace-owned and keep history out of the active queue',()=>{
    const jobs=new AnalysisJobs('unused',{directory:'.'},{jobs:new Map()});
    jobs.records=Array.from({length:45},(_,i)=>({jobId:String(i),user:'wa',status:'completed'})).concat({jobId:'active',user:'wa',status:'queued'},{jobId:'foreign',user:'wb',status:'queued'});
    assert.deepEqual(jobs.page('wa').jobs.map(j=>j.jobId),['active']);
    const history=jobs.page('wa',{view:'history',offset:20,limit:20});assert.equal(history.jobs.length,20);assert.equal(history.nextOffset,40);
    assert.equal(jobs.page('wa',{view:'overview'}).jobs.length,21);
    assert.throws(()=>jobs.page('wa',{view:'history',limit:1000}),{code:'INVALID_INPUT'});
});
test('overview includes an older held job when it finishes after newer history',()=>{
    const jobs=new AnalysisJobs('unused',{directory:'.'},{jobs:new Map()});
    jobs.records=[{jobId:'late',user:'wa',status:'held',created:'2026-10-01T00:00:00Z'},
        ...Array.from({length:25},(_,i)=>({jobId:`new-${i}`,user:'wa',status:'completed',created:'2026-10-02T00:00:00Z',finished:'2026-10-02T00:01:00Z'}))];
    assert.ok(jobs.page('wa',{view:'overview'}).jobs.some(j=>j.jobId==='late'));
    jobs.records[0].status='completed';jobs.records[0].finished='2026-10-03T00:00:00Z';
    assert.equal(jobs.page('wa',{view:'overview'}).jobs[0].jobId,'late');
});
test('ZIP result growth is checked once for the batch before any result is written',async t=>{
    const base=await fixture(t);storage.configure(base,{fileLimits:{defaults:{'result-mib':1}}});
    const ws=new Workspace(base,'wa');await ws.init();
    const transfers=new Transfers();t.after(()=>transfers.close());
    const actor={id:'agent',user:'wa',guard(){}};
    const zip=new (require('yazl').ZipFile)();
    for(let i=0;i<3;i++)zip.addBuffer(Buffer.alloc(400*1024),`results/new/${i}.bin`,{compress:true});
    const reading=(async()=>{const chunks=[];for await(const chunk of zip.outputStream)chunks.push(chunk);return Buffer.concat(chunks);})();
    zip.end();const bytes=await reading;
    const transfer=await transfers.begin(actor,ws,'backup',bytes.length);
    await transfers.append(actor,{transferId:transfer.transferId,offset:0,base64:bytes.toString('base64')});
    const preview=await transfers.preview(actor,ws,transfer);
    const input={...transfer,files:preview.files.map(f=>({id:f.id,revision:f.expectedRevision}))};
    await assert.rejects(transfers.import(actor,ws,input),{code:'QUOTA_EXCEEDED'});
    assert.equal((await ws.list({category:'results'})).total,0);
    storage.configure(base,{fileLimits:{defaults:{'result-mib':2}}});
    const limits=require('../automation/limits'),original=limits.checkResults;let checks=0;
    limits.checkResults=async(...args)=>{checks++;return original(...args);};
    try {assert.equal((await transfers.import(actor,ws,input)).applied.length,3);}
    finally {limits.checkResults=original;}
    assert.equal(checks,1);
});
test('overlapping commits of a mixed backup finish and leave the workspace writable',async t=>{
    const base=await fixture(t),ws=new Workspace(base,'wa');await ws.init();
    const transfers=new Transfers();t.after(()=>transfers.close());
    const actor={id:'agent',user:'wa',guard(){}};
    await ws.write('configs/a.yaml',Buffer.from('a: 1'),null);
    const zip=new (require('yazl').ZipFile)();
    zip.addBuffer(Buffer.from('a: 1'),'configs/a.yaml');
    zip.addBuffer(Buffer.from('result'),'results/run/data.txt');
    const reading=(async()=>{const chunks=[];for await(const b of zip.outputStream)chunks.push(b);return Buffer.concat(chunks);})();
    zip.end();const bytes=await reading;
    const transfer=await transfers.begin(actor,ws,'backup',bytes.length);
    await transfers.append(actor,{transferId:transfer.transferId,offset:0,base64:bytes.toString('base64')});
    const preview=await transfers.preview(actor,ws,transfer);
    const input={...transfer,files:preview.files.map(f=>({id:f.id,revision:f.expectedRevision}))};
    const attempts=Promise.allSettled([transfers.import(actor,ws,input),transfers.import(actor,ws,input)]);
    const outcome=await Promise.race([attempts,new Promise((_,reject)=>setTimeout(()=>reject(Error('Concurrent imports stalled')),3000))]);
    assert.equal(outcome[0].status,'fulfilled');
    assert.equal(outcome[1].status,'rejected');
    assert.equal(outcome[1].reason.code,'REVISION_CONFLICT');
    assert.equal((await ws.write('configs/after.yaml',Buffer.from('ok: true'),null)).kind,'file');
});
