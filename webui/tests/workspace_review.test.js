'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os');
const {AnalysisJobs}=require('../automation/jobs');
const {Workspace,noLinks}=require('../automation/workspace');
const {context}=require('../automation/identity-context');
async function fixture(t) {
    const base=await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()),'omfdm-review-'));
    t.after(async()=>{await noLinks(base);await fs.rm(base,{recursive:true,force:true});});
    return base;
}
test('history deletion is scoped to finished jobs and preserves result files across restart',async t=>{
    const base=await fixture(t), backend={jobs:new Map()}, jobs=new AnalysisJobs(base,{directory:base},backend);
    await jobs.init();jobs.stopping=true;
    const recent=new Date(Date.now()-1000).toISOString(), old=new Date(Date.now()-40*86400000).toISOString();
    jobs.records=[
        {user:'alice',jobId:'recent',status:'completed',created:recent,finished:recent,resultId:'results/run'},
        {user:'alice',jobId:'old',status:'failed',created:old,finished:old},
        {user:'alice',jobId:'active',status:'running',created:old},
        {user:'alice',jobId:'held',status:'held',created:old},
        {user:'bob',jobId:'other',status:'completed',created:recent,finished:recent},
        {user:'alice',jobId:'live',status:'completed',created:recent,finished:recent}
    ];
    backend.jobs.set('live',{status:'running'});
    const ws=new Workspace(base,'alice');await ws.init();await ws.write('results/run/data.txt',Buffer.from('keep'),null);
    await jobs.save();
    assert.equal((await jobs.deleteHistory('alice','day')).deleted,1);
    assert.equal((await jobs.deleteHistory('alice','older-month')).deleted,1);
    assert.deepEqual(jobs.records.map(j=>j.jobId).sort(),['active','held','live','other']);
    assert.equal((await ws.read('results/run/data.txt',{text:true})).content,'keep');
    await assert.rejects(jobs.deleteHistory('alice','invalid'),{code:'INVALID_INPUT'});
    await assert.rejects(context.run({role:'user',workspaces:['bob']},()=>jobs.deleteHistory('alice','all')),{code:'FORBIDDEN'});
    const saved=JSON.parse(await fs.readFile(path.join(base,'jobs.json'),'utf8'));
    assert.deepEqual(saved.map(j=>j.jobId).sort(),['active','held','live','other']);
    const restarted=new AnalysisJobs(base,{directory:base},{jobs:new Map()});await restarted.init();
    assert.ok(restarted.records.find(j=>j.jobId==='active').finished);
    assert.equal(restarted.records.find(j=>j.jobId==='held').status,'held');
});
test('WebUI and MCP submissions receive distinct result names and retain idempotency',async t=>{
    const base=await fixture(t),jobs=new AnalysisJobs(base,{directory:base},{jobs:new Map()});await jobs.init();jobs.stopping=true;
    const ws=new Workspace(base,'alice');await ws.init();
    const config=await ws.write('configs/a.yaml',Buffer.from('variables: {}'),null),image=await ws.write('images/a.png',Buffer.from('image'),null);
    const input={config:{id:config.id,revision:config.revision},image:{id:image.id,revision:image.revision},idempotencyKey:'one'};
    const gui={id:'gui:alice',name:'WebUI',user:'alice',guard(){}},mcp={id:'agent-id',name:'Agent',user:'alice',guard(){}};
    const a=await jobs.start(gui,input),b=await jobs.start(mcp,input);
    assert.match(a.resultId,/^results\/output_webui_/);assert.match(b.resultId,/^results\/output_mcp_/);
    assert.equal((await jobs.start(gui,input)).jobId,a.jobId);
});
test('GUI rename checks revisions and updates result history without overwriting',async t=>{
    const base=await fixture(t),express=require('express'),app=express();app.use(express.json());
    app.use((req,res,next)=>context.run({role:'user',workspaces:['alice']},next));
    const jobs=new AnalysisJobs(base,{directory:base},{jobs:new Map()});await jobs.init();jobs.stopping=true;
    jobs.records=[{user:'alice',jobId:'j',status:'completed',resultId:'results/old',resultPath:'outputs/alice/old'},
        {user:'bob',jobId:'other',status:'completed',resultId:'results/old',resultPath:'outputs/bob/old'}];await jobs.save();
    const ws=new Workspace(base,'alice');await ws.init();await ws.write('results/old/data.txt',Buffer.from('data'),null);
    const meta=await ws.describe('results/old');
    require('../automation/gui-files').installFileRoutes(app,base,()=>false,(...args)=>jobs.renameResult(...args));
    const server=await new Promise(resolve=>{const s=app.listen(0,'127.0.0.1',()=>resolve(s));});
    t.after(()=>new Promise(resolve=>server.close(resolve)));
    const endpoint=`http://127.0.0.1:${server.address().port}/api/user-files/rename`;
    const send=body=>fetch(endpoint,{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
    const body={userId:'alice',category:'results',name:'old',newName:'new',revision:meta.revision};
    const denied=await send({...body,userId:'bob'});assert.equal(denied.status,403);
    const stale=await send({...body,revision:'0'.repeat(64)});assert.equal(stale.status,409);
    await ws.write('results/occupied/data.txt',Buffer.from('other'),null);
    const conflict=await send({...body,newName:'occupied'});assert.equal(conflict.status,409);
    const result=await send(body);assert.equal(result.status,200);assert.equal(jobs.records[0].resultId,'results/new');
    assert.equal(jobs.get('alice','j').resultPath,'outputs/alice/new');
    assert.equal(jobs.get('bob','other').resultPath,'outputs/bob/old');
    const restarted=new AnalysisJobs(base,{directory:base},{jobs:new Map()});await restarted.init();
    assert.equal(restarted.get('alice','j').resultPath,'outputs/alice/new');
    assert.equal((await ws.read('results/new/data.txt',{text:true})).content,'data');
    const invalid=await send({...body,name:'new',newName:'../escape',revision:(await ws.describe('results/new')).revision});
    assert.equal(invalid.status,400);
});

test('result history rename rolls back both references when persistence fails',async t=>{
    const base=await fixture(t),jobs=new AnalysisJobs(base,{directory:base},{jobs:new Map()});await jobs.init();jobs.stopping=true;
    jobs.records=[{user:'alice',jobId:'j',status:'completed',resultId:'results/old',resultPath:'outputs/alice/old'},
        {user:'alice',jobId:'legacy',status:'completed',resultId:'results/old'}];await jobs.save();
    const before=structuredClone(jobs.records);
    jobs.save=async()=>{throw Error('Simulated persistence failure');};
    await assert.rejects(jobs.renameResult('alice','results/old','results/new'),/persistence failure/);
    assert.deepEqual(jobs.records,before);
    assert.deepEqual(JSON.parse(await fs.readFile(jobs.filename,'utf8')),before);
});

test('renamed library names remain editable and unsafe paths are rejected',async t=>{
    const {spawn}=require('node:child_process'),{once}=require('node:events');
    const f=await require('./console_fixture.cjs').createFixture();
    const child=spawn(f.executable,f.args,{cwd:f.base,env:f.env,windowsHide:true,stdio:'ignore'});
    t.after(async()=>{
        if(child.exitCode===null&&child.signalCode===null){const closed=once(child,'close');child.kill();await closed;}
        await noLinks(f.root);assert.ok(path.basename(f.root).startsWith('omfdm-console-'));await fs.rm(f.root,{recursive:true,force:true});
    });
    const endpoint=`http://127.0.0.1:${f.port}`;
    const send=(url,method='GET',body)=>fetch(endpoint+url,{method,headers:{'content-type':'application/json'},...(body?{body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(10000)});
    const deadline=Date.now()+20000;
    while(true){try{if((await send('/api/auth/me')).ok)break;}catch{}assert.ok(Date.now()<deadline&&child.exitCode===null,'Isolated server startup');await new Promise(r=>setTimeout(r,50));}
    for(const [index,name] of ['copper+core.yaml','material_\u9285.yml'].entries()) {
        const oldName=`old_${index}.yaml`;
        const created=await send(`/api/material-libraries/${oldName}`,'PUT',{userId:'alice',content:'material_presets: {}',expectedRevision:null});
        assert.equal(created.status,200);const original=await created.json();
        const renamed=await send('/api/user-files/rename','PUT',{userId:'alice',category:'libraries',name:oldName,newName:name,revision:original.revision});
        assert.equal(renamed.status,200);await renamed.json();
        const read=await send(`/api/material-libraries/${encodeURIComponent(name)}?userId=alice`);
        assert.equal(read.status,200);const revision=read.headers.get('x-file-revision');await read.text();
        const saved=await send(`/api/material-libraries/${encodeURIComponent(name)}`,'PUT',{userId:'alice',content:'material_presets: {}\n# edited',expectedRevision:revision});
        assert.equal(saved.status,200,await saved.text());
        const check=await send(`/api/material-libraries/${encodeURIComponent(name)}?userId=alice`);assert.match(await check.text(),/# edited/);
        const stale=await send(`/api/material-libraries/${encodeURIComponent(name)}`,'PUT',{userId:'alice',content:'material_presets: {}',expectedRevision:revision});
        assert.equal(stale.status,409);await stale.json();
    }
    for(const name of ['nested%2Fevil.yaml','nested%5Cevil.yaml','%25escape.yaml','CON.yaml']) {
        const unsafe=await send(`/api/material-libraries/${name}`,'PUT',{userId:'alice',content:'material_presets: {}',expectedRevision:null});
        assert.equal(unsafe.status,400,name);await unsafe.json();
    }
});
