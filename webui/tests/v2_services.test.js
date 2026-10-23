'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const os=require('node:os');
const path=require('node:path');
const express=require('express');
const {once}=require('node:events');
const {Accounts,passwordHash}=require('../automation/accounts');
const {Workspace,noLinks}=require('../automation/workspace');
const {AnalysisJobs}=require('../automation/jobs');
const {FileOperations}=require('../automation/operations');
const storage=require('../automation/storage');
async function temp(t){const dir=await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()),'omfdm-v2-'));t.after(async()=>{await noLinks(dir);assert.ok(path.basename(dir).startsWith('omfdm-v2-'));await fs.rm(dir,{recursive:true,force:true});});return dir;}
test('shared sessions enforce workspace ownership, CSRF, disable and restart persistence',async t=>{
    const base=await temp(t),directory=path.join(base,'private');await fs.mkdir(directory);
    const accounts=new Accounts(base,{mode:'shared'},{directory});
    const users={version:1,users:[{name:'alice',role:'user',revision:'a',password:await passwordHash('correct horse battery')},{name:'bob',role:'user',revision:'b',password:await passwordHash('second correct battery')}],owners:{wa:'alice',wb:'bob'}};
    await fs.writeFile(accounts.filename,JSON.stringify(users));
    const app=express();app.use(express.json());app.use((req,res,next)=>{req.socket.encrypted=true;next();});accounts.install(app);
    app.get('/private/:user',async(req,res)=>{try {const ws=new Workspace(base,req.params.user);res.json({user:ws.user});}catch(error){res.status(error.statusCode).json({error:error.message});}});
    app.post('/change',(req,res)=>res.json({success:true}));
    const server=app.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>new Promise(resolve=>server.close(resolve)));
    const url=`http://127.0.0.1:${server.address().port}`,origin=url.replace('http:','https:');
    assert.equal((await fetch(url+'/private/wa')).status,401);
    const login=await fetch(url+'/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json',Origin:origin},body:JSON.stringify({name:'alice',password:'correct horse battery',remember:true})});
    assert.equal(login.status,200);assert.match(login.headers.get('set-cookie'),/HttpOnly; Secure; SameSite=Strict/);
    const cookie=login.headers.get('set-cookie').split(';')[0];
    const identity=await (await fetch(url+'/api/auth/me',{headers:{Cookie:cookie}})).json();
    assert.deepEqual(identity.workspaces,['wa']);
    assert.equal((await fetch(url+'/private/wb',{headers:{Cookie:cookie}})).status,403);
    assert.equal((await fetch(url+'/private/wa',{headers:{Cookie:cookie}})).status,200);
    assert.equal((await fetch(url+'/change',{method:'POST',headers:{Cookie:cookie,Origin:origin}})).status,403);
    assert.equal((await fetch(url+'/change',{method:'POST',headers:{Cookie:cookie,Origin:origin,'X-CSRF-Token':identity.csrf}})).status,200);
    assert.equal((await fetch(url+'/private/wa?file=../../private/accounts.json',{headers:{Cookie:cookie}})).status,400);
    const restarted=new Accounts(base,{mode:'shared'},{directory});await restarted.ready;
    assert.equal((await restarted.identity({headers:{cookie}})).name,'alice');
    users.users[0].disabled=true;await fs.writeFile(accounts.filename,JSON.stringify(users));
    assert.equal((await fetch(url+'/private/wa',{headers:{Cookie:cookie}})).status,401);
});
test('queue freezes inputs, deduplicates, recovers held jobs and cancels without spawning',async t=>{
    const base=await temp(t),directory=path.join(base,'private');await fs.mkdir(directory);
    const jobs=new Map(),callbacks=new Map();let starts=0;
    const backend={jobs,maxConcurrentJobs:1,async submitAnalysisJob(input,options){starts++;jobs.set(options.jobId,{jobId:options.jobId,userId:input.userId,status:'running',process:{},log:[]});callbacks.set(options.jobId,options.onFinalized);}};
    const service=new AnalysisJobs(base,{directory},backend);await service.init();
    const ws=new Workspace(base,'wa');await ws.init();await fs.writeFile(await ws.resolve('configs/model.yaml',{create:true}),'coordinate_system: cartesian\n');await fs.writeFile(await ws.resolve('images/model.png',{create:true}),'image bytes');
    const input={config:{id:'configs/model.yaml',revision:(await ws.describe('configs/model.yaml')).revision},image:{id:'images/model.png',revision:(await ws.describe('images/model.png')).revision},idempotencyKey:'1'};
    const actor={id:'a',name:'test',user:'wa',guard:()=>{}};
    const first=await service.start(actor,input);assert.equal(first.status,'running');
    assert.equal((await service.start(actor,input)).jobId,first.jobId);assert.equal(starts,1);
    const second=await service.start(actor,{...input,idempotencyKey:'2'});assert.equal(second.status,'queued');
    await fs.writeFile(await ws.resolve('images/model.png'),'changed source');
    const record=service.records.find(r=>r.jobId===second.jobId);
    assert.equal(await fs.readFile(record.prepared.imagePath,'utf8'),'image bytes');
    const restored=new AnalysisJobs(base,{directory},{...backend,jobs:new Map()});await restored.init();
    assert.equal(restored.get('wa',first.jobId).status,'interrupted');assert.equal(restored.get('wa',second.jobId).status,'held');
    assert.equal((await restored.cancel('wa',second.jobId,()=>{})).status,'cancelled');assert.equal(starts,1);
    assert.throws(()=>restored.get('wb',second.jobId),/not found/);
    service.stopping=true;restored.stopping=true;
});
test('async deletion reports partial failures, persists outcomes and protects busy results',async t=>{
    const base=await temp(t),ws=new Workspace(base,'wa');await ws.init();
    await fs.mkdir(await ws.resolve('results/busy',{create:true}));await fs.writeFile(await ws.resolve('configs/ok.yaml',{create:true}),'one');await fs.writeFile(await ws.resolve('configs/changed.yaml',{create:true}),'two');
    const ids=['configs/ok.yaml','configs/changed.yaml','results/busy'];const files=[];
    for(const id of ids)files.push({id,revision:(await ws.describe(id)).revision});
    await fs.writeFile(await ws.resolve('configs/changed.yaml'),'different');
    const ops=new FileOperations(base,path.join(base,'private'),(user,id)=>id==='results/busy');await ops.ready;
    const record=await ops.start('wa',files);
    while(record.status==='running')await new Promise(resolve=>setTimeout(resolve,10));
    assert.deepEqual(record.results.map(r=>r.success),[true,false,false]);
    assert.throws(()=>ops.get('wb',record.id),/not found/);
    const restored=new FileOperations(base,path.join(base,'private'),()=>false);await restored.ready;
    assert.equal(restored.get('wa',record.id).results.length,3);
});

test('deletion progress stays in memory; only acceptance and completion write a journal',async t=>{
    const base=await temp(t),ws=new Workspace(base,'wa');await ws.init();const files=[];
    for(const name of ['a','b','c']){const id=`configs/${name}.yaml`;await ws.write(id,'data',null);files.push({id,revision:(await ws.describe(id)).revision});}
    const ops=new FileOperations(base,path.join(base,'private'),()=>false);await ops.ready;
    const save=ops.save.bind(ops),writes=[],pending=[];ops.save=record=>{writes.push({status:record.status,results:record.results.length});const saved=save(record);pending.push(saved);return saved;};
    const record=await ops.start('wa',files);
    while(record.status==='running')await new Promise(resolve=>setTimeout(resolve,10));
    await Promise.all(pending);
    assert.deepEqual(writes,[{status:'running',results:0},{status:'complete',results:3}]);
    const interrupted={...record,id:require('node:crypto').randomUUID(),status:'running',results:[]};await save(interrupted);
    const restored=new FileOperations(base,path.join(base,'private'),()=>false);await restored.ready;
    const recovered=restored.get('wa',interrupted.id);
    assert.equal(recovered.status,'interrupted');assert.deepEqual(recovered.targets,files.map(file=>file.id));
    assert.match(recovered.message,/Some targets may already be removed/);
});
test('external results root is used consistently by workspace reads and listings',async t=>{
    const base=await temp(t),root=path.join(base,'external');storage.configure(base,{resultsRoot:root});
    const ws=new Workspace(base,'wa');await ws.init();const filename=await ws.resolve('results/run/file.txt',{create:true});
    assert.equal(filename,path.join(root,'wa','run','file.txt'));await fs.mkdir(path.dirname(filename));await fs.writeFile(filename,'retained');
    assert.equal((await ws.read('results/run/file.txt',{text:true})).content,'retained');
    assert.ok((await ws.list({category:'results'})).items.some(f=>f.name==='run'));
});

test('large result deletion is independent of the bounded archive walk limit',async t=>{
    const base=await temp(t),ws=new Workspace(base,'wa');await ws.init();
    const dir=await ws.resolve('results/large/Az',{create:true});await fs.mkdir(dir,{recursive:true});
    for(let start=0;start<10001;start+=100)await Promise.all(Array.from({length:Math.min(100,10001-start)},(_,i)=>fs.writeFile(path.join(dir,`step_${start+i}.csv`),'1')));
    await assert.rejects(ws.walk('results/large'),{code:'LIMIT_EXCEEDED'});
    const meta=await ws.describe('results/large');assert.equal(meta.count,10001);
    const ops=new FileOperations(base,path.join(base,'private'),()=>false);await ops.ready;
    const record=await ops.start('wa',[{id:'results/large',revision:meta.revision}]);
    while(record.status==='running')await new Promise(resolve=>setTimeout(resolve,10));
    assert.equal(record.status,'complete');assert.equal(record.results[0].success,true,JSON.stringify(record));
    await assert.rejects(ws.resolve('results/large'),{code:'ENOENT'});
});

test('queue rotates across three workspaces and retries after an account is enabled',async t=>{
    const base=await temp(t),directory=path.join(base,'private');await fs.mkdir(directory);
    const launched=[],callbacks=new Map();let enabled=false;
    const backend={jobs:new Map(),maxConcurrentJobs:1,workspaceEnabled:user=>user!=='wb'||enabled,
        async submitAnalysisJob(input,options){
            const live={jobId:options.jobId,userId:input.userId,status:'running',process:{},log:[]};
            backend.jobs.set(live.jobId,live);callbacks.set(live.jobId,options.onFinalized);launched.push(live);
        }};
    const service=new AnalysisJobs(base,{directory},backend);await service.init();service.stopping=true;
    t.after(()=>service.stop());
    for(const user of ['wa','wb','wc']){
        const ws=new Workspace(base,user);await ws.init();
        await fs.writeFile(await ws.resolve('configs/model.yaml',{create:true}),'coordinate_system: cartesian\n');
        await fs.writeFile(await ws.resolve('images/model.png',{create:true}),'image');
        const input={config:{id:'configs/model.yaml',revision:(await ws.describe('configs/model.yaml')).revision},image:{id:'images/model.png',revision:(await ws.describe('images/model.png')).revision}};
        for(let i=0;i<2;i++)await service.start({id:user,name:'test',user,guard:()=>{}},{...input,idempotencyKey:String(i)});
    }
    const complete=async()=>{const job=launched.at(-1);job.status='completed';await callbacks.get(job.jobId)(job);await service.pump();};
    service.stopping=false;await service.pump();assert.equal(launched.at(-1).userId,'wa');
    enabled=true;await complete();assert.equal(launched.at(-1).userId,'wb');
    await complete();assert.equal(launched.at(-1).userId,'wc');
    await complete();assert.equal(launched.at(-1).userId,'wa');
    enabled=false;await complete();assert.equal(launched.at(-1).userId,'wc');
    await complete();assert.equal(launched.length,5);
    enabled=true;
    const deadline=Date.now()+4000;
    while(launched.length<6&&Date.now()<deadline)await new Promise(resolve=>setTimeout(resolve,25));
    assert.equal(launched.length,6);assert.equal(launched.at(-1).userId,'wb');
});

test('result backup round trip retains frozen inputs at an external results root',async t=>{
    const base=await temp(t);storage.configure(base,{resultsRoot:path.join(base,'external-results')});
    const {Transfers}=require('../automation/transfers');const transfers=new Transfers();t.after(()=>transfers.close());
    const ws=new Workspace(base,'wa');await ws.init();
    await ws.write('results/run/.inputs/model.png',Buffer.from('original input bytes'),null);
    await ws.write('results/run/run-inputs.json',Buffer.from('{"version":1}'),null);
    const hidden=await ws.resolve('results/run/.private/note',{create:true,internal:true});await fs.mkdir(path.dirname(hidden));await fs.writeFile(hidden,'not a snapshot');
    const actor={id:'test',user:'wa',guard:()=>{}};
    const output=await transfers.export(actor,ws,{files:[{id:'results/run',revision:(await ws.describe('results/run')).revision}]});
    const bytes=await fs.readFile(transfers.get(actor,output.transferId).filename);
    const target=new Workspace(base,'wb');await target.init();const other={id:'other',user:'wb',guard:()=>{}};
    const upload=await transfers.begin(other,target,'backup',bytes.length);
    await transfers.append(other,{transferId:upload.transferId,offset:0,base64:bytes.toString('base64')});
    const preview=await transfers.preview(other,target,upload);
    assert.deepEqual(preview.files.map(f=>f.id).sort(),['results/run/.inputs/model.png','results/run/run-inputs.json']);
    await transfers.import(other,target,{...upload,files:preview.files.map(f=>({id:f.id,revision:f.expectedRevision}))});
    assert.equal((await target.read('results/run/.inputs/model.png',{text:true})).content,'original input bytes');
    await assert.rejects(()=>target.resolve('results/.mcp-transfers/private'),{code:'INVALID_PATH'});
    await assert.rejects(()=>target.resolve('configs/run/.inputs/private'),{code:'INVALID_PATH'});
});
