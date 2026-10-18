'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const {EventEmitter}=require('node:events');
const {Client,StreamableHTTPClientTransport}=require('@modelcontextprotocol/client');
const {StdioClientTransport}=require('@modelcontextprotocol/client/stdio');
const {Workspace,noLinks}=require('../automation/workspace');
const {AutomationRuntime}=require('../automation/runtime');
const {execute}=require('../automation/tools');
const {localRequest}=require('../automation/access');
const Jimp=require('jimp');
async function fixture(t) {
    const temp=await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()),'omfdm-mcp-'));
    const base=path.join(temp,'workspace'),directory=path.join(temp,'private');
    await fs.mkdir(base);let starts=0;
    const backend={jobs:new Map(),isChildRunning:p=>p.running,terminateChildProcess:async p=>{p.running=false;},mergeLibraryIntoConfig:(a)=>a,
        async submitAnalysisJob(input,options){
            starts++;const process=new EventEmitter();process.running=true;
            const job={jobId:options.jobId,userId:input.userId,status:'running',process,progress:0,log:[],...options.prepared};
            backend.jobs.set(job.jobId,job);return {jobId:job.jobId};
        }};
    const runtime=new AutomationRuntime(base,backend,{directory});await runtime.init();assert.equal(runtime.ready,true,runtime.error);
    await runtime.access.change(s=>{s.enabled=true;});
    const a=await runtime.access.create({name:'Agent A',user:'user_a',permissions:['read','edit','images','replace','delete','run','cancel']});
    const b=await runtime.access.create({name:'Agent B',user:'user_b',permissions:['read']});
    const actor=runtime.access.actor(a.id),other=runtime.access.actor(b.id);
    const wa=new Workspace(base,actor.user,{guard:actor.guard,busy:id=>runtime.jobs.busy(actor.user,id)}),wb=new Workspace(base,other.user);
    await wa.init();await wb.init();
    t.after(async()=>{await runtime.close();await noLinks(temp);assert.ok(path.basename(temp).startsWith('omfdm-mcp-'));await fs.rm(temp,{recursive:true,force:true});});
    return {temp,base,runtime,actor,other,wa,wb,a,b,backend,starts:()=>starts};
}
test('trusted browsers persist with hashed tokens, expire, and can be forgotten without revoking agents',async t=>{
    const f=await fixture(t);
    const {Access,hash}=require('../automation/access');
    const a=f.runtime.access;
    const temporary=a.pair(a.pairingCode);
    const remembered=a.pair(a.pairingCode,true);
    await a.saveBrowsers();await a.flush();
    const raw=await fs.readFile(a.filename,'utf8');
    assert.ok(!raw.includes(remembered.token));
    const restored=new Access(f.base,{directory:a.directory});await restored.init();
    const request=token=>({socket:{remoteAddress:'127.0.0.1'},headers:{host:'localhost:3000',cookie:`omfdm_admin=${token}`,'x-omfdm-csrf':remembered.csrf}});
    assert.throws(()=>restored.admin(request(temporary.token)),{code:'ADMIN_REQUIRED'});
    assert.equal(restored.admin(request(remembered.token),true).remember,true);
    assert.throws(()=>restored.admin({...request(remembered.token),headers:{...request(remembered.token).headers,'x-omfdm-csrf':'wrong'}},true),{code:'CSRF'});
    assert.equal(restored.state.enabled,true);
    await restored.forgetBrowser(request(remembered.token));
    assert.throws(()=>restored.admin(request(remembered.token)),{code:'ADMIN_REQUIRED'});
    assert.equal(restored.state.connections.find(c=>c.id===f.a.id).revoked,false);
    const expired=restored.pair(restored.pairingCode,true);
    restored.sessions.get(hash(expired.token)).expires=Date.now()-1;
    assert.throws(()=>restored.admin(request(expired.token)),{code:'ADMIN_REQUIRED'});
    await restored.saveBrowsers();await restored.flush();
    assert.equal(restored.state.trustedBrowsers.length,0);
});

test('management cookies, solver settings and logout require paired CSRF-protected access',async t=>{
    const f=await fixture(t),express=require('express'),app=express();app.use(express.json());f.runtime.install(app);
    const server=await new Promise(resolve=>{const s=app.listen(0,'127.0.0.1',()=>resolve(s));});
    t.after(()=>new Promise(resolve=>server.close(resolve)));
    const url=`http://127.0.0.1:${server.address().port}`;
    const pair=await fetch(url+'/api/mcp/pair',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:f.runtime.access.pairingCode,remember:true})});
    const cookie=pair.headers.get('set-cookie');
    assert.match(cookie,/HttpOnly/);assert.match(cookie,/SameSite=Strict/);assert.match(cookie,/Max-Age=2592000/);
    const {csrf}=await pair.json();const headers={'Content-Type':'application/json',Cookie:cookie.split(';')[0],'X-Omfdm-Csrf':csrf};
    assert.equal((await fetch(url+'/api/mcp/solver',{method:'PUT',headers:{...headers,'X-Omfdm-Csrf':'wrong'},body:JSON.stringify({path:process.execPath})})).status,403);
    const saved=await fetch(url+'/api/mcp/solver',{method:'PUT',headers,body:JSON.stringify({path:process.execPath})});
    assert.equal(saved.status,200);assert.equal((await saved.json()).restartRequired,true);
    assert.equal(f.runtime.access.state.solverPath,process.execPath);
    assert.equal((await fetch(url+'/api/mcp/forget-browser',{method:'POST',headers,body:'{}'})).status,200);
    assert.equal((await fetch(url+'/api/mcp/status',{headers})).status,401);
});

test('MCP listener and saved solver setting restore on runtime restart',async t=>{
    const f=await fixture(t);
    f.runtime.access.state.port=0;await f.runtime.reconfigure();
    const port=f.runtime.listener.address().port;
    await f.runtime.access.change(s=>{s.port=port;s.solverPath=process.execPath;});
    await f.runtime.close();
    let configured;
    const restored=new AutomationRuntime(f.base,{...f.backend,configureSolver:p=>{configured=p;}},{directory:f.runtime.access.directory});
    try {await restored.init();assert.equal(restored.status().listening,true);assert.equal(configured,process.execPath);assert.equal(restored.access.state.connections.length,2);}
    finally {await restored.close();}
});

test('workspace IDs reject traversal, absolute paths, aliases, device names and links',async t=>{
    const f=await fixture(t);
    for(const id of ['../user_b/x','configs/../user_b/x','configs/C:/x','configs/\\server\\x','configs/a:secret','configs/%2e%2e/x','configs/NUL.txt','configs/a.','configs/.hidden','configs//x'])await assert.rejects(f.wa.resolve(id,{create:true}));
    await fs.symlink(path.join(f.base,'configs','user_b'),path.join(f.base,'configs','user_a','escape'),'junction');
    await assert.rejects(f.wa.resolve('configs/escape/secret.yaml',{create:true}),{code:'UNSAFE_PATH'});
    await assert.rejects(f.wa.list(),{code:'UNSAFE_PATH'});
});
test('workspace writes use compare-and-swap and never overwrite another revision',async t=>{
    const f=await fixture(t);
    const first=await f.wa.write('configs/a.yaml',Buffer.from('a: 1'),null);
    await assert.rejects(f.wa.write('configs/a.yaml',Buffer.from('a: 2'),null),{code:'REVISION_CONFLICT'});
    const second=await f.wa.write(first.id,Buffer.from('a: 2'),first.revision);
    await assert.rejects(f.wa.remove(first.id,first.revision),{code:'REVISION_CONFLICT'});
    assert.equal((await f.wa.read(first.id,{text:true})).content,'a: 2');
    await f.wa.remove(second.id,second.revision);
    assert.equal((await f.wa.list()).total,0);
    const source=await f.wa.write('configs/read.yaml',Buffer.from('old'),null);
    let unlock,entered;const gate=new Promise(resolve=>{unlock=resolve;}),ready=new Promise(resolve=>{entered=resolve;});
    const reader=new Workspace(f.base,f.actor.user),describe=reader.describe.bind(reader);
    reader.describe=async id=>{const meta=await describe(id);entered();await gate;return meta;};
    const reading=reader.read(source.id,{text:true});await ready;
    const writing=f.wa.write(source.id,Buffer.from('new'),source.revision);unlock();
    const data=await reading;await writing;
    assert.equal(data.content,'old');assert.equal(data.revision,source.revision);
});
test('user B sees neither files nor jobs of user A',async t=>{
    const f=await fixture(t);await f.wa.write('configs/private.yaml',Buffer.from('secret: yes'),null);
    assert.equal((await f.wb.list()).total,0);
    await assert.rejects(execute('files_read',{id:'configs/private.yaml',text:true},f.other,f.runtime),{code:'ENOENT'});
    assert.throws(()=>f.runtime.jobs.get(f.other.user,'unknown'),{code:'NOT_FOUND'});
});
test('revocation and OFF are checked again at mutation time',async t=>{
    const f=await fixture(t);const stale=f.actor;
    await f.runtime.access.change(s=>{s.enabled=false;});
    await assert.rejects(f.wa.write('configs/a.yaml',Buffer.from('a: 1'),null),{code:'UNAUTHORIZED'});
    await f.runtime.access.change(s=>{s.enabled=true;});
    await f.runtime.access.update(stale.id,{revoked:true});
    assert.throws(()=>stale.guard('read'),{code:'UNAUTHORIZED'});
});
test('read-only tokens cannot edit, start, cancel, delete or replace',async t=>{
    const f=await fixture(t);for(const p of ['edit','run','cancel','delete','replace','images'])assert.throws(()=>f.other.guard(p),{code:'FORBIDDEN'});
});
test('request guard rejects remote peers, forged hosts and cross-site browser requests',()=>{
    const req={socket:{remoteAddress:'127.0.0.1'},headers:{host:'127.0.0.1:3001'}};
    localRequest(req);
    for(const patch of [{socket:{remoteAddress:'10.0.0.1'}},{headers:{host:'evil.example'}},{headers:{...req.headers,origin:'https://evil.example'}},{headers:{...req.headers,'sec-fetch-site':'cross-site'}}])assert.throws(()=>localRequest({...req,...patch}));
});
test('pairing is one-use and management requires CSRF',async t=>{
    const f=await fixture(t),code=f.runtime.access.pairingCode;
    const session=f.runtime.access.pair(code);assert.throws(()=>f.runtime.access.pair(code),{code:'UNAUTHORIZED'});
    const req={socket:{remoteAddress:'127.0.0.1'},headers:{host:'localhost:3000',cookie:`omfdm_admin=${session.token}`}};
    assert.throws(()=>f.runtime.access.admin(req,true),{code:'CSRF'});
    req.headers['x-omfdm-csrf']=session.csrf;f.runtime.access.admin(req,true);
});
test('job input snapshot, idempotency, output protection and OFF preserve running jobs',async t=>{
    const f=await fixture(t);
    const config=await f.wa.write('configs/model.yaml',Buffer.from('coordinate_system: cartesian'),null);
    const img=await new Jimp(4,4,0xffffffff).getBufferAsync(Jimp.MIME_PNG);
    const image=await f.wa.write('images/model.png',img,null,{permission:'images'});
    const input={config:{id:config.id,revision:config.revision},image:{id:image.id,revision:image.revision},idempotencyKey:'one'};
    const first=await f.runtime.jobs.start(f.actor,input);const second=await f.runtime.jobs.start(f.actor,input);
    assert.equal(first.jobId,second.jobId);assert.equal(f.starts(),1);
    await assert.rejects(f.runtime.jobs.start(f.actor,{...input,image:{...input.image,revision:'a'.repeat(64)}}),{code:'IDEMPOTENCY_CONFLICT'});
    const meta=await f.wa.describe(first.resultId);await assert.rejects(f.wa.remove(first.resultId,meta.revision),{code:'BUSY'});
    assert.throws(()=>f.runtime.jobs.get(f.other.user,first.jobId),{code:'NOT_FOUND'});
    const snapshot=f.backend.jobs.get(first.jobId).configPath;
    await f.wa.write(config.id,Buffer.from('coordinate_system: polar'),config.revision);
    assert.match(await fs.readFile(snapshot,'utf8'),/cartesian/);
    await f.runtime.access.change(s=>{s.enabled=false;});await f.runtime.reconfigure();
    assert.equal(f.backend.jobs.get(first.jobId).process.running,true);
    await f.runtime.jobs.cancel(f.actor.user,first.jobId,()=>{});assert.equal(f.backend.jobs.get(first.jobId).process.running,false);
});
test('YAML references outside the workspace are rejected before solver spawn',async t=>{
    const f=await fixture(t);const image=await f.wa.write('images/m.png',await new Jimp(2,2,0xffffffff).getBufferAsync(Jimp.MIME_PNG),null,{permission:'images'});
    for(const reference of ['../user_b/x.bin','C:/Windows/a.bin','results/../../secret']){
        const config=await f.wa.write(`configs/${Math.random().toString(36).slice(2)}.yaml`,Buffer.from(`nonlinear_solver:\n  initial_az_path: ${JSON.stringify(reference)}`),null);
        await assert.rejects(f.runtime.jobs.start(f.actor,{config:{id:config.id,revision:config.revision},image:{id:image.id,revision:image.revision},idempotencyKey:reference}));
    }assert.equal(f.starts(),0);
});
test('Streamable HTTP serves modern and legacy SDK clients; tokens and strict arguments enforced',async t=>{
    const f=await fixture(t);f.runtime.access.state.port=0;await f.runtime.reconfigure();const port=f.runtime.listener.address().port;
    f.runtime.access.state.port=port;
    const url=new URL(`http://127.0.0.1:${port}/mcp`);
    assert.equal((await fetch(url,{method:'POST',body:'{}'})).status,401);
    for(const mode of ['auto','legacy']) {
        const client=new Client({name:'test',version:'1'},{versionNegotiation:{mode}});
        await client.connect(new StreamableHTTPClientTransport(url,{requestInit:{headers:{Authorization:`Bearer ${f.a.token}`}}}));
        try {
            const tools=await client.listTools();assert.ok(tools.tools.some(t=>t.name==='analysis_start'));
            const result=await client.callTool({name:'workspace_status',arguments:{}});
            assert.equal(result.isError,undefined);assert.equal(result.structuredContent.workspace,'user_a');
            const template=await client.callTool({name:'yaml_template',arguments:{kind:'cartesian'}});
            assert.notEqual(template.isError,true);
            const inspection=await client.callTool({name:'yaml_inspect',arguments:{content:template.structuredContent.content}});
            assert.notEqual(inspection.isError,true);assert.equal(inspection.structuredContent.valid,true);
            const written=await client.callTool({name:'yaml_write',arguments:{id:`configs/${mode}.yaml`,content:template.structuredContent.content,expectedRevision:null}});
            assert.notEqual(written.isError,true);assert.match(written.structuredContent.revision,/^[a-f0-9]{64}$/);
            const denied=await client.callTool({name:'files_list',arguments:{userId:'user_b'}});assert.equal(denied.isError,true);
        }finally{await client.close();}
    }
});
test('archive roundtrip is scoped and bounded; uploads cannot be taken over by another connection',async t=>{
    const f=await fixture(t);const meta=await f.wa.write('configs/a.yaml',Buffer.from('coordinate_system: cartesian'),null);
    const exported=await f.runtime.transfers.export(f.actor,f.wa,{files:[{id:meta.id,revision:meta.revision}]});
    assert.throws(()=>f.runtime.transfers.get(f.other,exported.transferId),{code:'NOT_FOUND'});
    const bytes=await f.runtime.transfers.read(f.actor,{transferId:exported.transferId,offset:0});
    const upload=await f.runtime.transfers.begin(f.actor,f.wa,'backup',Buffer.from(bytes.base64,'base64').length);
    await f.runtime.transfers.append(f.actor,{transferId:upload.transferId,offset:0,base64:bytes.base64});
    const preview=await f.runtime.transfers.preview(f.actor,f.wa,{transferId:upload.transferId});assert.equal(preview.files[0].id,meta.id);
    const result=await f.runtime.transfers.import(f.actor,f.wa,{transferId:upload.transferId,files:[{id:meta.id,revision:meta.revision}]});assert.deepEqual(result.applied,[meta.id]);
});
test('stdio bridge uses the same server and exits without cancelling a running analysis',async t=>{
    const f=await fixture(t);f.runtime.access.state.port=0;await f.runtime.reconfigure();
    const config=await f.wa.write('configs/model.yaml',Buffer.from('coordinate_system: cartesian'),null);
    const image=await f.wa.write('images/model.png',await new Jimp(2,2,0xffffffff).getBufferAsync(Jimp.MIME_PNG),null,{permission:'images'});
    const job=await f.runtime.jobs.start(f.actor,{config:{id:config.id,revision:config.revision},image:{id:image.id,revision:image.revision},idempotencyKey:'stdio-job'});
    f.runtime.access.state.port=f.runtime.listener.address().port;await f.runtime.access.change(()=>{});
    const bridge=require.resolve('../automation/bridge');
    const script=`require(${JSON.stringify(bridge)}).start(${JSON.stringify(f.base)},['--connection',${JSON.stringify(f.a.id)}],{directory:${JSON.stringify(f.runtime.access.directory)}}).catch(()=>process.exit(1))`;
    const client=new Client({name:'stdio-test',version:'1'});
    const transport=new StdioClientTransport({command:process.execPath,args:['-e',script],stderr:'pipe'});
    await client.connect(transport);
    try {
        const result=await client.callTool({name:'workspace_status',arguments:{}});assert.equal(result.structuredContent.workspace,'user_a');
        await f.runtime.access.update(f.a.id,{revoked:true});
        assert.equal((await client.callTool({name:'files_list',arguments:{}})).isError,true);
    }finally{await client.close();}
    assert.equal(f.runtime.listener.listening,true);
    assert.equal(f.starts(),1);assert.equal(f.backend.jobs.get(job.jobId).process.running,true);
});

async function zipBytes(name,mode=0o100644) {
    const zip=new (require('yazl').ZipFile)();zip.addBuffer(Buffer.from('value: 1'),name,{mode});
    const chunks=[];const reading=(async()=>{for await(const b of zip.outputStream)chunks.push(b);return Buffer.concat(chunks);})();
    zip.end();return reading;
}
async function uploadZip(f,bytes) {
    const transfer=await f.runtime.transfers.begin(f.actor,f.wa,'backup',bytes.length);
    await f.runtime.transfers.append(f.actor,{transferId:transfer.transferId,offset:0,base64:bytes.toString('base64')});
    return transfer;
}
test('ZIP links, traversal directories, ADS and foreign roots are rejected without writes',async t=>{
    const f=await fixture(t);
    const traversal=await zipBytes('configs/xx/secret.yaml');
    for(let at=traversal.indexOf('configs/xx/');at>=0;at=traversal.indexOf('configs/xx/',at+1))traversal.write('configs/../',at);
    for(const bytes of [traversal,await zipBytes('configs/link.yaml',0o120777),await zipBytes('user_b/secret.yaml'),await zipBytes('configs/a:ads.yaml')]) {
        const transfer=await uploadZip(f,bytes);
        await assert.rejects(f.runtime.transfers.preview(f.actor,f.wa,transfer));
        await f.runtime.transfers.release(f.actor,transfer.transferId);
    }
    assert.equal((await f.wa.list()).total,0);
});
test('GUI and MCP share revisions and block active-output deletion',async t=>{
    const f=await fixture(t),express=require('express');const app=express();app.use(express.json());
    let busy=false;require('../automation/gui-files').installFileRoutes(app,f.base,()=>busy);
    const server=await new Promise(resolve=>{const s=app.listen(0,'127.0.0.1',()=>resolve(s));});
    t.after(()=>new Promise(resolve=>server.close(resolve)));
    const url=`http://127.0.0.1:${server.address().port}`;
    const meta=await f.wa.write('configs/a.yaml',Buffer.from('a: 1'),null);
    const remove=revision=>fetch(url+'/api/user-files',{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify({userId:'user_a',category:'configs',name:'a.yaml',revision})});
    assert.equal((await remove()).status,428);
    const updated=await f.wa.write(meta.id,Buffer.from('a: 2'),meta.revision);
    assert.equal((await remove(meta.revision)).status,409);
    busy=true;assert.equal((await remove(updated.revision)).status,409);
    busy=false;assert.equal((await remove(updated.revision)).status,200);
});
test('GUI ZIP inspection is non-mutating and stale import commits conflict',async t=>{
    const f=await fixture(t),express=require('express'),multer=require('multer');const app=express();app.use(express.json());
    require('../automation/gui-backup').installBackupRoutes(app,f.base,()=>f.runtime,multer({dest:f.temp,limits:{fileSize:1024*1024}}));
    const server=await new Promise(resolve=>{const s=app.listen(0,'127.0.0.1',()=>resolve(s));});
    t.after(()=>new Promise(resolve=>server.close(resolve)));
    const url=`http://127.0.0.1:${server.address().port}`;
    const old=await f.wa.write('configs/a.yaml',Buffer.from('a: 1'),null);
    const form=new FormData();form.set('backup',new Blob([await zipBytes(old.id)]),'backup.zip');form.set('userId','user_a');
    const preview=await (await fetch(url+'/api/import',{method:'POST',body:form})).json();assert.equal(preview.success,true);
    assert.equal((await f.wa.read(old.id,{text:true})).content,'a: 1');
    await f.wa.write(old.id,Buffer.from('a: 2'),old.revision);
    const result=await fetch(url+'/api/import/commit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({userId:'user_a',transferId:preview.transferId,files:preview.files.map(file=>({id:file.id,revision:file.expectedRevision}))})});
    assert.equal(result.status,409);assert.equal((await f.wa.read(old.id,{text:true})).content,'a: 2');
});
test('durable records mark interrupted jobs without spawning on restart',async t=>{
    const f=await fixture(t);f.runtime.jobs.records.push({jobId:'test',user:'user_a',status:'running'});await f.runtime.jobs.save();
    const {AnalysisJobs}=require('../automation/jobs');const restarted=new AnalysisJobs(f.base,f.runtime.access,{jobs:new Map()});
    await restarted.init();assert.equal(restarted.list('user_a')[0].status,'interrupted');assert.equal(f.starts(),0);
});
test('bind conflict does not disable the GUI runtime and shutdown cannot resurrect the listener',async t=>{
    const f=await fixture(t);const http=require('node:http');
    const occupied=await new Promise(resolve=>{const s=http.createServer().listen(0,'127.0.0.1',()=>resolve(s));});
    t.after(()=>new Promise(resolve=>occupied.close(resolve)));
    f.runtime.access.state.port=occupied.address().port;await f.runtime.reconfigure();
    assert.equal(f.runtime.ready,true);assert.equal(f.runtime.status().listening,false);assert.match(f.runtime.error,/EADDRINUSE/);
    await f.runtime.close();await f.runtime.reconfigure();assert.equal(f.runtime.status().listening,false);
});
test('CSV row windows have a pre-allocation line bound and no full-table buffering',async t=>{
    const f=await fixture(t);await f.wa.write('results/a/data.csv',Buffer.from('a,b,c\n1,2,3\n4,5,6\n'),null);
    const r=await execute('results_table',{id:'results/a/data.csv',row:1,rows:1,column:1,columns:2},f.actor,f.runtime);
    assert.deepEqual(r.rows,[['2','3']]);
    await f.wa.write('results/a/large.csv',Buffer.alloc(4*1024*1024+1,97),null);
    await assert.rejects(execute('results_table',{id:'results/a/large.csv',row:0,rows:1,column:0,columns:1},f.actor,f.runtime),{code:'LIMIT_EXCEEDED'});
});
test('hidden descendants cannot bypass recursive link rejection',async t=>{
    const f=await fixture(t);await fs.mkdir(await f.wa.resolve('results/a/.inputs',{create:true,internal:true}),{recursive:true});
    await fs.symlink(path.join(f.base,'configs','user_b'),path.join(f.base,'outputs','user_a','a','.inputs','escape'),'junction');
    await assert.rejects(f.wa.describe('results/a'),{code:'UNSAFE_PATH'});
});
