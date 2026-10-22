'use strict';
// Explicit, isolated acceptance test. Not included in npm test: packaging and
// optional real solves must not run concurrently with pinned solver benchmarks.
// node tests/packaged_mcp_smoke.cjs --exe ../build_mcp/webui-server.exe [--solver ABS_PATH]
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const net=require('node:net');
const {spawn,execFileSync}=require('node:child_process');
const {once}=require('node:events');
const {Client,StreamableHTTPClientTransport}=require('@modelcontextprotocol/client');
const {StdioClientTransport}=require('@modelcontextprotocol/client/stdio');
const {noLinks}=require('../automation/workspace');
const argument=name=>{const i=process.argv.indexOf(name);return i<0?null:process.argv[i+1];};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function freePort(){const server=net.createServer();server.listen(0,'127.0.0.1');await once(server,'listening');const port=server.address().port;await new Promise(resolve=>server.close(resolve));return port;}
async function main(){
    const original=argument('--exe');if(!original)throw Error('--exe is required');
    const expectedNode=(await fs.readFile(path.join(__dirname,'../../.node-version'),'utf8')).trim();
    const runtime=JSON.parse(execFileSync(path.resolve(original),['--runtime-info'],{encoding:'utf8',windowsHide:true,timeout:15000}));
    assert.equal(runtime.node,expectedNode,'Packaged Node must match the reviewed security release');
    assert.equal(runtime.packaged,true);
    const solver=argument('--solver');
    const temp=await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()),'omfdm-packaged-mcp-'));
    const base=path.join(temp,'app');await fs.mkdir(base);
    const exe=path.join(base,process.platform==='win32'?'webui-server.exe':'webui-server');
    await fs.copyFile(path.resolve(original),exe);if(process.platform!=='win32')await fs.chmod(exe,0o700);
    if(solver) {
        const source=path.resolve(solver),target=path.join(base,process.platform==='win32'?'MagFDMsolver.exe':'MagFDMsolver');
        await fs.copyFile(source,target);if(process.platform!=='win32')await fs.chmod(target,0o700);
        for(const name of await fs.readdir(path.dirname(source)))if(/\.dll$/i.test(name))await fs.copyFile(path.join(path.dirname(source),name),path.join(base,name));
    }
    const repo=path.join(__dirname,'..','..');
    await fs.cp(path.join(repo,'webui','public'),path.join(base,'public'),{recursive:true});
    for(const name of ['sample_config.yaml','general_materials.yaml'])await fs.copyFile(path.join(repo,name),path.join(base,name));
    const port=await freePort(),mcpPort=await freePort();
    const env={...process.env,PORT:String(port),HOME:path.join(temp,'home'),LOCALAPPDATA:path.join(temp,'home'),OPENMAGFDM_NO_LISTEN:'0',OPENMAGFDM_SKIP_INIT:'0',OMP_NUM_THREADS:'2',
        PATH:(process.env.PATH||'').split(path.delimiter).filter(p=>path.resolve(p)!==path.dirname(process.execPath)).join(path.delimiter),
        };
    delete env.SOLVER_PATH; // Exercise the actual distribution layout, not a development override.
    let output='';const child=spawn(exe,[],{cwd:base,env,windowsHide:true,stdio:['ignore','pipe','pipe']});
    let stderr='';
    child.stdout.on('data',b=>{output=(output+b).slice(-65536);});child.stderr.on('data',b=>{stderr=(stderr+b).slice(-65536);});
    let cookie='',csrf='';const url=`http://127.0.0.1:${port}`;const clients=[];
    const admin=async(endpoint,method='GET',body)=>{
        const response=await fetch(url+endpoint,{method,headers:{'Content-Type':'application/json',Cookie:cookie,'X-Omfdm-Csrf':csrf},...(body===undefined?{}:{body:JSON.stringify(body)})});
        if(response.headers.get('set-cookie'))cookie=response.headers.get('set-cookie').split(';')[0];
        assert.equal(response.ok,true,`Management ${endpoint}: ${response.status}`);return response.json();
    };
    try {
        let code;const deadline=Date.now()+30000;
        while(!(code=/MCP management pairing code: ([A-Za-z0-9_-]+)/.exec(output)?.[1])){
            // Report only the sanitized initialization code, never raw output:
            // stdout may contain the private pairing code.
            const failure=/MCP unavailable \(([A-Z0-9_]+)\)/.exec(stderr)?.[1];
            if(failure||child.exitCode!==null||child.signalCode!==null||Date.now()>deadline)
                throw Error(`Packaged WebUI did not initialize MCP management (${failure||'STARTUP_TIMEOUT_OR_EXIT'}; exit=${child.exitCode}, signal=${child.signalCode})`);
            await delay(100);
        }
        assert.equal((await fetch(url+'/api/mcp/status')).status,401);
        csrf=(await admin('/api/mcp/pair','POST',{code})).csrf;
        assert.equal((await admin('/api/mcp/status')).enabled,false);
        if(solver){const location=await admin('/api/mcp/solver');assert.equal(location.source,'automatic');assert.equal(location.available,true);}
        const connection=await admin('/api/mcp/connections','POST',{name:'Packaged smoke',user:'smoke_user',permissions:['read','edit','images','run','cancel','replace','delete']});
        await admin('/api/mcp/settings','PUT',{enabled:true,port:mcpPort});
        const setup=await admin(`/api/mcp/setup/${connection.id}`);
        assert.equal(setup.url,`http://127.0.0.1:${mcpPort}/mcp`);
        assert.equal(setup.stdio.command,exe);
        assert.deepEqual(setup.stdio.args,['--mcp-stdio','--connection',connection.id]);
        assert.ok(!JSON.stringify(setup).includes(connection.token),'Settings must not disclose credentials');
        assert.equal((await admin(`/api/mcp/connections/${connection.id}/test`,'POST',{})).connected,true);
        console.log('PASS: packaged management setup and connection test.');
        const httpClient=new Client({name:'packaged-http',version:'1'},{versionNegotiation:{mode:'auto'}});clients.push(httpClient);
        await httpClient.connect(new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${mcpPort}/mcp`),{requestInit:{headers:{Authorization:`Bearer ${connection.token}`}}}));
        const stdio=new Client({name:'packaged-stdio',version:'1'});clients.push(stdio);
        await stdio.connect(new StdioClientTransport({command:exe,args:['--mcp-stdio','--connection',connection.id],env,stderr:'pipe'}));
        for(const [index,client] of clients.entries()) {
            const tool=async(name,args={})=>{const r=await client.callTool({name,arguments:args});assert.notEqual(r.isError,true,`Tool failed: ${name}: ${JSON.stringify(r.structuredContent?.error)}`);return r.structuredContent;};
            const catalog=await client.listTools();
            for(const name of ['workspace_status','files_list','yaml_inspect','yaml_write','analysis_start','analysis_get','result_summary','results_table']) {
                const entry=catalog.tools.find(t=>t.name===name);
                assert.ok(entry?.description&&entry.inputSchema&&entry.outputSchema,`Discoverable typed tool: ${name}`);
            }
            assert.equal((await tool('workspace_status')).workspace,'smoke_user');
            const template=await tool('yaml_template',{kind:'cartesian'});
            assert.equal((await tool('yaml_inspect',{content:template.content})).valid,true);
            const yaml=require('js-yaml');
            const acceptance=await require('../tools/create-mcp-benchmark.cjs').fixture();
            console.log(`Testing ${index===0?'HTTP':'stdio'} acceptance fixture.`);
            const model=acceptance.model;
            const library=await tool('yaml_inspect',{content:'material_presets:\n  air: {mu_r: 1}\n  test_body: {mu_r: 2}\n'});
            assert.deepEqual(library.summary,{kind:'materials',materialCount:2});
            const config=await tool('yaml_write',{id:`configs/smoke${index}.yaml`,expectedRevision:null,content:yaml.dump(model)});
            const conflict=await client.callTool({name:'yaml_write',arguments:{id:config.id,expectedRevision:null,content:yaml.dump(model)}});
            assert.equal(conflict.isError,true);assert.equal(conflict.structuredContent.error.code,'REVISION_CONFLICT');
            const bytes=acceptance.png;
            const transfer=await tool('transfer_begin',{kind:'image',size:bytes.length});
            await tool('transfer_append',{transferId:transfer.transferId,offset:0,base64:bytes.toString('base64')});
            const image=await tool('image_import',{transferId:transfer.transferId,id:`images/smoke${index}.png`,expectedRevision:null});
            await tool('transfer_release',{transferId:transfer.transferId});
            assert.equal((await tool('image_info',{id:image.id})).width,32);
            if(solver) {
                const finished=async job=>{
                    let state;const deadline=Date.now()+60000;
                    do {state=await tool('analysis_get',{jobId:job.jobId});if(!['queued','preparing','running','cancelling'].includes(state.status))return state;await delay(100);}while(Date.now()<deadline);
                    throw Error(`Real solver timed out (${index===0?'HTTP':'stdio'})`);
                };
                const input={config:{id:config.id,revision:config.revision},image:{id:image.id,revision:image.revision},idempotencyKey:`smoke-${index}`};
                const job=await tool('analysis_start',input);
                assert.equal((await tool('analysis_start',input)).jobId,job.jobId);
                const state=await finished(job);
                assert.equal(state.completed,true,`Real solver status: ${state.status}`);assert.equal(state.converged,true);
                const conditions=await tool('files_read',{id:`${job.resultId}/conditions.json`,text:true});assert.ok(JSON.parse(conditions.content));
                const summary=await tool('result_summary',{id:job.resultId});
                assert.equal(summary.converged,true);assert.ok(summary.totalSteps>=1);
                const table=await tool('results_table',{id:`${job.resultId}/Az/step_0001.csv`,row:0,rows:2,column:0,columns:3});
                assert.equal(table.rows.length,2);assert.ok(table.rows.every(row=>row.length===3));

                // A deliberately impossible tolerance must report warning-complete,
                // not a crash or a converged result. This is a diagnostic fixture.
                const warningConfig=await tool('yaml_write',{id:`configs/warning${index}.yaml`,expectedRevision:null,content:yaml.dump(acceptance.warning)});
                const warningJob=await tool('analysis_start',{...input,config:{id:warningConfig.id,revision:warningConfig.revision},idempotencyKey:`warning-${index}`});
                const warning=await finished(warningJob);
                assert.equal(warning.completed,true);assert.equal(warning.warning,true);assert.equal(warning.converged,false);assert.equal(warning.exitCode,2);
                const diagnostic=await tool('result_summary',{id:warningJob.resultId});
                assert.equal(diagnostic.converged,false);assert.ok(diagnostic.warnings.length>0);assert.ok(diagnostic.totalSteps>=1);
                console.log(`PASS: ${index===0?'HTTP':'stdio'} real solver, result tables, and nonconvergence warning.`);
            }
        }
        // Test actual transport permissions and isolation, not just service calls.
        for(const user of ['smoke_user','other_smoke_user']) {
            const readOnly=await admin('/api/mcp/connections','POST',{name:'Read-only acceptance',user,permissions:['read']});
            const reader=new Client({name:'packaged-reader',version:'1'},{versionNegotiation:{mode:'auto'}});clients.push(reader);
            await reader.connect(new StreamableHTTPClientTransport(new URL(setup.url),{requestInit:{headers:{Authorization:`Bearer ${readOnly.token}`}}}));
            const denied=await reader.callTool({name:'yaml_write',arguments:{id:'configs/forbidden.yaml',expectedRevision:null,content:'mesh: {}'}});
            assert.equal(denied.isError,true);assert.equal(denied.structuredContent.error.code,'FORBIDDEN');
            const listing=await reader.callTool({name:'files_list',arguments:{category:'configs'}});
            assert.equal(listing.isError,undefined);
            if(user==='other_smoke_user') {
                assert.equal(listing.structuredContent.items.length,0);
                const missing=await reader.callTool({name:'files_read',arguments:{id:'configs/smoke0.yaml',text:true}});
                assert.equal(missing.isError,true);assert.equal(missing.structuredContent.error.code,'NOT_FOUND');
            } else assert.ok(listing.structuredContent.items.some(file=>file.id==='configs/smoke0.yaml'));
            await reader.close();
        }
        await admin(`/api/mcp/connections/${connection.id}`,'PATCH',{revoked:true});
        await assert.rejects(httpClient.callTool({name:'files_list',arguments:{}}));
        console.log(`PASS: packaged HTTP + stdio, setup/test, tool discovery, read-only/isolation, revisions, revocation${solver?', real solver + warnings + result retrieval':''}.`);
    } catch(error) {
        console.error(error.stack);
        throw error;
    } finally {
        for(const client of clients)await Promise.race([client.close().catch(()=>{}),delay(3000)]);
        const running=()=>child.exitCode===null&&child.signalCode===null;
        if(running()) {
            const exited=once(child,'exit');
            child.kill('SIGTERM');await Promise.race([exited,delay(10000)]);
            if(running()){child.kill('SIGKILL');await Promise.race([exited,delay(10000)]);}
            assert.equal(running(),false,'Packaged server did not stop');
        }
        await noLinks(temp);assert.ok(path.basename(temp).startsWith('omfdm-packaged-mcp-'));await fs.rm(temp,{recursive:true,force:true});
    }
}
let complete=false;
process.on('beforeExit',()=>{if(!complete){console.error('FAIL: acceptance test exited before completing');process.exitCode=1;}});
main().then(()=>{complete=true;},error=>{complete=true;console.error(error.stack);process.exitCode=1;});
