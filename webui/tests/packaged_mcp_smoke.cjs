'use strict';
// Explicit, isolated acceptance test. Not included in npm test: packaging and
// optional real solves must not run concurrently with pinned solver benchmarks.
// node tests/packaged_mcp_smoke.cjs --exe ../build_mcp/webui-server.exe [--solver ABS_PATH]
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const net=require('node:net');
const {spawn}=require('node:child_process');
const {once}=require('node:events');
const {Client,StreamableHTTPClientTransport}=require('@modelcontextprotocol/client');
const {StdioClientTransport}=require('@modelcontextprotocol/client/stdio');
const {noLinks}=require('../automation/workspace');
const Jimp=require('jimp');
const argument=name=>{const i=process.argv.indexOf(name);return i<0?null:process.argv[i+1];};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function freePort(){const server=net.createServer();server.listen(0,'127.0.0.1');await once(server,'listening');const port=server.address().port;await new Promise(resolve=>server.close(resolve));return port;}
async function main(){
    const original=argument('--exe');if(!original)throw Error('--exe is required');
    const solver=argument('--solver');
    const temp=await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()),'omfdm-packaged-mcp-'));
    const base=path.join(temp,'app');await fs.mkdir(base);
    const exe=path.join(base,process.platform==='win32'?'webui-server.exe':'webui-server');
    await fs.copyFile(path.resolve(original),exe);if(process.platform!=='win32')await fs.chmod(exe,0o700);
    const repo=path.join(__dirname,'..','..');
    await fs.cp(path.join(repo,'webui','public'),path.join(base,'public'),{recursive:true});
    for(const name of ['sample_config.yaml','general_materials.yaml'])await fs.copyFile(path.join(repo,name),path.join(base,name));
    const port=await freePort(),mcpPort=await freePort();
    const env={...process.env,PORT:String(port),HOME:path.join(temp,'home'),LOCALAPPDATA:path.join(temp,'home'),OPENMAGFDM_NO_LISTEN:'0',OPENMAGFDM_SKIP_INIT:'0',
        PATH:(process.env.PATH||'').split(path.delimiter).filter(p=>path.resolve(p)!==path.dirname(process.execPath)).join(path.delimiter),
        ...(solver?{SOLVER_PATH:path.resolve(solver)}:{})};
    let output='';const child=spawn(exe,[],{cwd:base,env,windowsHide:true,stdio:['ignore','pipe','pipe']});
    child.stdout.on('data',b=>{output=(output+b).slice(-65536);});child.stderr.on('data',()=>{});
    let cookie='',csrf='';const url=`http://127.0.0.1:${port}`;const clients=[];
    const admin=async(endpoint,method='GET',body)=>{
        const response=await fetch(url+endpoint,{method,headers:{'Content-Type':'application/json',Cookie:cookie,'X-Omfdm-Csrf':csrf},...(body===undefined?{}:{body:JSON.stringify(body)})});
        if(response.headers.get('set-cookie'))cookie=response.headers.get('set-cookie').split(';')[0];
        assert.equal(response.ok,true,`Management ${endpoint}: ${response.status}`);return response.json();
    };
    try {
        let code;const deadline=Date.now()+30000;
        while(!(code=/MCP management pairing code: ([A-Za-z0-9_-]+)/.exec(output)?.[1])){
            if(child.exitCode!==null||Date.now()>deadline)throw Error('Packaged WebUI did not initialize MCP management');await delay(100);
        }
        assert.equal((await fetch(url+'/api/mcp/status')).status,401);
        csrf=(await admin('/api/mcp/pair','POST',{code})).csrf;
        assert.equal((await admin('/api/mcp/status')).enabled,false);
        const connection=await admin('/api/mcp/connections','POST',{name:'Packaged smoke',user:'smoke_user',permissions:['read','edit','images','run','cancel','replace','delete']});
        await admin('/api/mcp/settings','PUT',{enabled:true,port:mcpPort});
        const httpClient=new Client({name:'packaged-http',version:'1'},{versionNegotiation:{mode:'auto'}});clients.push(httpClient);
        await httpClient.connect(new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${mcpPort}/mcp`),{requestInit:{headers:{Authorization:`Bearer ${connection.token}`}}}));
        const stdio=new Client({name:'packaged-stdio',version:'1'});clients.push(stdio);
        await stdio.connect(new StdioClientTransport({command:exe,args:['--mcp-stdio','--connection',connection.id],env,stderr:'pipe'}));
        for(const [index,client] of clients.entries()) {
            const tool=async(name,args={})=>{const r=await client.callTool({name,arguments:args});assert.notEqual(r.isError,true,`Tool failed: ${name}`);return r.structuredContent;};
            assert.equal((await tool('workspace_status')).workspace,'smoke_user');
            const template=await tool('yaml_template',{kind:'cartesian'});
            assert.equal((await tool('yaml_inspect',{content:template.content})).valid,true);
            const yaml=require('js-yaml'),model=yaml.load(template.content);
            model.materials.coil={rgb:[255,0,0],mu_r:1,jz:1e6};
            const config=await tool('yaml_write',{id:`configs/smoke${index}.yaml`,expectedRevision:null,content:yaml.dump(model)});
            const bitmap=new Jimp(32,32,0xffffffff);
            for(let y=12;y<20;y++)for(let x=12;x<20;x++)bitmap.setPixelColor(0xff0000ff,x,y);
            const bytes=await bitmap.getBufferAsync(Jimp.MIME_PNG);
            const transfer=await tool('transfer_begin',{kind:'image',size:bytes.length});
            await tool('transfer_append',{transferId:transfer.transferId,offset:0,base64:bytes.toString('base64')});
            const image=await tool('image_import',{transferId:transfer.transferId,id:`images/smoke${index}.png`,expectedRevision:null});
            await tool('transfer_release',{transferId:transfer.transferId});
            assert.equal((await tool('image_info',{id:image.id})).width,32);
            if(solver) {
                const input={config:{id:config.id,revision:config.revision},image:{id:image.id,revision:image.revision},idempotencyKey:`smoke-${index}`};
                const job=await tool('analysis_start',input);
                assert.equal((await tool('analysis_start',input)).jobId,job.jobId);
                let state;const deadline=Date.now()+60000;
                do {state=await tool('analysis_get',{jobId:job.jobId});if(!['preparing','running','cancelling'].includes(state.status))break;await delay(100);}while(Date.now()<deadline);
                assert.equal(state.completed,true,`Real solver status: ${state.status}`);assert.equal(state.converged,true);
                const conditions=await tool('files_read',{id:`${job.resultId}/conditions.json`,text:true});assert.ok(JSON.parse(conditions.content));
            }
        }
        await admin(`/api/mcp/connections/${connection.id}`,'PATCH',{revoked:true});
        await assert.rejects(httpClient.callTool({name:'files_list',arguments:{}}));
        console.log(`PASS: packaged HTTP + stdio, auth, workspace IO${solver?', real solver round trips':''}.`);
    } finally {
        for(const client of clients)await client.close().catch(()=>{});
        if(child.exitCode===null){child.kill('SIGTERM');await Promise.race([once(child,'exit'),delay(10000)]);if(child.exitCode===null){child.kill('SIGKILL');await once(child,'exit');}}
        await noLinks(temp);assert.ok(path.basename(temp).startsWith('omfdm-packaged-mcp-'));await fs.rm(temp,{recursive:true,force:true});
    }
}
main().catch(error=>{console.error(error.message);process.exitCode=1;});
