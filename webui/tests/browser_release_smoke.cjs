'use strict';
// Real installed Chrome/Edge, isolated headless profiles and copied application.
// Does not touch the user's browser sessions, accounts, settings or analysis data.
const fs = require('node:fs/promises');
const path = require('node:path');
const net = require('node:net');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const arg = key => process.argv[process.argv.indexOf(key) + 1];
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(fn, label, timeout = 30000) {
    const end = Date.now() + timeout;
    while (Date.now() < end) { const value = await fn(); if (value) return value; await delay(100); }
    throw Error(`Timed out: ${label}`);
}
async function connect(url) {
    const socket = new WebSocket(url);
    await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, {once:true}); socket.addEventListener('error', reject, {once:true}); });
    let id = 0; const pending = new Map(), errors = [], events = new Map();
    socket.addEventListener('message', event => {
        const data = JSON.parse(event.data);
        if(events.has(data.method)){events.get(data.method)(data.params);events.delete(data.method);}
        if (data.method === 'Runtime.exceptionThrown') errors.push(data.params.exceptionDetails);
        const entry = pending.get(data.id);
        if (entry) { clearTimeout(entry.timer); pending.delete(data.id); data.error ? entry.reject(Error(JSON.stringify(data.error))) : entry.resolve(data.result); }
    });
    const send = (method, params = {}) => new Promise((resolve, reject) => {
        const key = ++id; const timer = setTimeout(() => { pending.delete(key); reject(Error(`CDP timeout: ${method}`)); }, 30000);
        pending.set(key, {resolve,reject,timer}); socket.send(JSON.stringify({id:key,method,params}));
    });
    const evaluate = async expression => {
        const r = await send('Runtime.evaluate', {expression,awaitPromise:true,returnByValue:true});
        if (r.exceptionDetails) throw Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
        return r.result.value;
    };
    const event = name => new Promise((resolve,reject)=>{const timer=setTimeout(()=>{events.delete(name);reject(Error(`CDP event timeout: ${name}`));},30000);events.set(name,value=>{clearTimeout(timer);resolve(value);});});
    return {send,evaluate,event,errors,close:()=>socket.close()};
}
async function browser(exe, label, url, base, root) {
    const profile = path.join(root,label); await fs.mkdir(profile);
    const child = spawn(exe, ['--headless=new','--no-first-run','--no-default-browser-check','--disable-background-networking',`--user-data-dir=${profile}`,'--remote-debugging-port=0','about:blank'], {windowsHide:true,stdio:'ignore'});
    let cdp;
    try {
        const port = await until(async()=>{try{return (await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];}catch{return false;}}, `${label} debugger`);
        const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
        cdp = await connect(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
        await cdp.send('Runtime.enable'); await cdp.send('Page.enable');
        await cdp.send('Emulation.setDeviceMetricsOverride',{width:1600,height:1200,deviceScaleFactor:1,mobile:false});
        await cdp.send('Page.navigate',{url});
        const ready = () => until(()=>cdp.evaluate("typeof AppState !== 'undefined' && Boolean(AppState.userId) && typeof Plotly !== 'undefined'"), `${label} application`);
        const reload = async () => {const loaded=cdp.event('Page.loadEventFired');await cdp.send('Page.reload');await loaded;await ready();};
        await ready();
        const user = await cdp.evaluate('AppState.userId'); assert.match(user,/^[a-zA-Z0-9_-]+$/);
        assert.equal(await cdp.evaluate("typeof FieldGPU === 'undefined' && !document.querySelector('#fieldRenderer option[value=gpu]')"),true);
        assert.equal(await cdp.evaluate("document.getElementById('polarAspectLocked').checked"),true);
        const fixture = await require('../tools/create-mcp-benchmark.cjs').fixture();
        fixture.model.transient = {enabled:true,enable_sliding:false,total_steps:4,parallel_chunks:1,export_fields:['Az','Mu']};
        for (const directory of ['configs','uploads']) await fs.mkdir(path.join(base,directory,user),{recursive:true});
        await fs.writeFile(path.join(base,'configs',user,'browser.yaml'),require('js-yaml').dump(fixture.model));
        await fs.writeFile(path.join(base,'uploads',user,'browser.png'),fixture.png);
        await cdp.evaluate(`JobsPanel.submit({userId:AppState.userId,configFile:'browser.yaml',imageFile:'browser.png'})`);
        await reload();
        const job = await until(async()=>{
            const data = await cdp.evaluate("fetch('/api/jobs?userId='+encodeURIComponent(AppState.userId)).then(r=>r.json())");
            const j=data.jobs?.[0]; if(j && ['failed','interrupted','cancelled'].includes(j.status))throw Error(JSON.stringify(j));return j?.status==='completed'?j:false;
        }, `${label} native job after reload`,60000);
        await cdp.evaluate('JobsPanel.refresh()');
        assert.ok((await cdp.evaluate("document.getElementById('analysisJobs').textContent")).includes(job.jobId.slice(0,8)));
        const loadJobResult = async () => {
            await cdp.evaluate('refreshResultsList()');
            await until(()=>cdp.evaluate(`Array.from(document.getElementById('resultSelect').options).some(o=>o.value===${JSON.stringify(job.resultPath)})`),`${label} result option`);
            await cdp.evaluate(`document.getElementById('resultSelect').value=${JSON.stringify(job.resultPath)}; updateCustomSelectDisplay(${JSON.stringify(job.resultPath)}); loadSelectedResult()`);
        };
        await loadJobResult();
        await until(()=>cdp.evaluate("Boolean(AppState.resultsData.currentResult) && !document.getElementById('resultMemoSave').disabled"),`${label} result loaded`);
        await cdp.evaluate("document.getElementById('resultMemoText').value='Browser acceptance description'; ResultImprovements.save()");
        await loadJobResult();
        await until(()=>cdp.evaluate("document.getElementById('resultMemoText').value==='Browser acceptance description'"),`${label} saved description`);
        await cdp.evaluate("switchTab('dashboard'); addPlotWidget('az_heatmap',0,0,6,3); addPlotWidget('az_boundary',6,0,6,3)");
        const visiblePlots = "[...document.querySelectorAll('.js-plotly-plot')].filter(p=>p.offsetWidth>100 && p.offsetHeight>100).length>=2";
        await until(()=>cdp.evaluate(visiblePlots),`${label} visible scalar and field-line plots`);
        const timings=[];
        for(let i=0;i<16;++i){const started=Date.now();await cdp.evaluate(`setStep(${i%4+1})`);await delay(120);assert.ok(await cdp.evaluate(visiblePlots));timings.push(Date.now()-started);}
        await cdp.evaluate("document.getElementById('polarAspectLocked').click()");
        await reload();
        assert.equal(await cdp.evaluate("document.getElementById('polarAspectLocked').checked"),false);
        await loadJobResult();
        await cdp.evaluate("switchTab('dashboard'); addPlotWidget('az_heatmap',0,0,6,3); addPlotWidget('az_boundary',6,0,6,3)");
        await until(()=>cdp.evaluate(visiblePlots),`${label} visible plots after reload`);
        const screenshot = await cdp.send('Page.captureScreenshot',{format:'png'});
        await fs.writeFile(path.join(root,`${label}.png`),Buffer.from(screenshot.data,'base64'));
        assert.deepEqual(cdp.errors,[],`${label} uncaught browser errors`);
        const info = await cdp.send('Browser.getVersion');
        return {browser:info.product,passed:true,reconnectedJob:job.jobId,description:true,aspectPersistence:true,plotUpdates:timings.length,updateWallMilliseconds:timings};
    } catch(error) {
        if(cdp){
            const state=await cdp.evaluate("({url:location.href,user:typeof AppState==='undefined'?null:AppState.userId,result:typeof AppState==='undefined'?null:AppState.resultsData.currentResult,memo:document.getElementById('resultMemoStatus')?.textContent,status:document.getElementById('solverStatus')?.textContent,picker:document.getElementById('resultSelect')?.innerHTML})").catch(()=>null);
            await fs.writeFile(path.join(root,`${label}-failure.json`),JSON.stringify({state,errors:cdp.errors},null,2));
            const shot=await cdp.send('Page.captureScreenshot',{format:'png'}).catch(()=>null);if(shot)await fs.writeFile(path.join(root,`${label}-failure.png`),Buffer.from(shot.data,'base64'));
            console.error(JSON.stringify(state));
        }
        throw error;
    } finally {
        if(cdp){try{await cdp.send('Browser.close');}catch{}cdp.close();}
        if(child.exitCode===null && child.signalCode===null){const closed=once(child,'close');child.kill();await closed;}
    }
}
async function main(){
    for(const key of ['--solver','--chrome','--edge','--artifacts'])assert.ok(process.argv.includes(key),`Missing ${key}`);
    const repo=path.resolve(__dirname,'../..');const root=path.resolve(arg('--artifacts'),`browser-${Date.now()}`),base=path.join(root,'app');await fs.mkdir(path.join(base,'webui'),{recursive:true});
    for(const file of ['server.js','launcher.js'])await fs.copyFile(path.join(repo,'webui',file),path.join(base,'webui',file));
    for(const directory of ['automation','public'])await fs.cp(path.join(repo,'webui',directory),path.join(base,'webui',directory),{recursive:true});
    for(const file of ['sample_config.yaml','general_materials.yaml'])await fs.copyFile(path.join(repo,file),path.join(base,file));
    const probe=net.createServer().listen(0,'127.0.0.1');await once(probe,'listening');const port=probe.address().port;await new Promise(resolve=>probe.close(resolve));
    const child=spawn(process.execPath,[path.join(base,'webui/launcher.js')],{cwd:base,windowsHide:true,env:{...process.env,PORT:String(port),LOCALAPPDATA:path.join(root,'private'),NODE_PATH:path.join(repo,'webui/node_modules'),SOLVER_PATH:path.resolve(arg('--solver'))},stdio:['ignore','pipe','pipe']});
    let log='';child.stdout.on('data',b=>{log+=b;});child.stderr.on('data',b=>{log+=b;});
    try{
        const url=`http://127.0.0.1:${port}`;await until(async()=>{try{return (await fetch(url+'/api/auth/me')).ok;}catch{return false;}},'isolated server');
        const results=[];for(const name of ['chrome','edge'])results.push(await browser(arg('--'+name),name,url,base,root));
        await fs.writeFile(path.join(root,'verification.json'),JSON.stringify(results,null,2));console.log(`PASS: Chrome and Edge browser checks. Artifacts: ${root}`);
    }finally{await fs.writeFile(path.join(root,'server.log'),log);if(child.exitCode===null&&child.signalCode===null){const closed=once(child,'close');child.kill();await closed;}}
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
