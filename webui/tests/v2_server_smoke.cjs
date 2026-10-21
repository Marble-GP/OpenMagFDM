'use strict';
// Explicit Windows integration test. Uses a copied application, a one-day test
// certificate and private test accounts; never enables sharing in the source repo.
// node webui/tests/v2_server_smoke.cjs --solver ABS_EXE --openssl ABS_EXE --artifacts ABS_DIR
const fs=require('node:fs/promises'),path=require('node:path'),https=require('node:https'),net=require('node:net');
const {spawn,spawnSync}=require('node:child_process');
const {once}=require('node:events');const crypto=require('node:crypto');const assert=require('node:assert/strict');
const {passwordHash}=require('../automation/accounts');
const arg=name=>{const i=process.argv.indexOf(name);return i<0?null:process.argv[i+1];};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function main(){
    if(process.platform!=='win32')throw Error('This fixture isolates Windows LOCALAPPDATA; use Windows for this test');
    const solver=arg('--solver'),openssl=arg('--openssl'),artifacts=arg('--artifacts');
    if(!solver||!openssl||!artifacts)throw Error('Supply --solver, --openssl and --artifacts');
    const root=path.resolve(artifacts,`v2-server-${Date.now()}-${crypto.randomUUID()}`),base=path.join(root,'app');
    const repo=path.resolve(__dirname,'../..'),local=path.join(root,'localappdata');
    await fs.mkdir(base,{recursive:true});
    await fs.mkdir(path.join(base,'webui'));
    for(const filename of ['server.js','launcher.js'])await fs.copyFile(path.join(repo,'webui',filename),path.join(base,'webui',filename));
    for(const directory of ['automation','public'])await fs.cp(path.join(repo,'webui',directory),path.join(base,'webui',directory),{recursive:true});
    for(const filename of ['sample_config.yaml','general_materials.yaml'])await fs.copyFile(path.join(repo,filename),path.join(base,filename));
    const privateDir=path.join(local,'OpenMagFDM','automation',crypto.createHash('sha256').update(base).digest('hex').slice(0,20));
    await fs.mkdir(privateDir,{recursive:true});
    const cert=path.join(root,'test-cert.pem'),key=path.join(root,'test-key.pem');
    const opensslConfig=path.join(root,'openssl.cnf');await fs.writeFile(opensslConfig,'[req]\ndistinguished_name=dn\n[dn]\n');
    const generated=spawnSync(openssl,['req','-config',opensslConfig,'-x509','-newkey','rsa:2048','-nodes','-keyout',key,'-out',cert,'-days','1','-subj','/CN=localhost','-addext','subjectAltName=IP:127.0.0.1'],{windowsHide:true});
    assert.equal(generated.status,0,'Test certificate generation');
    const resultsRoot=path.join(root,'external-results');
    await fs.writeFile(path.join(privateDir,'server-settings.json'),JSON.stringify({version:1,mode:'shared',host:'127.0.0.1',tlsCert:cert,tlsKey:key,resultsRoot}));
    const accounts={version:1,users:[],owners:{wa:'alice',wb:'bob'}};
    for(const name of ['alice','bob','admin'])accounts.users.push({name,role:name==='admin'?'admin':'user',revision:crypto.randomUUID(),password:await passwordHash('test-only-password-1234')});
    await fs.writeFile(path.join(privateDir,'accounts.json'),JSON.stringify(accounts));
    const fixture=await require('../tools/create-mcp-benchmark.cjs').fixture();
    for(const category of ['configs','uploads'])await fs.mkdir(path.join(base,category,'wa'),{recursive:true});
    await fs.writeFile(path.join(base,'configs','wa','model.yaml'),require('js-yaml').dump(fixture.model));
    await fs.writeFile(path.join(base,'uploads','wa','model.png'),fixture.png);
    const probe=net.createServer().listen(0,'127.0.0.1');await once(probe,'listening');const port=probe.address().port;await new Promise(resolve=>probe.close(resolve));
    const env={...process.env,PORT:String(port),LOCALAPPDATA:local,NODE_PATH:path.join(repo,'webui','node_modules'),SOLVER_PATH:path.resolve(solver),OPENMAGFDM_NO_LISTEN:'0',OPENMAGFDM_SKIP_INIT:'0',OMP_NUM_THREADS:'2'};
    const child=spawn(process.execPath,[path.join(base,'webui','launcher.js')],{cwd:base,env,windowsHide:true,stdio:['ignore','pipe','pipe']});
    let log='';child.stdout.on('data',b=>{log=(log+b).slice(-32768);});child.stderr.on('data',b=>{log=(log+b).slice(-32768);});
    const ca=await fs.readFile(cert);let cookie='',csrf='';
    const request=(endpoint,method='GET',body,authenticated=true)=>new Promise((resolve,reject)=>{
        const payload=body===undefined?null:JSON.stringify(body);
        const req=https.request({host:'127.0.0.1',port,path:endpoint,method,ca,headers:{Origin:`https://127.0.0.1:${port}`,...(authenticated?{Cookie:cookie,'X-CSRF-Token':csrf}:{}),...(payload?{'Content-Type':'application/json','Content-Length':Buffer.byteLength(payload)}:{})}},res=>{
            const chunks=[];res.on('data',b=>chunks.push(b));res.on('end',()=>{const bytes=Buffer.concat(chunks);let data;try{data=JSON.parse(bytes);}catch{}resolve({status:res.statusCode,headers:res.headers,data,bytes});});
        });req.on('error',reject);req.setTimeout(10000,()=>req.destroy(Error('Request timeout')));req.end(payload);
    });
    try {
        const deadline=Date.now()+30000;
        while(true){try{if((await request('/api/auth/me','GET',undefined,false)).status===401)break;}catch{}if(child.exitCode!==null||Date.now()>deadline)throw Error('Test server failed to start');await delay(100);}
        const login=await request('/api/auth/login','POST',{name:'alice',password:'test-only-password-1234'},false);assert.equal(login.status,200);
        cookie=login.headers['set-cookie'][0].split(';')[0];const me=await request('/api/auth/me');csrf=me.data.csrf;assert.deepEqual(me.data.workspaces,['wa']);
        for(const endpoint of ['/api/results?userId=wb','/api/jobs?userId=wb','/api/backup-manifest?userId=wb','/api/export?userId=wb&include=results','/uploads/wb/secret.png','/api/load-csv-raw?result=outputs/wb/run&file=Az/step_0000.csv'])assert.equal((await request(endpoint)).status,403,endpoint);
        assert.equal((await request('/data/sample_config.yaml')).status,410);
        assert.equal((await request('/api/solve','POST',{userId:'wa'})).status,410);
        let ready;
        while((ready=await request('/api/jobs?userId=wa')).status===503){assert.ok(Date.now()<deadline);await delay(100);}
        assert.equal(ready.status,200);
        const input={userId:'wa',configFile:'model.yaml',imageFile:'model.png',idempotencyKey:'https-smoke'};
        const accepted=await request('/api/jobs','POST',input);assert.equal(accepted.status,202,JSON.stringify(accepted.data));
        assert.equal((await request('/api/jobs','POST',input)).data.jobId,accepted.data.jobId);
        await fs.writeFile(path.join(base,'uploads','wa','model.png'),'changed after acceptance');
        let job;const finishDeadline=Date.now()+60000;
        do {job=(await request(`/api/jobs/${accepted.data.jobId}?userId=wa`)).data;if(!['queued','preparing','running','cancelling'].includes(job.status))break;assert.ok(Date.now()<finishDeadline,'Native analysis timeout');await delay(100);}while(true);
        assert.equal(job.completed,true,JSON.stringify(job));assert.equal(job.converged,true);
        const out=path.join(resultsRoot,'wa',path.basename(job.resultId));
        assert.deepEqual(await fs.readFile(path.join(out,'.inputs','model.png')),fixture.png);
        const manifest=JSON.parse(await fs.readFile(path.join(out,'run-inputs.json'),'utf8'));assert.ok(manifest.effectiveConfigHash);assert.ok(manifest.files.length>=2);
        const backup=await request('/api/export','POST',{userId:'wa',entries:[job.resultId]});assert.equal(backup.status,200);assert.equal(backup.bytes.subarray(0,2).toString(),'PK');
        await fs.writeFile(path.join(root,'result-backup.zip'),backup.bytes);
        assert.equal((await request('/api/auth/logout','POST')).status,200);
        assert.equal((await request('/api/jobs?userId=wa')).status,401);
        const migrated=path.join(root,'migrated-results');
        const cli=args=>spawnSync(process.execPath,[path.join(base,'webui','launcher.js'),...args],{cwd:base,env,windowsHide:true,encoding:'utf8'});
        assert.notEqual(cli(['storage','migrate-results',migrated]).status,0,'Migration must reject a live server');
        const closed=once(child,'close');child.kill();await closed;
        const migration=cli(['storage','migrate-results',migrated]);assert.equal(migration.status,0,migration.stderr);
        assert.equal(JSON.parse(await fs.readFile(path.join(privateDir,'server-settings.json'),'utf8')).resultsRoot,migrated);
        assert.deepEqual(await fs.readFile(path.join(migrated,'wa',path.basename(job.resultId),'.inputs','model.png')),fixture.png);
        assert.deepEqual(await fs.readFile(path.join(out,'.inputs','model.png')),fixture.png,'Migration retains its source');
        await fs.writeFile(path.join(root,'verification.json'),JSON.stringify({passed:true,https:true,nativeCompleted:job.completed,resultPath:job.resultPath,backupBytes:backup.bytes.length,cliMigration:true},null,2));
        console.log(`PASS: HTTPS sessions, ownership, retired APIs, idempotent jobs, real solver, frozen inputs, external results, ZIP and CLI migration. Artifacts: ${root}`);
    } finally {
        // Only the child started by this fixture is stopped. Keep artifacts for inspection.
        if(child.exitCode===null&&child.signalCode===null){const closed=once(child,'close');child.kill();await closed;}
        log='';
    }
}
main().catch(error=>{console.error(error.message);process.exitCode=1;});
