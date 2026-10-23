'use strict';
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),net=require('node:net');
const assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{once}=require('node:events');
const arg=key=>{const i=process.argv.indexOf(key);return i<0?null:process.argv[i+1];};
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn,label,timeout=20000){const end=Date.now()+timeout;while(Date.now()<end){const result=await fn();if(result)return result;await delay(25);}throw Error(`Timed out: ${label}`);}
async function main(){
    const {repo,root,base,port,env,executable,args}=await require('./console_fixture.cjs').createFixture({exe:arg('--exe'),solver:arg('--solver')});
    args.push('--console');
    const child=spawn(executable,args,{cwd:base,env,windowsHide:true,stdio:['pipe','pipe','pipe']});let output='';
    child.stdout.on('data',b=>{output=(output+b).slice(-262144);});child.stderr.on('data',b=>{output=(output+b).slice(-262144);});
    const request=(endpoint,body)=>fetch(`http://127.0.0.1:${port}${endpoint}`,{...(body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(3000)});
    async function command(line,expected){const begin=output.length;child.stdin.write(line+'\n');return until(()=>{const text=output.slice(begin);return text.includes(expected)&&text.includes('OpenMagFDM>')?text:false;},line);}
    try{
        await until(()=>output.includes('Server ready:')&&output.includes('OpenMagFDM>'),'console startup');
        assert.equal((await request('/api/auth/me')).status,200);
        await command('help','Guided file-limit settings');
        const status=await command('status','"pid"');const pid=JSON.parse(status.slice(status.indexOf('{'),status.lastIndexOf('}')+1)).pid;
        await command('limits set console_test configs 0','Saved and applied');
        const write=()=>request('/api/config',{userId:'console_test',file:'a.yaml',content:'variables: {}',expectedRevision:null});
        assert.equal((await write()).status,409,'Live quota must affect real HTTP writes without a restart');
        await command('limits set console_test configs 2','Saved and applied');assert.equal((await write()).status,200);
        assert.equal(JSON.parse((await command('status','"pid"')).match(/\{[\s\S]*\}/)[0]).pid,pid,'Limits must not restart the server');
        await command('limits set-default configs invalid','Error:');await command('limits show console_test','"configs": 2');
        await command('unknown-command','Unknown command');
        const before=await command('storage set-results-root "'+path.join(root,'new results')+'"','Use stop');assert.match(before,/Error:/);
        await command('jobs','OpenMagFDM>');
        if(arg('--solver')) {
            const fixture=await require('../tools/create-mcp-benchmark.cjs').fixture();
            fixture.model.transient={enabled:true,enable_sliding:false,total_steps:10000,parallel_chunks:1,export_fields:['Az']};
            await fs.mkdir(path.join(base,'uploads','console_test'),{recursive:true});
            await fs.writeFile(path.join(base,'uploads','console_test','model.png'),fixture.png);
            await fs.writeFile(path.join(base,'configs','console_test','run.yaml'),require('js-yaml').dump(fixture.model));
            const response=await request('/api/jobs',{userId:'console_test',configFile:'run.yaml',imageFile:'model.png',idempotencyKey:'console-active-stop'});
            assert.equal(response.status,202);const job=await response.json();
            await until(async()=>{const j=await (await request(`/api/jobs/${job.jobId}?userId=console_test`)).json();return j.status==='running';},'running native solver');
            await command('stop','Analyses are active');assert.equal((await request('/api/auth/me')).status,200);
            await command('stop --force','Server stopped');
        } else await command('stop','Server stopped');
        await assert.rejects(request('/api/auth/me'));assert.throws(()=>process.kill(pid,0));
        await command('storage set-results-root "'+path.join(root,'new results')+'"','Original data was retained');
        await command('start','Server ready:');assert.equal((await request('/api/auth/me')).status,200);
        await command('limits show console_test','"configs": 2');
        await command('restart','Server ready:');assert.equal((await request('/api/auth/me')).status,200);
        const closed=once(child,'exit');child.stdin.write('exit\n');const [code]=await closed;assert.equal(code,0);
        await assert.rejects(request('/api/auth/me'));
        console.log(`PASS: resident console, live HTTP quotas, invalid-command recovery, stop/settings/start/restart, no orphan server${arg('--solver')?', real-analysis stop guard and forced cancellation':''}.`);
    }catch(error){console.error(output.replace(/(pairing code: )\S+/g,'$1[redacted]'));throw error;}
    finally{
        if(child.exitCode===null&&child.signalCode===null){const closed=once(child,'exit');child.kill('SIGTERM');await Promise.race([closed,delay(45000)]);}
        assert.ok(path.basename(root).startsWith('omfdm-console-'));await fs.rm(root,{recursive:true,force:true,maxRetries:20,retryDelay:100});
    }
}
async function disconnected(mode){
    const {root,base,port,env,executable,args}=await require('./console_fixture.cjs').createFixture({exe:arg('--exe')});
    const child=spawn(executable,[...args,'--console'],{cwd:base,env,windowsHide:true,stdio:['pipe','pipe','pipe']});let output='',serverPid;
    child.stdout.on('data',b=>{output+=b;});child.stderr.on('data',b=>{output+=b;});
    try{
        await until(()=>output.includes('Server ready:')&&output.includes('OpenMagFDM>'),'lifecycle startup');
        const before=output.length;child.stdin.write('status\n');await until(()=>output.slice(before).includes('"activity"'),'lifecycle status');
        serverPid=JSON.parse(output.slice(before).match(/\{[\s\S]*\}/)[0]).pid;
        const closed=once(child,'exit');
        if(mode==='eof')child.stdin.end();else child.kill('SIGKILL');
        const result=await closed;if(mode==='eof')assert.equal(result[0],0);
        await until(()=>{try{process.kill(serverPid,0);return false;}catch(e){if(e.code==='ESRCH')return true;throw e;}},`server exits after ${mode}`,15000);
        await assert.rejects(fetch(`http://127.0.0.1:${port}/api/auth/me`,{signal:AbortSignal.timeout(1000)}));
        console.log(`PASS: ${mode} leaves no server process or listening HTTP endpoint.`);
    }finally{
        if(child.exitCode===null&&child.signalCode===null)child.kill('SIGKILL');
        if(serverPid)await until(()=>{try{process.kill(serverPid,0);return false;}catch{return true;}},'fixture shutdown',15000);
        assert.ok(path.basename(root).startsWith('omfdm-console-'));await fs.rm(root,{recursive:true,force:true,maxRetries:20,retryDelay:100});
    }
}
main().then(()=>disconnected('eof')).then(()=>disconnected('abrupt parent termination')).catch(error=>{console.error(error.stack);process.exitCode=1;});
