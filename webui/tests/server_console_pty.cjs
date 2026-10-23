'use strict';
// Real pseudoterminal: ConPTY on Windows, PTY on Linux/macOS. No user data.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {stripVTControlCharacters}=require('node:util'),pty=require('node-pty');
const {createFixture}=require('./console_fixture.cjs');
const arg=key=>{const i=process.argv.indexOf(key);return i<0?null:process.argv[i+1];};
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function main(){
    const fixture=await createFixture({exe:arg('--exe')});const {root,base,port,env,executable,args,privateDir}=fixture;
    // The test runner itself may use TERM=dumb; the PTY emulates a real terminal.
    env.TERM='xterm-256color';
    const terminal=pty.spawn(executable,args,{name:'xterm-256color',cols:110,rows:35,cwd:base,env});
    let output='',exited=false;let exitResult;const closed=new Promise(resolve=>terminal.onExit(result=>{exited=true;exitResult=result;resolve(result);}));
    terminal.onData(data=>{output=(output+data).slice(-1024*1024);});
    const text=()=>stripVTControlCharacters(output);
    async function until(fn,label){const end=Date.now()+25000;while(Date.now()<end){if(fn())return;assert.equal(exited,false,`PTY exited during ${label}`);await delay(30);}throw Error(`Timed out: ${label}`);}
    async function send(input,expected){const before=text().length;if(input.endsWith('\r')&&input.length>1){terminal.write(input.slice(0,-1));await delay(100);terminal.write('\r');}else terminal.write(input);await until(()=>text().slice(before).includes(expected),expected);await delay(120);}
    const request=(body)=>fetch(`http://127.0.0.1:${port}/api/config`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(3000)});
    try{
        await until(()=>text().includes('Server ready:')&&text().includes('OpenMagFDM>'),'default interactive startup');
        await send('he\t\r','Guided file-limit settings'); // Tab completes help.
        await send('\x1b[A\r','Guided file-limit settings'); // Up recalls help.
        console.log('PTY: default startup, Tab and history passed');
        await send('menu\r','Server action');await send('\x1b[B\r','File-limit scope');
        await send('\r','Files to limit');await send('\r','Maximum files');await send('4\r','Saved and applied');
        const settings=JSON.parse(await fs.readFile(path.join(privateDir,'server-settings.json'),'utf8'));assert.equal(settings.fileLimits.defaults.configs,4);
        console.log('PTY: Inquirer menu and settings passed');
        const password='Disposable-console-secret-413!';
        await send('users create console_admin --admin\r','Password (12+ characters)');
        await send(password+'\r','Repeat password');await send(password+'\r','Saved. No workspace files were deleted');
        assert.equal(output.includes(password),false,'Password must not be echoed');
        const accounts=JSON.parse(await fs.readFile(path.join(privateDir,'accounts.json'),'utf8'));
        assert.equal(accounts.users[0].name,'console_admin');
        assert.ok(await require('../automation/accounts').verify(password,accounts.users[0].password));
        await send('\x1b[A\r','Account exists'); // A secret must never replace command history.
        assert.equal(output.includes(password),false,'Password must not enter history');
        await send('users reset-password console_admin\r','New password');await send('\x03','Input cancelled');
        assert.deepEqual(JSON.parse(await fs.readFile(path.join(privateDir,'accounts.json'),'utf8')).users[0].password,accounts.users[0].password,'Cancelled reset must preserve the old password');
        console.log('PTY: password, history and cancellation passed');
        terminal.resize(74,25);
        terminal.write('sta');await delay(100);
        assert.equal((await request({userId:'log_test',file:'x.yaml',content:'variables: {}',expectedRevision:null})).status,200);
        await delay(100);await send('tus\r','"activity"'); // Async logs preserve the partially typed command.
        await send('configure\r','File-limit scope');await send('\x04','Console closed');await Promise.race([closed,delay(15000).then(()=>{throw Error('Console process did not exit after closing');})]);assert.equal(exitResult.exitCode,0);
        await assert.rejects(fetch(`http://127.0.0.1:${port}/api/auth/me`,{signal:AbortSignal.timeout(1000)}));
        assert.equal(output.includes(password),false);
        if(arg('--artifacts')){
            const destination=path.resolve(arg('--artifacts'));await fs.mkdir(destination,{recursive:true});
            await fs.writeFile(path.join(destination,arg('--exe')?'packaged-console-pty.txt':'source-console-pty.txt'),text().replace(/(pairing code: )\S+/g,'$1[redacted]'));
        }
        console.log(`PASS: ${arg('--exe')?'packaged':'source'} real PTY default startup, Tab/history, Inquirer menu/forms, hidden password/hash, cancel, resize, async-log input recovery and shutdown.`);
    }catch(error){console.error(text().replace(/(pairing code: )\S+/g,'$1[redacted]'));throw error;}
    finally{if(!exited){terminal.write('\x03exit --force\r');await Promise.race([closed,delay(8000)]);if(!exited)terminal.kill();await delay(1500);}assert.ok(path.basename(root).startsWith('omfdm-console-'));await fs.rm(root,{recursive:true,force:true,maxRetries:20,retryDelay:100});}
}
// ConPTY's helper can retain a native handle after onExit. All application
// shutdown and fixture cleanup checks above must finish before ending the runner.
main().then(()=>process.exit(0),e=>{console.error(e.stack);process.exit(1);});
