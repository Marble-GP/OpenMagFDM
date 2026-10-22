'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const {privateDirectory,atomicJSON,ensurePrivateDirectory} = require('./access');
const {passwordHash} = require('./accounts');
const {userKey,noLinks,digest,locked} = require('./workspace');
const {readSettings,root,configure} = require('./storage');
const HELP = `OpenMagFDM server CLI
  help | --help | -h                 Show this help (also: <group> --help)
  --runtime-info                    Print embedded Node runtime without starting the server
  users list
  users create <name> [--admin]      Set password through hidden interactive input
  users disable|enable|reset-password <name>
  workspaces list
  workspaces assign <workspace> <account>
  storage show
  storage set-results-root <absolute-directory>
  storage migrate-results <absolute-empty-directory>
  network local
  network shared --cert <path> --key <path> [--host address]
  limits show [workspace]
  limits set-default <configs|images|libraries> <count|unlimited>
  limits set <workspace> <configs|images|libraries> <count|unlimited|default>

With no arguments the WebUI server starts. For source installs, use:
  node webui/launcher.js <command>
For packaged installs, use: webui-server[.exe] <command>
Storage, network and limit changes require the server to be stopped.
Limits apply per workspace (the existing browser userId), across WebUI and MCP.
Counts include nested files in each input category, not result artifacts.
Defaults: configs 20, images 20, libraries unlimited. 0 blocks new files.
Existing files are retained; replacement remains possible at/above the limit.
Restart after changing settings. 'default' removes a workspace override.
`;
async function password(prompt) {
    if(!process.stdin.isTTY)throw Error('Password entry requires an interactive terminal');
    process.stdout.write(prompt); process.stdin.setRawMode(true); process.stdin.resume();
    return new Promise((resolve,reject)=>{
        let value='';
        const finish=error=>{process.stdin.removeListener('data',input);process.stdin.setRawMode(false);process.stdin.pause();process.stdout.write('\n');error?reject(error):resolve(value);};
        const input=chunk=>{for(const c of chunk.toString()){if(c==='\u0003')return finish(Error('Cancelled'));if(c==='\r'||c==='\n')return finish();if(c==='\u007f'||c==='\b')value=value.slice(0,-1);else if(c>=' ')value+=c;}};
        process.stdin.on('data',input);
    });
}
async function stopped(directory) {
    try {const {pid}=JSON.parse(await fs.readFile(path.join(directory,'server-lock.json'),'utf8'));try{process.kill(pid,0);}catch(e){if(e.code==='ESRCH')return true;throw e;}throw Error('Stop the server before changing storage or network settings');}
    catch(e){if(e.code!=='ENOENT')throw e;return false;}
}
async function files(directory) {
    const result=[];
    async function walk(dir) {await noLinks(dir);for(const entry of await fs.readdir(dir,{withFileTypes:true})){const item=path.join(dir,entry.name);await noLinks(item);if(entry.isDirectory())await walk(item);else if(entry.isFile())result.push({relative:path.relative(directory,item),hash:await digest(item)});else throw Error('Non-regular entry');}}
    await walk(directory);return result;
}
async function command(base,args) {
    const [group,command,value,...rest]=args;
    const directory=privateDirectory(base);
    const filename=path.join(directory,'accounts.json');
    const settingsFile=path.join(directory,'server-settings.json');
    if(group==='users'||group==='workspaces')return locked(filename,async()=>{
        let data;try{data=JSON.parse(await fs.readFile(filename,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;data={version:1,users:[],owners:{}};}
        if(group==='users') {
            if(command==='list'){console.table(data.users.map(({name,role,disabled})=>({name,role,disabled})));return;}
            const name=userKey(value);let account=data.users.find(u=>u.name===name);
            if(command==='create') {
                if(account)throw Error('Account exists');
                const first=await password('Password (12+ characters): '),second=await password('Repeat password: ');if(first!==second)throw Error('Passwords differ');
                account={name,role:rest.includes('--admin')?'admin':'user',disabled:false,password:await passwordHash(first),revision:crypto.randomUUID()};data.users.push(account);
            }else{
                if(!account)throw Error('Unknown account');
                if(command==='disable') {if(account.role==='admin'&&data.users.filter(u=>u.role==='admin'&&!u.disabled).length<=1)throw Error('Cannot disable the last administrator');account.disabled=true;}
                else if(command==='enable')account.disabled=false;
                else if(command==='reset-password'){const first=await password('New password: '),second=await password('Repeat password: ');if(first!==second)throw Error('Passwords differ');account.password=await passwordHash(first);}
                else throw Error('Unknown users command');
                account.revision=crypto.randomUUID();
            }
        } else {
            if(command==='list') {
                const found=new Set(Object.keys(data.owners));configure(base);
                for(const folder of ['configs','uploads','user-libs','outputs'])try{for(const e of await fs.readdir(root(base,folder),{withFileTypes:true}))if(e.isDirectory())found.add(e.name);}catch(e){if(e.code!=='ENOENT')throw e;}
                console.table([...found].map(workspace=>({workspace,owner:data.owners[workspace]||'(unassigned)'})));return;
            }
            if(command!=='assign')throw Error('Use workspaces assign <workspace> <account>');
            const workspace=userKey(value),owner=userKey(rest[0]);if(!data.users.some(u=>u.name===owner&&!u.disabled))throw Error('Active account required');
            data.owners[workspace]=owner;
        }
        await atomicJSON(filename,data);console.log('Saved. No workspace files were deleted.');
    });
    const settings=readSettings(base);configure(base,settings);
    if(group==='limits') {
        const limits=require('./limits');
        if(command==='show') {console.log(JSON.stringify({workspace:value?userKey(value):null,limits:limits.effective(settings,value?userKey(value):undefined),overrides:settings.fileLimits||{}},null,2));return;}
        settings.fileLimits ||= {defaults:{},workspaces:{}};
        settings.fileLimits.defaults ||= {};settings.fileLimits.workspaces ||= {};
        const category=command==='set-default'?value:rest[0];
        if(!Object.hasOwn(limits.defaults,category))throw Error('Category must be configs, images or libraries');
        if(command==='set-default')settings.fileLimits.defaults[category]=limits.value(rest[0]);
        else if(command==='set') {
            const workspace=userKey(value);settings.fileLimits.workspaces[workspace] ||= {};
            if(rest[1]==='default')delete settings.fileLimits.workspaces[workspace][category];
            else settings.fileLimits.workspaces[workspace][category]=limits.value(rest[1]);
        } else throw Error('Use limits show, set-default or set');
        await atomicJSON(settingsFile,settings);console.log('Saved. Restart the server. Existing files were retained.');return;
    }
    if(group==='storage') {
        if(command==='show'){console.log(JSON.stringify({resultsRoot:root(base,'outputs')},null,2));return;}
        if(!value||!path.isAbsolute(value))throw Error('An absolute directory is required');
        const target=path.resolve(value),source=root(base,'outputs');await noLinks(target,{missing:true});
        if(command==='migrate-results'){
            const within=(a,b)=>{const rel=path.relative(a,b);return !rel||(!rel.startsWith('..')&&!path.isAbsolute(rel));};
            if(within(source,target)||within(target,source))throw Error('Source and destination must not overlap');
            await fs.mkdir(target,{recursive:true});if((await fs.readdir(target)).length)throw Error('Migration destination must be empty');
            const manifest=await files(source);
            for(const name of await fs.readdir(source))await fs.cp(path.join(source,name),path.join(target,name),{recursive:true,force:false,errorOnExist:true});
            for(const file of manifest)if(await digest(path.join(target,file.relative))!==file.hash)throw Error('Copy verification failed; settings were not changed');
            if((await files(target)).length!==manifest.length)throw Error('Copy file count differs');
        } else if(command!=='set-results-root')throw Error('Unknown storage command');
        await fs.mkdir(target,{recursive:true});await fs.access(target,fs.constants.W_OK);
        settings.resultsRoot=target;await atomicJSON(settingsFile,settings);console.log('Saved. Restart the server. Original data was retained.');return;
    }
    if(group==='network') {
        if(command==='local')settings.mode='local';
        else if(command==='shared') {
            const data=JSON.parse(await fs.readFile(filename,'utf8'));if(!data.users.some(u=>u.role==='admin'&&!u.disabled))throw Error('Create an administrator first');
            const flags=[value,...rest],get=name=>flags[flags.indexOf(name)+1];
            if(!flags.includes('--cert')||!flags.includes('--key'))throw Error('Use network shared --cert <path> --key <path> [--host address]');
            settings.tlsCert=path.resolve(get('--cert'));settings.tlsKey=path.resolve(get('--key'));
            require('node:tls').createSecureContext({cert:await fs.readFile(settings.tlsCert),key:await fs.readFile(settings.tlsKey)});
            settings.host=flags.includes('--host')?get('--host'):'0.0.0.0';settings.mode='shared';
        }else throw Error('Use network local or network shared');
        await atomicJSON(settingsFile,settings);console.log('Saved. Restart the server.');return;
    }
    throw Error('Commands: users, workspaces, storage, network');
}
async function run(base,args) {
    if(args.some(a=>['help','--help','-h'].includes(a))) {console.log(HELP);return;}
    const directory=privateDirectory(base);await ensurePrivateDirectory(directory);
    const maintenance=args[0]==='network'||(['storage','limits'].includes(args[0])&&args[1]!=='show');
    if(!maintenance)return command(base,args);
    // Share the server's exclusive startup lock throughout copy, verification
    // and settings replacement. A second process cannot start during migration.
    const stale=await stopped(directory);
    const lock=path.join(directory,'server-lock.json');await noLinks(lock,{missing:true});
    if(stale)try{await fs.unlink(lock);}catch(error){if(error.code!=='ENOENT')throw error;}
    await fs.writeFile(lock,JSON.stringify({pid:process.pid,maintenance:true}),{flag:'wx',mode:0o600});
    try{return await command(base,args);}
    finally{if(JSON.parse(await fs.readFile(lock,'utf8')).pid===process.pid)await fs.unlink(lock);}
}
module.exports={run};
