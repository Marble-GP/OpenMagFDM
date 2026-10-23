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
  storage set-workspace-root <absolute-directory>
  storage migrate-workspaces <absolute-empty-directory>
  storage set-results-root <absolute-directory>
  storage migrate-results <absolute-empty-directory>
  network local
  network shared --cert <path> --key <path> [--host address]
  limits show [workspace]
  limits set-default <configs|images|libraries> <count|unlimited>
  limits set <workspace> <configs|images|libraries> <count|unlimited|default>

With a terminal attached, starting the server opens an interactive console.
Type help at OpenMagFDM> for commands, or use --no-console for service mode.
For source installs, one-shot commands remain available with:
  node webui/launcher.js <command>
For packaged installs, use: webui-server[.exe] <command>
One-shot storage/network/limit changes require the server to be stopped.
In the interactive console, account/owner/limit changes apply while running;
storage/network changes use stop, change settings, then start in that console.
Workspace-root settings cover configs/uploads/user-libs/outputs; a separate
results-root override takes precedence. migrate-workspaces consolidates all
four categories and clears that override after verified copying. Originals stay.
Limits apply per workspace (the existing browser userId), across WebUI and MCP.
Counts include nested files in each input category, not result artifacts.
Defaults: configs 20, images 20, libraries unlimited. 0 blocks new files.
Existing files are retained; replacement remains possible at/above the limit.
Start again after offline changes. 'default' removes a workspace override.
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
const within=(parent,child)=>{const relative=path.relative(parent,child);return !relative||(!relative.startsWith(`..${path.sep}`)&&relative!=='..'&&!path.isAbsolute(relative));};
async function copyVerified(source,target) {
    await noLinks(source,{missing:true});
    const exists=await fs.stat(source).then(()=>true,error=>{if(error.code==='ENOENT')return false;throw error;});
    const manifest=exists?await files(source):[];
    await fs.mkdir(target,{recursive:true});
    if((await fs.readdir(target)).length)throw Error('Migration destination must be empty');
    if(exists) {
        for(const name of await fs.readdir(source))await fs.cp(path.join(source,name),path.join(target,name),{recursive:true,force:false,errorOnExist:true});
    }
    const key=items=>JSON.stringify(items.sort((a,b)=>a.relative.localeCompare(b.relative)));
    if(key(await files(target))!==key(manifest))throw Error('Copy verification failed; settings were not changed');
    // External tools may still modify data while the server is stopped.
    const after=await fs.stat(source).then(()=>files(source),error=>{if(error.code==='ENOENT')return [];throw error;});
    if(key(after)!==key(manifest))throw Error('Source changed during migration; settings were not changed');
}
async function command(base,args,{password:askPassword=password,live=false}={}) {
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
                const first=await askPassword('Password (12+ characters): '),second=await askPassword('Repeat password: ');if(first!==second)throw Error('Passwords differ');
                account={name,role:rest.includes('--admin')?'admin':'user',disabled:false,password:await passwordHash(first),revision:crypto.randomUUID()};data.users.push(account);
            }else{
                if(!account)throw Error('Unknown account');
                if(command==='disable') {if(account.role==='admin'&&data.users.filter(u=>u.role==='admin'&&!u.disabled).length<=1)throw Error('Cannot disable the last administrator');account.disabled=true;}
                else if(command==='enable')account.disabled=false;
                else if(command==='reset-password'){const first=await askPassword('New password: '),second=await askPassword('Repeat password: ');if(first!==second)throw Error('Passwords differ');account.password=await passwordHash(first);}
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
    const settings=readSettings(base);
    if(group!=='limits')configure(base,settings);
    if(group==='limits') {
        const limits=require('./limits');
        if(command==='show') {console.log(JSON.stringify({workspace:value?userKey(value):null,limits:limits.effective(settings,value?userKey(value):undefined),overrides:settings.fileLimits||{}},null,2));return;}
        settings.fileLimits ||= {defaults:{},workspaces:{}};
        settings.fileLimits.defaults ||= {};settings.fileLimits.workspaces ||= {};
        const category=command==='set-default'?value:rest[0];
        if(!Object.hasOwn(limits.defaults,category))throw Error('Category must be configs, images or libraries');
        if(command==='set-default')settings.fileLimits.defaults[category]=limits.value(rest[0]);
        else if(command==='set') {
            const workspace=userKey(value);
            if(!Object.hasOwn(settings.fileLimits.workspaces,workspace))Object.defineProperty(settings.fileLimits.workspaces,workspace,{value:{},enumerable:true,writable:true,configurable:true});
            if(rest[1]==='default')delete settings.fileLimits.workspaces[workspace][category];
            else settings.fileLimits.workspaces[workspace][category]=limits.value(rest[1]);
        } else throw Error('Use limits show, set-default or set');
        await atomicJSON(settingsFile,settings);configure(base,settings);
        console.log(live?'Saved and applied to the running server. Existing files were retained.':'Saved. Restart the server. Existing files were retained.');return;
    }
    if(group==='storage') {
        const workspaceRoot=settings.workspaceRoot?path.resolve(settings.workspaceRoot):path.resolve(base);
        if(command==='show'){console.log(JSON.stringify({workspaceRoot,configsRoot:root(base,'configs'),imagesRoot:root(base,'uploads'),librariesRoot:root(base,'user-libs'),resultsRoot:root(base,'outputs'),resultsRootOverride:settings.resultsRoot||null},null,2));return;}
        if(!['set-workspace-root','migrate-workspaces','set-results-root','migrate-results'].includes(command))throw Error('Unknown storage command');
        if(!value||!path.isAbsolute(value))throw Error('An absolute directory is required');
        const target=path.resolve(value),source=root(base,'outputs');await noLinks(target,{missing:true});
        if(within(target,directory)||within(directory,target))throw Error('Workspace storage must not overlap private server state');
        const targets=command.endsWith('workspaces')||command==='set-workspace-root'?['configs','uploads','user-libs','outputs'].map(folder=>path.join(target,folder)):[target];
        for(const location of targets)for(const publicDir of [path.join(base,'public'),path.join(base,'webui','public'),path.join(base,'icon'),path.join(base,'webui','icon')]) {
            if(within(location,publicDir)||within(publicDir,location))throw Error('Workspace storage must not overlap public web assets');
        }
        if(command==='migrate-workspaces') {
            const folders=['configs','uploads','user-libs','outputs'];
            for(const folder of folders){const original=root(base,folder);if(within(original,target)||within(target,original))throw Error('Source and destination must not overlap');}
            await fs.mkdir(target,{recursive:true});if((await fs.readdir(target)).length)throw Error('Migration destination must be empty');
            for(const folder of folders)await copyVerified(root(base,folder),path.join(target,folder));
            settings.workspaceRoot=target;delete settings.resultsRoot;
        } else if(command==='migrate-results') {
            if(within(source,target)||within(target,source))throw Error('Source and destination must not overlap');
            await copyVerified(source,target);settings.resultsRoot=target;
        } else if(command==='set-workspace-root')settings.workspaceRoot=target;
        else settings.resultsRoot=target;
        await fs.mkdir(target,{recursive:true});await fs.access(target,fs.constants.W_OK);
        await atomicJSON(settingsFile,settings);configure(base,settings);
        console.log('Saved. Restart the server. Original data was retained.');
        if(command==='set-workspace-root'&&settings.resultsRoot)console.log('The separate results-root override is still active; use storage show to inspect all paths.');
        return;
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
async function run(base,args,options={}) {
    if(args.some(a=>['help','--help','-h'].includes(a))) {console.log(HELP);return;}
    const directory=privateDirectory(base);await ensurePrivateDirectory(directory);
    const maintenance=args[0]==='network'||(['storage','limits'].includes(args[0])&&args[1]!=='show');
    if(options.live) {
        const lock=JSON.parse(await fs.readFile(path.join(directory,'server-lock.json'),'utf8'));
        if(lock.pid!==process.pid)throw Error('Live commands must run inside the owning server');
        if(maintenance && args[0]!=='limits')throw Error('Use stop, change settings here, then start. Running analyses must finish first.');
        return command(base,args,options);
    }
    if(!maintenance)return command(base,args,options);
    // Share the server's exclusive startup lock throughout copy, verification
    // and settings replacement. A second process cannot start during migration.
    const stale=await stopped(directory);
    const lock=path.join(directory,'server-lock.json');await noLinks(lock,{missing:true});
    if(stale)try{await fs.unlink(lock);}catch(error){if(error.code!=='ENOENT')throw error;}
    await fs.writeFile(lock,JSON.stringify({pid:process.pid,maintenance:true}),{flag:'wx',mode:0o600});
    try{return await command(base,args,options);}
    finally{if(JSON.parse(await fs.readFile(lock,'utf8')).pid===process.pid)await fs.unlink(lock);}
}
module.exports={run};
