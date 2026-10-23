'use strict';
const readline=require('node:readline');
const {spawn}=require('node:child_process');
const {PassThrough}=require('node:stream');
const commands=['help','menu','configure','status','start','stop','restart','exit','jobs list','jobs cancel','jobs resume','users list','users create','users disable','users enable','users reset-password','workspaces list','workspaces assign','limits show','limits set-default','limits set','storage show','storage set-results-root','storage migrate-results','network local','network shared'];
const help=`OpenMagFDM interactive server console
  help / /help / ?                Show commands
  menu                           Choose an action with arrow keys
  configure                      Guided file-limit settings
  status                         Server address, running work and job status
  jobs [list [workspace]]         Show up to 100 recent jobs
  jobs cancel <workspace> <id>    Request cancellation of a job
  jobs resume <workspace>         Resume jobs held after a restart
  users list|create|disable|enable|reset-password ...
  workspaces list|assign ...      List workspaces or assign an owner
  limits show [workspace]
  limits set-default <configs|images|libraries> <count|unlimited>
  limits set <workspace> <category> <count|unlimited|default>
                                 Apply file limits immediately, without restarting
  storage show|set-results-root|migrate-results ...
  network local|shared ...        Change storage/network after stop, then start
  start / stop / restart         Keep this console open while controlling the server
  exit / quit                    Shut down the server and leave the console

Tab completes commands; Up/Down recalls this session's history.
Quoted paths are supported, e.g. storage set-results-root "D:\\Analysis Results".
User passwords are requested privately and never added to command history.
Use <group> --help for administration arguments. Account/owner/limit changes
apply while running. Storage/network changes require stop then start here.
stop/restart/exit refuse active work; --force explicitly terminates analyses.
Queued jobs are preserved and held after restart; use jobs resume <workspace>.
Ctrl+C clears a partial line; on an empty line it requests exit. Ctrl+D closes
the console and shuts down the server (including active analyses).
`;
function tokenize(line) {
    const tokens=[];let word='',quote=null,started=false;
    for(const c of line.trim()) {
        if(quote){if(c===quote)quote=null;else word+=c;started=true;}
        else if(c==='"'||c==="'"){quote=c;started=true;}
        else if(/\s/.test(c)){if(started){tokens.push(word);word='';started=false;}}
        else {word+=c;started=true;}
    }
    if(quote)throw Error('Unclosed quote');
    if(started)tokens.push(word);
    return tokens;
}
function complete(line) {const hits=commands.filter(c=>c.startsWith(line));return [hits.length?hits:[],line];}
class ServerConsole {
    constructor(base,{input=process.stdin,output=process.stdout}={}) {
        this.base=base;this.input=input;this.output=output;this.child=null;this.requests=new Map();this.id=0;
        this.busy=false;this.closing=false;this.replacingEditor=false;this.secret=false;this.deferredLogs='';this.chain=Promise.resolve();
    }
    print(text) {this.output.write(`${text}\n`);}
    log(bytes) {
        if(this.secret){this.deferredLogs=(this.deferredLogs+bytes).slice(-65536);return;}
        if(this.rl?.terminal&&!this.busy){readline.clearLine(this.output,0);readline.cursorTo(this.output,0);}
        this.output.write(bytes);
        if(this.rl&&!this.busy)this.rl.prompt(true);
    }
    editor(history=[]) {
        const rl=readline.createInterface({input:this.input,output:this.output,terminal:Boolean(this.input.isTTY&&this.output.isTTY),historySize:100,completer:complete,prompt:'OpenMagFDM> '});
        this.rl=rl;rl.history=history;
        rl.on('line',line=>{this.chain=this.chain.then(()=>this.execute(line)).catch(e=>this.print(`Error: ${e.message}`)).finally(()=>{if(!this.closing)this.rl?.prompt();});});
        rl.on('SIGINT',()=>{if(rl.line){rl.write(null,{ctrl:true,name:'u'});rl.prompt();}else this.enqueueExit();});
        rl.on('close',()=>{if(!this.replacingEditor&&!this.closing)this.enqueueExit(true);});
    }
    enqueueExit(force=false) {
        this.chain=this.chain.then(()=>this.execute(force?'exit --force':'exit')).catch(e=>{this.print(`Error: ${e.message}`);this.rl?.prompt();});
    }
    async ask(kind,options) {
        if(!this.input.isTTY||!this.output.isTTY)throw Error('This prompt requires an interactive terminal');
        // Keep the command editor detached throughout a command's prompt sequence.
        // Reattaching it between password/confirmation prompts can consume input
        // intended for the next prompt and would also expose secrets to history.
        if(this.rl){this.history=[...this.rl.history];this.replacingEditor=true;this.rl.close();this.rl=null;this.replacingEditor=false;}
        this.secret=true;
        const controller=new AbortController();this.cancelPrompt=()=>controller.abort();
        const eof=()=>{controller.abort();this.enqueueExit(true);};
        const keypress=(_text,key)=>{if(key?.ctrl&&key.name==='d')eof();};
        this.input.once('end',eof);this.input.on('keypress',keypress);
        try {
            const prompts=require('@inquirer/prompts');
            const value=await prompts[kind](options,{input:this.input,output:this.output,clearPromptOnDone:true,signal:controller.signal});
            return value;
        } catch(error) {
            if(['ExitPromptError','AbortPromptError'].includes(error.name))throw Error('Input cancelled');
            throw error;
        } finally {
            this.input.off('end',eof);this.input.off('keypress',keypress);
            this.cancelPrompt=null;this.secret=false;
            if(this.deferredLogs){this.output.write(this.deferredLogs);this.deferredLogs='';}
        }
    }
    password(message){return this.ask('password',{message,mask:true,toggleMask:false});}
    rpc(operation,args=[]) {
        const child=this.child;
        if(!child?.connected) return Promise.reject(Error('Server is stopped. Use start.'));
        return new Promise((resolve,reject)=>{
            const id=++this.id;
            const timeout=operation==='command'?null:setTimeout(()=>{this.requests.delete(id);reject(Error('Server did not respond; check status before retrying.'));},operation==='stop'?40000:10000);
            this.requests.set(id,{resolve,reject,timeout,operation});
            child.send({type:'request',id,operation,args},error=>{if(error){clearTimeout(timeout);this.requests.delete(id);reject(error);}});
        });
    }
    async start() {
        if(this.child)throw Error('Server is already running');
        const args=process.pkg?['--server-child']:[require.resolve('../launcher.js'),'--server-child'];
        const child=spawn(process.execPath,args,{cwd:this.base,env:{...process.env},windowsHide:true,stdio:['ignore','pipe','pipe','ipc']});
        this.child=child;this.expectedExit=false;
        child.stdout.on('data',b=>this.log(b));child.stderr.on('data',b=>this.log(b));
        await new Promise((resolve,reject)=>{
            const timer=setTimeout(()=>reject(Error('Server startup timed out; inspect status or stop --force.')),30000);
            child.on('error',error=>{clearTimeout(timer);reject(error);});
            child.on('message',async message=>{
                if(message?.type==='ready'){clearTimeout(timer);this.print(`Server ready: ${message.value.url}`);resolve();}
                if(message?.type==='reply') {
                    const entry=this.requests.get(message.id);if(!entry)return;
                    this.requests.delete(message.id);clearTimeout(entry.timeout);
                    message.error?entry.reject(Error(message.error)):entry.resolve(message.value);
                }
                if(message?.type==='password') {
                    try {const value=await this.password(message.prompt);if(child.connected)child.send({type:'password-result',id:message.id,value},()=>{});}
                    catch(error){if(child.connected)child.send({type:'password-result',id:message.id,error:error.message},()=>{});}
                }
            });
            child.once('exit',(code,signal)=>{
                this.cancelPrompt?.();
                clearTimeout(timer);if(this.child===child)this.child=null;
                for(const entry of this.requests.values()) {
                    clearTimeout(entry.timeout);
                    if(entry.operation==='stop'&&code===0)entry.resolve();else entry.reject(Error(`Server exited (${code??signal}); shutdown may be incomplete.`));
                }
                this.requests.clear();reject(Error(`Server exited before becoming ready (${code??signal})`));
                if(!this.expectedExit)this.log(`Server exited (${code??signal}). Console remains available; use start after correcting the problem.\n`);
            });
        });
    }
    async stop(force=false) {
        if(!this.child)return;
        this.expectedExit=true;
        try {await this.rpc('stop',force?['--force']:[]);this.print('Server stopped.');}
        catch(error){this.expectedExit=false;throw error;}
    }
    async execute(line) {
        if(this.closing)return;
        const args=tokenize(line);if(!args.length)return;
        args[0]=args[0].replace(/^\//,'');
        this.busy=true;
        try {
            if(['help','?','--help','-h'].includes(args[0]))return this.print(help);
            if(args[0]==='menu') {
                const action=await this.ask('select',{message:'Server action',choices:['status','configure','users list','workspaces list','jobs list',this.child?'stop':'start','exit'].map(value=>({name:value,value}))});
                return await this.execute(action);
            }
            if(args[0]==='configure') {
                const scope=await this.ask('select',{message:'File-limit scope',choices:[{name:'Default for all workspaces',value:'default'},{name:'One workspace',value:'workspace'}]});
                const workspace=scope==='workspace'?await this.ask('input',{message:'Workspace ID',validate:value=>/^[A-Za-z0-9_-]{1,128}$/.test(value)||'Enter a workspace ID'}):null;
                const category=await this.ask('select',{message:'Files to limit',choices:[{name:'Configuration YAML',value:'configs'},{name:'Images',value:'images'},{name:'Material libraries',value:'libraries'}]});
                const count=await this.ask('input',{message:'Maximum files (0, a count, unlimited, or default for an override)',validate:value=>{if(workspace&&value==='default')return true;try{require('./limits').value(value);return true;}catch{return 'Enter a nonnegative integer or unlimited';}}});
                return await this.execute(workspace?`limits set ${workspace} ${category} ${count}`:`limits set-default ${category} ${count}`);
            }
            if(args[0]==='status') {
                if(!this.child)return this.print('Server stopped. Use start.');
                return this.print(JSON.stringify(await this.rpc('status'),null,2));
            }
            if(['start','stop','restart','exit','quit'].includes(args[0])) {
                if(args.slice(1).some(a=>a!=='--force'))throw Error(`Use ${args[0]} [--force]`);
                if(args[0]!=='start')await this.stop(args.includes('--force'));
                if(args[0]==='start'||args[0]==='restart')await this.start();
                if(args[0]==='exit'||args[0]==='quit'){this.closing=true;this.rl?.close();this.releaseInput?.();this.removeSignals();this.print('Console closed.');}
                return;
            }
            if(!['users','workspaces','limits','storage','network','jobs'].includes(args[0]))throw Error('Unknown command. Type help.');
            if(this.child)await this.rpc('command',args);
            else {
                if(args[0]==='jobs')throw Error('Use start before managing jobs.');
                await require('./admin-cli').run(this.base,args,{password:prompt=>this.password(prompt)});
            }
        }finally{this.busy=false;if(!this.closing&&!this.rl)this.editor(this.history||[]);}
    }
    removeSignals(){process.removeListener('SIGTERM',this.terminate);process.removeListener('SIGINT',this.terminate);process.removeListener('exit',this.lastResort);}
    async run() {
        // Own the native TTY for the console's lifetime. Editors share a stream
        // above it, so prompt transitions cannot stop an outstanding Windows
        // console read or reset the native raw mode while another prompt starts.
        if(this.input.isTTY&&this.output.isTTY){
            const native=this.input,channel=new PassThrough(),wasRaw=native.isRaw;
            channel.isTTY=true;channel.setRawMode=()=>channel;
            native.setRawMode(true);native.pipe(channel);native.resume();this.input=channel;
            this.releaseInput=()=>{native.unpipe(channel);native.setRawMode(wasRaw);native.pause();channel.destroy();};
        }
        this.terminate=()=>{this.cancelPrompt?.();void this.stop(true).catch(e=>{this.print(e.message);process.exitCode=1;}).finally(()=>{this.closing=true;this.rl?.close();this.releaseInput?.();this.child?.kill();this.removeSignals();});};
        this.lastResort=()=>this.child?.kill();
        process.once('SIGTERM',this.terminate);process.once('SIGINT',this.terminate);process.once('exit',this.lastResort);
        this.print('OpenMagFDM interactive server console\nType help for commands. Tab: completion | Up/Down: history');
        // Read input immediately, but serialize it after server initialization.
        this.busy=true;this.chain=this.start().catch(e=>this.print(`Error: ${e.message}`)).finally(()=>{this.busy=false;this.rl?.prompt();});
        this.editor();
    }
}
module.exports={ServerConsole,tokenize,complete};
