'use strict';
// Private parent/child IPC only; no HTTP administration endpoint is added.
const {once}=require('node:events');
async function start(base) {
    if(!process.send)throw Error('Server child requires its supervising console');
    const server=require('../server');
    const listener=server.startServer();
    const send=message=>{if(process.connected)process.send(message,()=>{});};
    const passwords=new Map();let passwordId=0, stopping=false;
    const readPassword=prompt=>new Promise((resolve,reject)=>{const id=++passwordId;passwords.set(id,{resolve,reject});send({type:'password',id,prompt});});
    async function shutdown() {
        if(stopping)return;
        stopping=true;
        for(const p of passwords.values())p.reject(Error('Server is stopping'));
        const result=await server.shutdownServer('console');
        process.exit(result.clean?0:1);
    }
    process.on('disconnect',()=>{void shutdown().catch(()=>process.exit(1));});
    // Windows can terminate the supervisor without delivering a JS signal.
    // Do not depend on inherited pipe handles closing to detect that case.
    const supervisor=process.ppid;
    const parentCheck=setInterval(()=>{
        try {process.kill(supervisor,0);}catch(error){if(error.code==='ESRCH')void shutdown().catch(()=>process.exit(1));}
    },1000);
    parentCheck.unref();
    const status=()=>({pid:process.pid,mode:require('./storage').settings(base).mode,url,ready:server.automation.ready,error:server.automation.error,
        activity:server.consoleActivity(),jobs:server.automation.jobs.records.map(j=>({workspace:j.user,...server.automation.jobs.public(j)}))});
    let chain=Promise.resolve(),url;
    process.on('message',message=>{
        if(message?.type==='password-result') {
            const pending=passwords.get(message.id);passwords.delete(message.id);
            if(pending)message.error?pending.reject(Error(message.error)):pending.resolve(message.value);
            return;
        }
        if(message?.type!=='request')return;
        chain=chain.then(async()=>{
            try {
                const {operation,args=[]}=message;
                if(operation==='status')return send({type:'reply',id:message.id,value:status()});
                if(operation==='stop') {
                    if(!args.includes('--force')&&Object.values(server.consoleActivity()).some(n=>n>0))throw Error('Analyses are active. Use jobs to inspect/cancel, or stop --force to terminate them.');
                    return await shutdown();
                }
                if(operation!=='command'||!Array.isArray(args)||args.some(a=>typeof a!=='string'||a.length>4096))throw Error('Invalid console request');
                if(args[0]==='jobs') {
                    const jobs=server.automation.jobs;
                    if(!server.automation.ready)throw Error('Job service is not ready');
                    if(args[1]==='cancel'&&args.length===4)console.log(JSON.stringify(await jobs.cancel(require('./workspace').userKey(args[2]),args[3],()=>{}),null,2));
                    else if(args[1]==='resume'&&args.length===3)console.log(JSON.stringify(await jobs.resume(require('./workspace').userKey(args[2])),null,2));
                    else if(args.length===1||(args[1]==='list'&&args.length<=3))console.table(status().jobs.filter(j=>!args[2]||j.workspace===args[2]).slice(-100).map(({workspace,jobId,status,progress})=>({workspace,jobId,status,progress})));
                    else throw Error('Use jobs [list [workspace]], jobs cancel <workspace> <jobId>, or jobs resume <workspace>');
                } else {
                    if(!['users','workspaces','limits','storage','network'].includes(args[0]))throw Error('Unknown command. Type help.');
                    await require('./admin-cli').run(base,args,{live:true,password:readPassword});
                }
                send({type:'reply',id:message.id,value:true});
            }catch(error){send({type:'reply',id:message.id,error:error.message});}
        });
    });
    if(!listener.listening)await once(listener,'listening');
    await server.automation.initializing;
    const settings=require('./storage').settings(base);
    url=`${settings.mode==='shared'?'https':'http'}://localhost:${listener.address().port}`;
    send({type:'ready',value:status()});
}
module.exports={start};
