'use strict';
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),net=require('node:net');
const {once}=require('node:events'),{createHash}=require('node:crypto');
async function createFixture({exe,solver}={}){
    const repo=path.resolve(__dirname,'../..'),root=await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()),'omfdm-console-')),base=path.join(root,'app');
    await fs.mkdir(path.join(base,'webui'),{recursive:true});
    for(const name of ['launcher.js','server.js'])await fs.copyFile(path.join(repo,'webui',name),path.join(base,'webui',name));
    for(const name of ['automation','public'])await fs.cp(path.join(repo,'webui',name),path.join(base,'webui',name),{recursive:true});
    for(const name of ['sample_config.yaml','general_materials.yaml'])await fs.copyFile(path.join(repo,name),path.join(base,name));
    const probe=net.createServer().listen(0,'127.0.0.1');await once(probe,'listening');const port=probe.address().port;await new Promise(r=>probe.close(r));
    const env={...process.env,PORT:String(port),HOME:path.join(root,'private'),LOCALAPPDATA:path.join(root,'private'),NODE_PATH:path.join(repo,'webui/node_modules'),OMP_NUM_THREADS:'2'};
    delete env.OPENMAGFDM_SKIP_INIT;delete env.OPENMAGFDM_NO_LISTEN;
    if(solver)env.SOLVER_PATH=path.resolve(solver);else delete env.SOLVER_PATH;
    let executable=process.execPath,args=[path.join(base,'webui/launcher.js')];
    if(exe){
        executable=path.join(base,process.platform==='win32'?'webui-server.exe':'webui-server');args=[];
        await fs.copyFile(path.resolve(exe),executable);if(process.platform!=='win32')await fs.chmod(executable,0o700);
        await fs.cp(path.join(repo,'webui/public'),path.join(base,'public'),{recursive:true});delete env.NODE_PATH;
        env.PATH=(env.PATH||'').split(path.delimiter).filter(p=>path.resolve(p)!==path.dirname(process.execPath)).join(path.delimiter);
    }
    const privateHome=process.platform==='win32'?env.LOCALAPPDATA:path.join(env.HOME,'.local','share');
    const privateDir=path.join(privateHome,'OpenMagFDM','automation',createHash('sha256').update(base).digest('hex').slice(0,20));
    return {repo,root,base,port,env,executable,args,privateDir};
}
module.exports={createFixture};
