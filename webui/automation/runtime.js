'use strict';
const http=require('node:http');
const path=require('node:path');
const fs=require('node:fs/promises');
const {createMcpHandler}=require('@modelcontextprotocol/server');
const {toNodeHandler}=require('@modelcontextprotocol/node');
const {Client,StreamableHTTPClientTransport}=require('@modelcontextprotocol/client');
const {Access,localRequest}=require('./access');
const {AnalysisJobs}=require('./jobs');
const {Images}=require('./images');
const {Transfers}=require('./transfers');
const {Workspace,fail,locked,noLinks}=require('./workspace');
const {serverFor}=require('./tools');
class AutomationRuntime {
    constructor(base,backend,options={}) {
        this.base=base;this.backend=backend;this.access=new Access(base,options);this.jobs=new AnalysisJobs(base,this.access,backend);
        this.images=new Images(backend);this.transfers=new Transfers();this.listener=null;this.handler=null;this.error=null;this.ready=false;this.active=0;this.closed=false;
    }
    async init() {
        this.initializing=(async()=>{
            try {await this.access.init();this.backend.configureSolver?.(this.access.state.solverPath);await this.jobs.init();if(this.closed){this.access.ready=false;return;}this.ready=true;await this.reconfigure();}
            catch(e){this.error=`MCP unavailable (${e.code||'INITIALIZATION_FAILED'})`;this.access.ready=false;this.ready=false;}
        })();
        return this.initializing;
    }
    status() {return {...this.access.summary(),ready:this.ready,listening:!!this.listener?.listening,error:this.error,url:`http://127.0.0.1:${this.access.state.port}/mcp`};}
    async stopListener() {
        const server=this.listener;this.listener=null;
        if(this.handler){await this.handler.close();this.handler=null;}
        if(server){server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}
    }
    async reconfigure() {
        return locked(`${this.access.filename}:listener`,async()=>{
            await this.stopListener();this.error=null;
            if(this.closed||!this.ready||!this.access.state.enabled)return;
            this.handler=createMcpHandler(ctx=>serverFor(this.access.actor(ctx.authInfo.clientId),this),{legacy:'stateless',maxRequestBodySize:2*1024*1024,maxSubscriptions:8});
            const handle=toNodeHandler(this.handler,{maxRequestBodySize:2*1024*1024});
            const server=http.createServer(async(req,res)=>{
                res.setHeader('Cache-Control','no-store');
                let counted=false;
                try {
                    localRequest(req);
                    if(req.url!=='/mcp')fail('NOT_FOUND','Not found',404);
                    const actor=this.access.authenticate(req.headers.authorization);
                    if(this.active>=8)fail('BUSY','Too many concurrent MCP requests',429);
                    this.active++;counted=true;
                    req.auth={token:'',clientId:actor.id,scopes:[]};
                    await handle(req,res);
                }catch(e){if(!res.headersSent){res.writeHead(e.statusCode||500,{'Content-Type':'application/json'});res.end(JSON.stringify({error:{code:e.code||'INTERNAL_ERROR',message:e.statusCode?e.message:'MCP request failed'}}));}}
                finally{if(counted)this.active--;}
            });
            server.requestTimeout=30000;server.headersTimeout=10000;server.keepAliveTimeout=5000;
            try {
                await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(this.access.state.port,'127.0.0.1',()=>{server.removeListener('error',reject);resolve();});});
                this.listener=server;
                server.on('error',e=>{this.error=`MCP listener failed (${e.code||'IO_ERROR'})`;});
            }catch(e){this.error=`MCP port ${this.access.state.port} unavailable (${e.code||'BIND_FAILED'}); WebUI remains available`;await this.handler.close();this.handler=null;}
        });
    }
    async close(){
        this.closed=true;this.jobs.stop();this.ready=false;this.access.ready=false;
        await this.initializing;
        await locked(`${this.access.filename}:listener`,()=>this.stopListener());
        await this.transfers.close();await this.images.close?.();await this.access.flush();
    }
    async testConnection(id) {
        const actor=this.access.actor(id);actor.guard('read');
        const filename=path.join(this.access.directory,`${id}.json`);await noLinks(filename);
        const {token}=JSON.parse(await fs.readFile(filename,'utf8'));
        const client=new Client({name:'OpenMagFDM-connection-test',version:'1.0.0'},{versionNegotiation:{mode:'auto'}});
        try {
            await client.connect(new StreamableHTTPClientTransport(new URL(this.status().url),{requestInit:{headers:{Authorization:`Bearer ${token}`}}}),{timeout:5000});
            const result=await client.callTool({name:'workspace_status',arguments:{}});
            if(result.isError)fail('CONNECTION_FAILED','MCP tool call failed',502);
            return {connected:true,workspace:actor.user,transport:'Streamable HTTP'};
        }finally{await client.close();}
    }
    install(app) {
        const route=(fn,mutate=false)=>async(req,res)=>{
            res.set('Cache-Control','no-store');
            try {if(!this.ready)fail('UNAVAILABLE',this.error||'MCP is initializing',503);const session=this.access.admin(req,mutate);res.json(await fn(req,res,session));}
            catch(e){res.status(e.statusCode||500).json({error:e.statusCode?e.message:'Management operation failed',code:e.code||'INTERNAL_ERROR'});}
        };
        app.post('/api/mcp/pair',async(req,res)=>{
            try{
                localRequest(req);if(!this.ready)fail('UNAVAILABLE',this.error||'MCP is initializing',503);
                const remember=req.body?.remember===true;
                const pair=this.access.pair(req.body?.code,remember);
                try {await this.access.saveBrowsers();} catch(e) {this.access.sessions.delete(require('./access').hash(pair.token));throw e;}
                res.cookie('omfdm_admin',pair.token,{httpOnly:true,sameSite:'strict',...(remember?{maxAge:pair.maxAge}:{}),path:'/api/mcp'});
                res.set('Cache-Control','no-store').json({csrf:pair.csrf});
            }catch(e){res.status(e.statusCode||500).json({error:e.statusCode?e.message:'Pairing failed',code:e.code||'INTERNAL_ERROR'});}
        });
        app.get('/api/mcp/status',route((req,res,session)=>({...this.status(),csrf:session.csrf,browser:{remember:!!session.remember,expires:session.expires}})));
        app.post('/api/mcp/forget-browser',route(async(req,res)=>{
            await this.access.forgetBrowser(req);
            res.clearCookie('omfdm_admin',{httpOnly:true,sameSite:'strict',path:'/api/mcp'});
            return {forgotten:true};
        },true));
        app.get('/api/mcp/solver',route(()=>({...this.backend.solverStatus?.(),savedPath:this.access.state.solverPath||''})));
        app.put('/api/mcp/solver',route(async req=>{
            const filename=req.body?.path;
            if(filename!=='')require('./solver-location').validate(filename);
            await this.access.change(next=>{next.solverPath=filename;});
            return {saved:true,restartRequired:true};
        },true));
        app.put('/api/mcp/settings',route(async req=>{
            const {enabled,port}=req.body||{};
            if(typeof enabled!=='boolean'||!Number.isInteger(port)||port<1024||port>65535)fail('INVALID_INPUT','Expected enabled:boolean and port:1024..65535');
            await this.access.change(next=>{next.enabled=enabled;next.port=port;});await this.reconfigure();return this.status();
        },true));
        app.post('/api/mcp/connections',route(async req=>{
            const {name,user,permissions}=req.body||{};
            const ws=new Workspace(this.base,user);await ws.init();
            return this.access.create({name,user,permissions});
        },true));
        app.patch('/api/mcp/connections/:id',route(req=>this.access.update(req.params.id,req.body||{}),true));
        app.post('/api/mcp/connections/:id/test',route(req=>this.testConnection(req.params.id),true));
        app.get('/api/mcp/setup/:id',route(req=>{
            const c=this.access.state.connections.find(c=>c.id===req.params.id);
            if(!c||c.revoked)fail('NOT_FOUND','Connection not found',404);
            return {url:this.status().url,headers:{Authorization:'Bearer <token shown once when created>'},
                stdio:{command:process.execPath,args:[...(process.pkg?[]:[path.join(__dirname,'..','launcher.js')]),'--mcp-stdio','--connection',c.id]},
                note:'The WebUI server must already be running with MCP enabled. No credentials are included in command-line arguments.'};
        }));
        app.get('/api/automation/activity',async(req,res)=>{
            try{localRequest(req);const ws=new Workspace(this.base,req.query.userId);res.set('Cache-Control','no-store').json({jobs:this.jobs.list(ws.user),events:this.access.events.filter(e=>e.user===ws.user).slice(-50)});}
            catch(e){res.status(e.statusCode||500).json({error:'Activity unavailable'});}
        });
        app.post('/api/automation/jobs/:id/cancel',async(req,res)=>{
            try{localRequest(req);const ws=new Workspace(this.base,req.body.userId);res.json(await this.jobs.cancel(ws.user,req.params.id,()=>{}));}
            catch(e){res.status(e.statusCode||500).json({error:e.statusCode?e.message:'Cancellation failed'});}
        });
    }
}
module.exports={AutomationRuntime};
