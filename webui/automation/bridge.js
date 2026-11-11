'use strict';
const path=require('node:path');
const fs=require('node:fs/promises');
const {Client,StreamableHTTPClientTransport}=require('@modelcontextprotocol/client');
// pkg does not resolve package export subpaths while scanning; include this
// known SDK entry explicitly in pkg.scripts and resolve it relative to main.
const {serveStdio}=require(path.join(path.dirname(require.resolve('@modelcontextprotocol/server')),'stdio.cjs'));
const {privateDirectory}=require('./access');
const {noLinks,fail}=require('./workspace');
const {createToolServer}=require('./tools');
// Reconnection design notes:
// - The bridge re-reads the private settings and token profile whenever it
//   (re)connects, so a server restart with a changed port is picked up without
//   restarting the bridge.
// - A tool call is never replayed automatically. A call that fails during a
//   transport outage returns an explicit error and the caller decides whether
//   to retry it, so update operations cannot run twice after a recovery.
// - Authorization refusals (401/403) are reported separately from transport
//   outages so an agent does not treat a restart as a token problem.
const AUTH_MESSAGE='MCP authorization was refused: the connection token is invalid, revoked or the connection is disabled. Recreate or re-enable the connection in WebUI Help & Integrations; running analyses are not cancelled.';
const DISABLED_MESSAGE='MCP is disabled in the WebUI (Help & Integrations). This call did not complete; running analyses are not cancelled.';
const RETRY_MESSAGE='MCP transport was interrupted (WebUI server restart or port change). This call did not complete; the bridge has reconnected to the current server settings. Retry the same operation if needed; running analyses are not cancelled. Update operations are never replayed automatically.';
const OFFLINE_MESSAGE='MCP transport is unavailable (the WebUI server may be stopped or restarting). This call did not complete. Retry after the server is back; running analyses are not cancelled. Update operations are never replayed automatically.';
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
function authorizationRefused(e) {
    return e?.name==='UnauthorizedError'||e?.status===401||e?.status===403||e?.code==='CLIENT_HTTP_AUTHENTICATION'||e?.code==='CLIENT_HTTP_FORBIDDEN';
}
function mcpDisabled(e) {return e?.status===503||e?.code==='MCP_DISABLED';}
function transportInterrupted(e) {
    if(typeof e?.code==='number')return false; // JSON-RPC protocol error: the server answered.
    if(typeof e?.code==='string')return ['NOT_CONNECTED','CONNECTION_CLOSED','SEND_FAILED','REQUEST_TIMEOUT','CLIENT_HTTP_FAILED_TO_OPEN_STREAM','CLIENT_HTTP_NOT_IMPLEMENTED','CLIENT_HTTP_FAILED_TO_TERMINATE_SESSION','CLIENT_HTTP_UNEXPECTED_CONTENT'].includes(e.code);
    return true; // fetch failures (ECONNREFUSED, socket reset, ...) and unknown errors.
}
function detail(e) {return e?.message||String(e);}
class StdioBridge {
    constructor({base,directory,id}) {
        this.base=base;this.directory=directory;this.id=id;
        this.snapshot=null;this.client=null;this.broken=false;this.closing=false;this.reconnecting=null;
    }
    async loadSnapshot() {
        // Atomic JSON writes make the rename a read barrier and noLinks rejects
        // link swaps; a failed reload keeps the last known-good snapshot
        // instead of following a replaced file.
        try {
            const settingsPath=path.join(this.directory,'settings.json'),profilePath=path.join(this.directory,`${this.id}.json`);
            await noLinks(settingsPath);await noLinks(profilePath);
            const settings=JSON.parse(await fs.readFile(settingsPath,'utf8'));
            if(!settings.enabled)fail('MCP_DISABLED','MCP is disabled in the WebUI');
            const connection=settings.connections.find(c=>c.id===this.id&&!c.revoked);
            if(!connection)fail('MCP_CONNECTION_REVOKED','This connection is revoked or removed');
            const {token}=JSON.parse(await fs.readFile(profilePath,'utf8'));
            this.snapshot={port:settings.port,token};
        }catch(e) {if(!this.snapshot)throw e;console.error(`MCP bridge settings reload failed; using the last known settings (${detail(e)})`);}
        return this.snapshot;
    }
    async open() {
        const snapshot=await this.loadSnapshot();
        if(this.client)await this.client.close().catch(()=>{});
        const client=new Client({name:'OpenMagFDM-stdio-bridge',version:'1.0.0'},{versionNegotiation:{mode:'auto'}});
        const transport=new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${snapshot.port}/mcp`),{requestInit:{headers:{Authorization:`Bearer ${snapshot.token}`}}});
        try {await client.connect(transport,{timeout:5000});}
        catch(e){await client.close().catch(()=>{});throw e;}
        if(this.closing) {await client.close().catch(()=>{});fail('MCP_CLOSED','The MCP bridge is closing');}
        this.client=client;this.broken=false;
    }
    async connect(options={}) {
        // Wait a bounded time for a server that starts after the bridge, e.g.
        // when both are launched together or the WebUI is restarting.
        const windowMs=options.connectWindowMs??30000,baseDelayMs=options.connectBaseDelayMs??250,maxDelayMs=options.connectMaxDelayMs??2000;
        const deadline=Date.now()+windowMs;let first=true;
        for(let attempt=0;;attempt++) {
            try {await this.open();return;}
            catch(e) {
                if(this.closing||authorizationRefused(e)||mcpDisabled(e)) {
                    console.error(`MCP bridge startup stopped: ${this.closing?'client disconnected':authorizationRefused(e)?AUTH_MESSAGE:DISABLED_MESSAGE}`);
                    throw e;
                }
                if(Date.now()>=deadline) {console.error(`MCP bridge could not reach the WebUI server after ${windowMs} ms: ${detail(e)}`);throw e;}
                if(first) {console.error(`MCP bridge is waiting for the WebUI server (${detail(e)})`);first=false;}
                await sleep(Math.min(maxDelayMs,baseDelayMs*2**Math.min(attempt,4)));
            }
        }
    }
    async reconnect() {
        this.reconnecting??=(async()=>{
            try {await this.open();return true;}
            catch(e){this.broken=true;return false;}
            finally{this.reconnecting=null;}
        })();
        return this.reconnecting;
    }
    async call(name,input) {
        if(this.broken)await this.reconnect();
        try {return await this.client.callTool({name,arguments:input});}
        catch(e) {
            if(authorizationRefused(e))return {isError:true,content:[{type:'text',text:AUTH_MESSAGE}]};
            if(mcpDisabled(e))return {isError:true,content:[{type:'text',text:DISABLED_MESSAGE}]};
            if(transportInterrupted(e)) {
                this.broken=true;
                const recovered=await this.reconnect();
                return {isError:true,content:[{type:'text',text:recovered?RETRY_MESSAGE:OFFLINE_MESSAGE}]};
            }
            return {isError:true,content:[{type:'text',text:`MCP call failed: ${detail(e)}`}]};
        }
    }
    close() {
        this.closing=true;
        return Promise.resolve(this.client?.close()).catch(()=>{});
    }
}
async function start(base,args,options={}) {
    const id=args[args.indexOf('--connection')+1];
    if(!/^[a-f0-9-]{36}$/.test(id||''))fail('INVALID_CONNECTION','A connection ID is required');
    const directory=options.directory||privateDirectory(base);
    const bridge=new StdioBridge({base,directory,id});
    let handle=null;
    const close=async()=>{
        if(bridge.closing)return;
        await bridge.close();
        await handle?.close();
    };
    process.stdin.once('end',()=>void close());
    process.once('SIGINT',()=>void close());process.once('SIGTERM',()=>void close());
    await bridge.connect(options);
    handle=await serveStdio(()=>createToolServer(async(name,input)=>bridge.call(name,input)),{onerror:()=>console.error('MCP bridge protocol error')});
    if(bridge.closing)await handle.close();
    return {close};
}
module.exports={start,authorizationRefused,mcpDisabled,transportInterrupted};
