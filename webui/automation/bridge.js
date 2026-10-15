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
async function start(base,args,options={}) {
    const id=args[args.indexOf('--connection')+1];
    if(!/^[a-f0-9-]{36}$/.test(id||''))fail('INVALID_CONNECTION','A connection ID is required');
    const directory=options.directory||privateDirectory(base);
    const settingsPath=path.join(directory,'settings.json'),profilePath=path.join(directory,`${id}.json`);
    await noLinks(settingsPath);await noLinks(profilePath);
    const settings=JSON.parse(await fs.readFile(settingsPath,'utf8'));
    const connection=settings.connections.find(c=>c.id===id&&!c.revoked);
    if(!settings.enabled||!connection)fail('MCP_DISABLED','Enable this connection in the WebUI first');
    const {token}=JSON.parse(await fs.readFile(profilePath,'utf8'));
    const client=new Client({name:'OpenMagFDM-stdio-bridge',version:'1.0.0'},{versionNegotiation:{mode:'auto'}});
    try {
        await client.connect(new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${settings.port}/mcp`),{requestInit:{headers:{Authorization:`Bearer ${token}`}}}),{timeout:5000});
    }catch(e){await client.close();throw e;}
    const handle=await serveStdio(()=>createToolServer(async(name,input)=>{
        try{return await client.callTool({name,arguments:input});}
        catch{return {isError:true,content:[{type:'text',text:'MCP unavailable or access revoked. Check Help / Integrations; running analyses are not cancelled.'}]};}
    }),{onerror:()=>console.error('MCP bridge protocol error')});
    let closing=false;
    const close=async()=>{if(closing)return;closing=true;await handle.close();await client.close();};
    process.stdin.once('end',()=>void close());
    process.once('SIGINT',()=>void close());process.once('SIGTERM',()=>void close());
    return {close};
}
module.exports={start};
