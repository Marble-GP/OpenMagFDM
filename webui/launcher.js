'use strict';
// Dispatch BEFORE loading server.js: a stdio client must never start another WebUI.
const path=require('node:path');
const base=process.pkg?path.dirname(process.execPath):path.join(__dirname,'..');
if(process.argv.includes('--mcp-stdio')) {
    require('./automation/bridge').start(base,process.argv).catch(()=>{console.error('MCP connection unavailable. Start OpenMagFDM, enable MCP, and check the connection profile in Help / Integrations.');process.exitCode=1;});
} else if(process.argv.includes('--mcp-admin-code')) {
    const {privateDirectory}=require('./automation/access');
    const {noLinks}=require('./automation/workspace');
    const fs=require('node:fs/promises');
    const filename=path.join(privateDirectory(base),'pairing.json');
    noLinks(filename).then(()=>fs.readFile(filename,'utf8')).then(text=>console.log(JSON.parse(text).code)).catch(()=>{console.error('Start the WebUI server first.');process.exitCode=1;});
} else require('./server');
