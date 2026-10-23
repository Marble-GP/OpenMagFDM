'use strict';
// Dispatch BEFORE loading server.js: a stdio client must never start another WebUI.
const path=require('node:path');
const base=process.pkg?path.dirname(process.execPath):path.join(__dirname,'..');
if(process.argv[2]==='--runtime-info') {
    console.log(JSON.stringify({node:process.versions.node,versions:process.versions,platform:process.platform,arch:process.arch,packaged:Boolean(process.pkg)},null,2));
} else if(['help','--help','-h','users','workspaces','storage','network','limits'].includes(process.argv[2])) {
    require('./automation/admin-cli').run(base,process.argv.slice(2)).catch(error=>{console.error(error.message);process.exitCode=1;});
} else if(process.argv.includes('--mcp-stdio')) {
    require('./automation/bridge').start(base,process.argv).catch(()=>{console.error('MCP connection unavailable. Start OpenMagFDM, enable MCP, and check the connection profile in Help / Integrations.');process.exitCode=1;});
} else if(process.argv.includes('--mcp-admin-code')) {
    const {privateDirectory}=require('./automation/access');
    const {noLinks}=require('./automation/workspace');
    const fs=require('node:fs/promises');
    const filename=path.join(privateDirectory(base),'pairing.json');
    noLinks(filename).then(()=>fs.readFile(filename,'utf8')).then(text=>console.log(JSON.parse(text).code)).catch(()=>{console.error('Start the WebUI server first.');process.exitCode=1;});
} else if(process.argv.includes('--server-child')) {
    require('./automation/console-child').start(base).catch(error=>{console.error(error.message);process.exitCode=1;});
} else if(!process.argv.includes('--no-console')&&(process.argv.includes('--console')||(process.stdin.isTTY&&process.stdout.isTTY))) {
    new (require('./automation/server-console').ServerConsole)(base).run().catch(error=>{console.error(error.message);process.exitCode=1;});
} else require('./server');
