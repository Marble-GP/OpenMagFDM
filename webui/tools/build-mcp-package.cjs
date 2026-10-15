'use strict';
const path=require('node:path');
const platform={win32:'win',darwin:'macos',linux:'linux'}[process.platform];
if(!platform)throw Error('Unsupported packaging platform');
process.chdir(path.join(__dirname,'..'));
require('@yao-pkg/pkg').exec(['package.json','--targets',`node24-${platform}-${process.arch}`,'--output',path.join('..','build_mcp',process.platform==='win32'?'webui-server.exe':'webui-server')])
    .catch(error=>{console.error(error.message);process.exitCode=1;});
