'use strict';
// SEA uses official Node binaries, independently of pkg-fetch's older runtime.
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {createHash} = require('node:crypto');
const webui = path.resolve(__dirname, '..');
const version = fs.readFileSync(path.join(webui, '../.node-version'), 'utf8').trim();
const platform = {win32:'win', darwin:'macos', linux:'linux'}[process.platform];
const nativeTarget = `node${version}-${platform}-${process.arch}`;
const args = process.argv.slice(2);
function option(name, fallback) {
    const index = args.indexOf(name);
    if (index < 0) return fallback;
    if (!args[index + 1] || args[index + 1].startsWith('--')) throw Error(`Missing ${name}`);
    return args[index + 1];
}
async function main() {
    if (process.versions.node !== version) throw Error(`Build with Node ${version}; found ${process.versions.node}`);
    if (!platform) throw Error('Unsupported packaging platform');
    for (let i = 0; i < args.length; i += 2) if (!['--target','--output'].includes(args[i])) throw Error(`Unknown option: ${args[i]}`);
    const requested = option('--target', nativeTarget);
    const target = /^(win|linux|macos)-(x64|arm64)$/.test(requested) ? `node${version}-${requested}` : requested;
    if (!target.startsWith(`node${version}-`)) throw Error(`Target must pin the security-reviewed Node ${version}`);
    const output = path.resolve(option('--output', path.join(webui, '../build_mcp', process.platform === 'win32' ? 'webui-server.exe' : 'webui-server')));
    process.chdir(webui);
    await require('@yao-pkg/pkg').exec(['package.json','--sea','--targets',target,'--output',output]);
    let runtime = null;
    if (target === nativeTarget) {
        runtime = JSON.parse(execFileSync(output, ['--runtime-info'], {encoding:'utf8',timeout:15000,windowsHide:true}));
        if (runtime.node !== version || !runtime.packaged) throw Error(`Unexpected packaged runtime: ${JSON.stringify(runtime)}`);
    }
    fs.writeFileSync(`${output}.runtime.json`, JSON.stringify({target,node:version,mode:'sea',runtimeVerified:Boolean(runtime),runtime,sha256:createHash('sha256').update(fs.readFileSync(output)).digest('hex')},null,2));
    console.log(`Built ${target}; native runtime verification: ${Boolean(runtime)}`);
}
main().catch(error => { console.error(error.stack); process.exitCode = 1; });
