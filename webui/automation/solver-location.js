'use strict';
const fs=require('node:fs');
const path=require('node:path');
const {fail}=require('./workspace');
function available(filename) {
    try {return fs.statSync(filename).isFile() && (process.platform==='win32' || (fs.accessSync(filename,fs.constants.X_OK),true));} catch {return false;}
}
function validate(filename) {
    if(typeof filename!=='string'||!path.isAbsolute(filename)||filename.includes('\0')||!available(filename) || (process.platform==='win32'&&!filename.toLowerCase().endsWith('.exe')))
        fail('INVALID_SOLVER','Specify an existing solver executable using its full path');
    return filename;
}
function locate(base,saved='',env=process.env.SOLVER_PATH) {
    const name=process.platform==='win32'?'MagFDMsolver.exe':'MagFDMsolver';
    const candidates=[path.join(base,name),path.join(base,'build','Release',name),path.join(base,'build',name)];
    const selected=env||saved||candidates.find(available)||candidates[0];
    return {path:selected,available:available(selected),source:env?'environment':saved?'saved':'automatic'};
}
module.exports={locate,validate};
