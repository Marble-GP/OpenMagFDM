'use strict';
const defaults = Object.freeze({configs:20, images:20, libraries:null, results:null, 'result-mib':null});
// Live transfer reservations are memory-only and shared by every admission path.
const reservationSources=new Set();
function registerReservations(source) {reservationSources.add(source);return ()=>reservationSources.delete(source);}
function value(text) {
    if(text === 'unlimited') return null;
    if(!/^(0|[1-9][0-9]*)$/.test(String(text)) || !Number.isSafeInteger(Number(text))) throw Error('Limit must be a nonnegative integer or unlimited');
    return Number(text);
}
function effective(settings, workspace) {
    const overrides=settings.fileLimits?.workspaces;
    const result={...defaults,...settings.fileLimits?.defaults,...(overrides && Object.hasOwn(overrides,workspace) ? overrides[workspace] : {})};
    for(const [key,count] of Object.entries(result)) if(!Object.hasOwn(defaults,key) || (count!==null && (!Number.isSafeInteger(count)||count<0))) throw Error('Invalid file limit settings');
    return result;
}
// Read actual storage; no quota cache or per-file accounting journal.
async function resultUsage(base, workspace, bytes = true) {
    const fs=require('node:fs/promises'),path=require('node:path');
    const {noLinks,fail}=require('./workspace');
    const directory=path.join(require('./storage').root(base,'outputs'),workspace);
    let count=0,size=0;
    async function visit(dir,depth=0) {
        if(depth>64)fail('LIMIT_EXCEEDED','Result storage nesting exceeds 64 levels',413);
        let entries;
        try {await noLinks(dir);entries=await fs.readdir(dir,{withFileTypes:true});}
        catch(e){if(e.code==='ENOENT'&&depth===0)return;throw e;}
        for(const entry of entries) {
            const p=path.join(dir,entry.name),st=await fs.lstat(p);
            if(st.isSymbolicLink()||(st.isFile()&&st.nlink>1))fail('UNSAFE_PATH','Linked result storage is not supported');
            if(st.isDirectory()){if(depth===0 && entry.name!=='.mcp-transfers')count++;if(bytes)await visit(p,depth+1);}
            else if(st.isFile()){if(bytes)size+=st.size;}
            else fail('UNSAFE_PATH','Non-regular result storage entry');
        }
    }
    await visit(directory);
    return {count,bytes:size};
}
async function checkResults(base,workspace,{newResults=0,bytes=0,starting=false}={}) {
    const limits=effective(require('./storage').settings(base),workspace);
    if(limits.results===null&&limits['result-mib']===null)return;
    // Capture before scanning: chunks arriving during the scan may then be
    // counted twice conservatively, but never disappear from both quantities.
    const reserved=[...reservationSources].reduce((sum,source)=>sum+source(require('node:path').resolve(base),workspace),0);
    const usage=await resultUsage(base,workspace,limits['result-mib']!==null);
    const {fail}=require('./workspace');
    if(limits.results!==null&&usage.count+newResults>limits.results)
        fail('QUOTA_EXCEEDED',`Result limit (${limits.results} folders) reached. Remove results or ask the administrator to adjust limits.`,409);
    if(limits['result-mib']!==null&&(usage.bytes+bytes+reserved>limits['result-mib']*1024*1024||(starting&&usage.bytes+reserved>=limits['result-mib']*1024*1024)))
        fail('QUOTA_EXCEEDED',`Result storage limit (${limits['result-mib']} MiB) reached. Existing results were retained.`,409);
}
async function checkResultWrite(ws,id,bytes,current) {
    if(!id.startsWith('results/'))return;
    const limits=effective(require('./storage').settings(ws.base),ws.user);
    if(limits.results===null&&limits['result-mib']===null)return;
    const fs=require('node:fs/promises');
    const parent=id.split('/').slice(0,2).join('/');
    const exists=await fs.stat(await ws.resolve(parent,{create:true})).then(()=>true,e=>{if(e.code==='ENOENT')return false;throw e;});
    const growth=Math.max(0,bytes-(current?.size||0));
    // Keep edits which reduce storage possible when an administrator lowers a cap.
    if(exists&&!growth)return;
    await checkResults(ws.base,ws.user,{newResults:exists?0:1,bytes:growth});
}
module.exports={defaults,value,effective,resultUsage,checkResults,checkResultWrite,registerReservations};
