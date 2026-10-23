'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync(require.resolve('../public/app.js'),'utf8');
function fixture(files) {
    const elements={fileManagerStatus:{},userFilesList:{},deleteSelectedUserFilesBtn:{},userFileSelectedCount:{},selectAllUserFiles:{}};
    const state={userId:'alice',userFiles:files,userFileSelection:new Set()};
    const context=vm.createContext({AppState:state,document:{getElementById:id=>elements[id],querySelectorAll:()=>[]},console,AbortSignal,performance,
        WorkspaceCore:require('../public/workspace-core'),confirm:()=>true,fmtBytes:String,
        refreshResultsList:async()=>{},refreshConfigList:async()=>{},refreshImageList:async()=>{},clearFilePreview(){},setTimeout:fn=>{fn();}});
    vm.runInContext(source.slice(source.indexOf('function fileManagerEscape'),source.indexOf('/**\n * Refresh the list of output folders')),context);
    return {context,state,elements};
}
const response=data=>({ok:true,json:async()=>data});
test('application-owned public UI source uses English',()=>{
    const path=require('node:path'),root=path.dirname(require.resolve('../public/index.html'));
    for(const name of fs.readdirSync(root))if(/\.(html|js)$/.test(name)) {
        assert.doesNotMatch(fs.readFileSync(path.join(root,name),'utf8'),/[\u3040-\u30ff\u3400-\u9fff]/,`${name} contains Japanese public text`);
    }
    assert.match(fs.readFileSync(path.join(root,'index.html'),'utf8'),/<html lang="en">/);
    assert.match(fs.readFileSync(path.join(root,'login.html'),'utf8'),/<html lang="en">/);
});
test('partial deletion retains failure details and failed selection after list refresh and restores controls',async()=>{
    const files=[{category:'configs',name:'ok.yaml'},{category:'configs',name:'busy.yaml'}],f=fixture(files);
    f.context.fetch=async(url)=>{
        if(url.startsWith('/api/workspace/file'))return response({file:{revision:'a'.repeat(64)}});
        if(url==='/api/file-operations')return response({success:true,operation:{id:'op1',status:'complete',results:[{id:'configs/ok.yaml',success:true},{id:'configs/busy.yaml',success:false,code:'EACCES',error:'Permission denied'}]}});
        return response({success:true,files:[files[1]]});
    };
    await f.context.deleteUserFiles(files);
    assert.equal(f.state.fileDeleteBusy,false);assert.equal(f.elements.deleteSelectedUserFilesBtn.disabled,false);
    assert.equal(f.state.userFileSelection.size,1);assert.ok(f.state.userFileSelection.has('configs\0busy.yaml'));
    assert.match(f.elements.fileManagerStatus.textContent,/Deletion failed.*1 deleted, 1 failed/s);
    assert.match(f.elements.fileManagerStatus.textContent,/busy.yaml: Permission denied \(EACCES\)/);
    assert.equal(f.elements.fileManagerStatus.className,'status-error');
});
test('failed metadata fetch preserves the actual error after refresh and permits retry',async()=>{
    const files=[{category:'results',name:'large'}],f=fixture(files);
    f.context.fetch=async url=>url.startsWith('/api/workspace/file')?{ok:false,json:async()=>({error:'Storage unavailable'})}:response({success:true,files});
    await f.context.deleteUserFiles(files);
    assert.match(f.elements.fileManagerStatus.textContent,/Storage unavailable/);
    assert.equal(f.state.fileDeleteBusy,false);assert.equal(f.elements.deleteSelectedUserFilesBtn.disabled,false);
});
test('selecting over 100 files uses multiple durable operations',async()=>{
    const files=Array.from({length:101},(_,i)=>({category:'configs',name:`${i}.yaml`})),f=fixture(files),counts=[];
    f.context.fetch=async(url,options)=>{
        if(url.startsWith('/api/workspace/file'))return response({file:{revision:'a'.repeat(64)}});
        if(url==='/api/file-operations') {const batch=JSON.parse(options.body).files;counts.push(batch.length);return response({success:true,operation:{status:'complete',results:batch.map(file=>({id:file.id,success:true}))}});}
        return response({success:true,files:[]});
    };
    await f.context.deleteUserFiles(files);
    assert.deepEqual(counts,[100,1]);assert.match(f.elements.fileManagerStatus.textContent,/Deletion succeeded: 101 deleted, 0 failed/);
    assert.equal(f.state.userFileSelection.size,0);
});
