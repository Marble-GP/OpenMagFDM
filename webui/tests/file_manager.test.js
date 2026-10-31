'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync(require.resolve('../public/app.js'),'utf8');
function fixture(files) {
    const elements={fileManagerStatus:{},userFilesList:{},deleteSelectedUserFilesBtn:{},userFileSelectedCount:{},selectAllUserFiles:{}};
    for (const id of ['fileEditDialog','fileEditTitle','fileEditCategory','fileEditName','fileEditRenameField','fileEditDescriptionField','fileEditHelp','fileEditSave','fileEditCancel','fileEditClose','fileEditDiscard','fileEditNewName','fileEditDescription','fileEditStatus','fileManagerSearch']) elements[id]={value:'',dataset:{},hidden:false,setAttribute(k,v){this[k]=v;},focus(){this.focused=true;},setSelectionRange(){},showModal(){this.open=true;},close(){this.open=false;}};
    const state={userId:'alice',userFiles:files,userFileSelection:new Set()};
    const context=vm.createContext({AppState:state,document:{getElementById:id=>elements[id],querySelectorAll:()=>[],querySelector:()=>elements.fileEditCancel},console,AbortSignal,performance,
        WorkspaceCore:require('../public/workspace-core'),confirm:()=>true,fmtBytes:String,
        refreshResultsList:async()=>{},refreshConfigList:async()=>{},refreshImageList:async()=>{},clearFilePreview(){},setTimeout:fn=>{fn();}});
    vm.runInContext(source.slice(source.indexOf('function fileManagerEscape'),source.indexOf('/**\n * Refresh the list of output folders')),context);
    return {context,state,elements};
}
const response=data=>({ok:true,json:async()=>data});
test('Files offers name editing and a multiline result description with revision checks',async()=>{
    const f=fixture([{category:'results',name:'run'}]);
    assert.match(f.context.userFileHtml({category:'configs',name:'x.yaml'}),/fileManagerEdit/);
    assert.match(f.context.userFileHtml({category:'results',name:'run'}),/fileManagerEdit/);
    const field=f.elements.fileEditDescription,save=f.elements.fileEditSave;
    const calls=[];f.context.fetch=async(url,options)=>{calls.push({url,options});if(url.startsWith('/api/workspace/file'))return response({file:{revision:'file-revision'}});return response(options?{success:true,revision:'new-revision'}:{success:true,description:'first\nsecond',revision:'old-revision'});};
    const button={closest:()=>({dataset:{category:'results',name:'run'}})};
    await f.context.fileManagerEdit(button);assert.equal(field.value,'first\nsecond');assert.equal(save.disabled,true);
    field.value='changed\ntext';f.context.fileEditorChanged();await f.context.saveFileEditor();
    assert.deepEqual(JSON.parse(calls.find(call=>call.options).options.body),{userId:'alice',description:'changed\ntext',expectedRevision:'old-revision'});
    assert.equal(f.elements.fileEditDialog.open,false);assert.match(f.elements.fileManagerStatus.textContent,/Changes saved/);
});
test('renaming the loaded configuration keeps its draft and revision',async()=>{
    const f=fixture([{category:'configs',name:'old.yaml'}]);
    Object.assign(f.state,{loadedConfigFile:'old.yaml',configRevision:'existing-revision',configData:'saved',aceEditor:{getValue:()=> 'unsaved draft'}});
    f.elements.configFileSelect={value:'old.yaml'};
    f.context.syncConfigPicker=()=>{};f.context.Workbench={refresh(){}};
    f.context.refreshUserFiles=async()=>{};
    f.context.fetch=async(url,options)=>url.startsWith('/api/workspace/file')?response({file:{revision:'other-revision'}}):response({success:true,file:{name:'new.yaml',revision:'other-revision'}});
    await f.context.fileManagerEdit({closest:()=>({dataset:{category:'configs',name:'old.yaml'}})});
    f.elements.fileEditNewName.value='new.yaml';await f.context.saveFileEditor();
    assert.equal(f.state.loadedConfigFile,'new.yaml');assert.equal(f.state.configRevision,'existing-revision');
    assert.equal(f.state.aceEditor.getValue(),'unsaved draft');assert.equal(f.elements.configFileSelect.value,'new.yaml');
});
test('renaming a loaded result keeps its viewer and description pointed at the new folder',async()=>{
    const f=fixture([{category:'results',name:'old'}]);
    f.state.resultsData={currentResult:'outputs/alice/old'};f.elements.resultSelect={value:'outputs/alice/old'};
    const calls=[];f.context.refreshUserFiles=async()=>{};
    f.context.ResultImprovements={rename:(a,b)=>calls.push([a,b])};f.context.updateCustomSelectDisplay=value=>calls.push(value);
    f.context.fetch=async(url)=>url.startsWith('/api/workspace/file')?response({file:{revision:'a'.repeat(64)}}):response({success:true,file:{name:'new'}});
    await f.context.fileManagerEdit({closest:()=>({dataset:{category:'results',name:'old'}})});
    f.elements.fileEditNewName.value='new';await f.context.saveFileEditor();
    assert.equal(f.state.resultsData.currentResult,'outputs/alice/new');assert.equal(f.elements.resultSelect.value,'outputs/alice/new');
    assert.deepEqual(calls,[['old','new'],'outputs/alice/new']);
});

test('renaming an active library persists its selection and preserves the editor draft',async()=>{
    const f=fixture([{category:'libraries',name:'old.yaml'}]),cookies={magfdm_last_library:'old.yaml'};
    Object.assign(f.state,{selectedLibrary:'old.yaml',currentLibraryFile:'old.yaml',detectLibraryName:'old.yaml',libraryRevision:'existing-revision',libraryAceEditor:{getValue:()=> 'unsaved draft'}});
    f.elements.activeLibraryName={textContent:'old.yaml'};f.elements.libEditFilename={textContent:'old.yaml'};
    f.context.refreshUserFiles=async()=>{};
    f.context.setCookie=(name,value)=>{cookies[name]=value;};
    f.context.fetch=async url=>url.startsWith('/api/workspace/file')?response({file:{revision:'a'.repeat(64)}}):response({success:true});
    await f.context.fileManagerEdit({closest:()=>({dataset:{category:'libraries',name:'old.yaml'}})});
    f.elements.fileEditNewName.value='new.yaml';await f.context.saveFileEditor();
    assert.equal(f.state.selectedLibrary,'new.yaml');assert.equal(f.state.currentLibraryFile,'new.yaml');
    assert.equal(cookies.magfdm_last_library,'new.yaml');assert.equal(f.elements.activeLibraryName.textContent,'new.yaml');
    assert.equal(f.elements.libEditFilename.textContent,'new.yaml');assert.equal(f.state.detectLibraryName,'new.yaml');
    assert.equal(f.state.libraryRevision,'existing-revision');assert.equal(f.state.libraryAceEditor.getValue(),'unsaved draft');
});
test('successful ZIP import discovers results immediately and restores operation controls',async()=>{
    let refreshes=0;
    const state={userId:'alice'};
    const context=vm.createContext({AppState:state,console,confirm:()=>true,fileOperationProgress(){},updateUserFileSelectionControls(){},fileManagerStatus(){},
        uploadBackupWithProgress:async()=>({success:true,transferId:'one',files:[{id:'results/run/data.txt',expectedRevision:null}]}),
        fetch:async()=>response({success:true,written:{results:1}}),
        refreshConfigList:async()=>{},refreshImageList:async()=>{},refreshUserFiles:async()=>{},refreshResultsList:async()=>{refreshes++;}});
    vm.runInContext(source.slice(source.indexOf('async function importUserData('),source.indexOf('async function insertPolarYaml(')),context);
    await context.importUserData({files:[{name:'backup.zip',size:100}],value:'backup.zip'});
    assert.equal(refreshes,1);
    assert.equal(state.fileImportBusy,false);
});
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
test('deletion exposes a progress bar during work and clears it after completion',async()=>{
    const files=[{category:'configs',name:'old.yaml'}],f=fixture(files);
    f.elements.fileOperationProgress={hidden:true};f.elements.fileOperationLabel={};
    f.elements.fileOperationBar={removeAttribute(name){delete this[name];}};
    let visibleDuringDelete=false;
    f.context.fetch=async url=>{
        if(url.startsWith('/api/workspace/file'))return response({file:{revision:'a'.repeat(64)}});
        if(url==='/api/file-operations') {
            visibleDuringDelete=!f.elements.fileOperationProgress.hidden&&/Deleting/.test(f.elements.fileOperationLabel.textContent);
            return response({success:true,operation:{status:'complete',results:[{id:'configs/old.yaml',success:true}]}});
        }
        return response({success:true,files:[]});
    };
    await f.context.deleteUserFiles(files);
    assert.equal(visibleDuringDelete,true);
    assert.equal(f.elements.fileOperationProgress.hidden,true);
});
test('an active import prevents deletion from replacing its progress',async()=>{
    const files=[{category:'configs',name:'old.yaml'}],f=fixture(files);
    f.state.fileImportBusy=true;
    let requested=false;f.context.fetch=async()=>{requested=true;throw Error('unexpected request');};
    await f.context.deleteUserFiles(files);
    assert.equal(requested,false);
    assert.equal(f.state.fileDeleteBusy,undefined);
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

test('file editor preserves drafts on cancel and offers explicit discard',async()=>{
    const f=fixture([]);f.context.fetch=async()=>response({file:{revision:'r1'}});
    await f.context.fileManagerEdit({closest:()=>({dataset:{category:'configs',name:'old.yaml'}})});
    f.elements.fileEditNewName.value='draft.yaml';f.context.fileEditorChanged();
    f.context.closeFileEditor();assert.equal(f.elements.fileEditDialog.open,true);assert.equal(f.elements.fileEditDiscard.hidden,false);
    f.context.keepFileEditorChanges();assert.equal(f.elements.fileEditNewName.value,'draft.yaml');assert.equal(f.elements.fileEditDiscard.hidden,true);
    f.context.closeFileEditor(true);assert.equal(f.elements.fileEditDialog.open,false);
});

test('rename validates names locally and retains input on server conflict',async()=>{
    const f=fixture([]);let writes=0;
    f.context.fetch=async(url,options)=>{if(options){writes++;return {ok:false,json:async()=>({error:'The destination already exists'})};}return response({file:{revision:'r1'}});};
    await f.context.fileManagerEdit({closest:()=>({dataset:{category:'configs',name:'old.yaml'}})});
    f.elements.fileEditNewName.value='../bad.yaml';await f.context.saveFileEditor();assert.equal(writes,0);
    assert.equal(f.elements.fileEditNewName['aria-invalid'],'true');
    f.elements.fileEditNewName.value='existing.yaml';f.context.fileEditorChanged();await f.context.saveFileEditor();
    assert.equal(writes,1);assert.equal(f.elements.fileEditNewName.value,'existing.yaml');assert.equal(f.elements.fileEditDialog.open,true);
    assert.match(f.elements.fileEditStatus.textContent,/destination already exists/);assert.equal(f.elements.fileEditSave.disabled,false);
});

test('save blocks duplicate writes and closing until the request finishes',async()=>{
    const f=fixture([]);let resolveSave,writes=0;
    f.context.fetch=async(url,options)=>{if(options){writes++;return new Promise(resolve=>{resolveSave=resolve;});}return url.startsWith('/api/workspace/file')?response({file:{revision:'f1'}}):response({success:true,description:'old',revision:'r1'});};
    await f.context.fileManagerEdit({closest:()=>({dataset:{category:'results',name:'run'}})});
    f.elements.fileEditDescription.value='new';const pending=f.context.saveFileEditor();
    await f.context.saveFileEditor();f.context.closeFileEditor(true);
    assert.equal(writes,1);assert.equal(f.elements.fileEditDialog.open,true);assert.equal(f.elements.fileEditClose.disabled,true);
    resolveSave(response({success:true,revision:'r2'}));await pending;
    assert.equal(f.elements.fileEditDialog.open,false);
});

test('a closed editor ignores its late load and cannot replace the next draft',async()=>{
    const f=fixture([]);let finishFirst;
    f.context.fetch=async url=>url.startsWith('/api/workspace/file')?response({file:{revision:'f1'}}):url.includes('first')?new Promise(resolve=>{finishFirst=resolve;}):response({success:true,description:'second notes',revision:'r2'});
    const first=f.context.fileManagerEdit({closest:()=>({dataset:{category:'results',name:'first'}})});
    f.context.closeFileEditor();await f.context.fileManagerEdit({closest:()=>({dataset:{category:'results',name:'second'}})});
    finishFirst(response({success:true,description:'stale notes',revision:'r1'}));await first;
    assert.equal(f.elements.fileEditDescription.value,'second notes');assert.equal(f.elements.fileEditName.textContent,'second');
});

test('workspace changes block saving a stale file dialog',async()=>{
    const f=fixture([]);let writes=0;
    f.context.fetch=async(url,options)=>{if(options)writes++;return response({file:{revision:'r1'}});};
    await f.context.fileManagerEdit({closest:()=>({dataset:{category:'configs',name:'old.yaml'}})});
    f.elements.fileEditNewName.value='new.yaml';f.state.userId='bob';await f.context.saveFileEditor();
    assert.equal(writes,0);assert.match(f.elements.fileEditStatus.textContent,/Workspace changed/);
});

test('combined edit retries only the unsaved description after a successful rename',async()=>{
    const f=fixture([]),writes=[];let failDescription=true;
    f.context.refreshUserFiles=async()=>{};
    f.context.fetch=async(url,options)=>{
        if(!options)return url.startsWith('/api/workspace/file')?response({file:{revision:'file-r1'}}):response({success:true,description:'old notes',revision:'notes-r1'});
        writes.push({url,body:JSON.parse(options.body)});
        if(url==='/api/user-files/rename')return response({success:true,file:{name:'new',revision:'file-r2'}});
        if(failDescription)return {ok:false,json:async()=>({error:'Description write failed'})};
        return response({success:true,revision:'notes-r2'});
    };
    await f.context.fileManagerEdit({closest:()=>({dataset:{category:'results',name:'old'}})});
    f.elements.fileEditNewName.value='new';f.elements.fileEditDescription.value='new notes';await f.context.saveFileEditor();
    assert.equal(f.elements.fileEditDialog.open,true);assert.equal(f.elements.fileEditName.textContent,'new');
    assert.match(f.elements.fileEditStatus.textContent,/Name saved. Description was not saved/);
    failDescription=false;await f.context.saveFileEditor();
    assert.deepEqual(writes.map(w=>w.url),['/api/user-files/rename','/api/user-outputs/new/description','/api/user-outputs/new/description']);
    assert.equal(writes[0].body.revision,'file-r1');assert.equal(writes[2].body.expectedRevision,'notes-r1');
    assert.equal(f.elements.fileEditDialog.open,false);
});
