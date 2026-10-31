'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
function fixture() {
    const elements={};for(const id of ['resultMemoEditor','resultMemoTitle','resultOriginalInput','resultMemoText','resultMemoSave','resultMemoReload','resultMemoStatus','resultMemoBadge','resultMemoDiscard'])elements[id]={value:'',dataset:{},removeAttribute(k){delete this[k];},hasAttribute(k){return k in this;},focus(){}};
    const state={userId:'alice'},writes=[];
    const context=vm.createContext({AppState:state,document:{getElementById:id=>elements[id],addEventListener(){}},refreshResultsList:async()=>{}});
    context.fetch=async(url,options)=>{
        if(options?.method==='HEAD')return {ok:true};
        if(options?.method==='PUT'){writes.push(JSON.parse(options.body));return {ok:true,json:async()=>({success:true,revision:'r2'})};}
        return {ok:true,json:async()=>({success:true,description:'saved notes',revision:'r1'})};
    };
    const source=fs.readFileSync(require.resolve('../public/result-improvements.js'),'utf8');
    vm.runInContext(source.slice(0,source.indexOf('async function optionalMaterialCanvas'))+'\nthis.api=ResultImprovements;',context);
    return {context,elements,state,writes,api:context.api};
}
test('result notes track unsaved changes and save with the loaded revision',async()=>{
    const f=fixture();await f.api.description('run');assert.equal(f.elements.resultMemoSave.disabled,true);
    f.elements.resultMemoText.value='new notes';f.api.changed();assert.equal(f.elements.resultMemoBadge.textContent,'Unsaved changes');
    await f.api.save();assert.deepEqual(f.writes,[{userId:'alice',description:'new notes',expectedRevision:'r1'}]);
    assert.equal(f.elements.resultMemoBadge.textContent,'Saved');assert.equal(f.elements.resultMemoSave.disabled,true);
});
test('reload requires explicit discard of a result description draft',async()=>{
    const f=fixture();await f.api.description('run');f.elements.resultMemoText.value='draft';f.api.changed();
    await f.api.requestReload();assert.equal(f.elements.resultMemoDiscard.hidden,false);assert.equal(f.elements.resultMemoText.value,'draft');
    f.api.keepEditing();assert.equal(f.elements.resultMemoDiscard.hidden,true);
    await f.api.requestReload(true);assert.equal(f.elements.resultMemoText.value,'saved notes');
});
test('Files edits sync into clean Results but never overwrite an unsaved draft',async()=>{
    const f=fixture();await f.api.description('run');f.api.syncDescription('run','edited in Files','r2');
    assert.equal(f.elements.resultMemoText.value,'edited in Files');
    f.elements.resultMemoText.value='local draft';f.api.changed();f.api.syncDescription('run','other notes','r3');
    assert.equal(f.elements.resultMemoText.value,'local draft');assert.match(f.elements.resultMemoStatus.textContent,/draft is still here/);
    await f.api.save();assert.equal(f.writes[0].expectedRevision,'r2');
});
test('failed result notes saves retain input and allow retry',async()=>{
    const f=fixture();await f.api.description('run');f.elements.resultMemoText.value='draft';f.api.changed();
    f.context.fetch=async()=>({ok:false,json:async()=>({error:'Revision conflict'})});await f.api.save();
    assert.equal(f.elements.resultMemoText.value,'draft');assert.equal(f.elements.resultMemoText.disabled,false);
    assert.equal(f.elements.resultMemoSave.disabled,false);assert.match(f.elements.resultMemoStatus.textContent,/Revision conflict/);
});

test('the actual result-finalization path preserves an unsaved description',async()=>{
    const f=fixture();await f.api.description('run');
    f.elements.resultMemoText.value='keep this draft';f.api.changed();
    f.state.resultsData={currentResult:'outputs/alice/run'};
    Object.assign(f.context,{assertResultLoadCurrent(){},loadQuickPreviewFromResult:async()=>{},loadResultLog:async()=>{},updateAllPlots:async()=>{}});
    const app=fs.readFileSync(require.resolve('../public/app.js'),'utf8');
    vm.runInContext(app.slice(app.indexOf('async function finalizeCommittedResult('),app.indexOf('async function restoreCommittedResultAfterValidationFailure(')),f.context);
    await f.context.finalizeCommittedResult({resultPath:'outputs/alice/run'});
    assert.equal(f.elements.resultMemoText.value,'keep this draft');assert.equal(f.elements.resultMemoBadge.textContent,'Unsaved changes');
});

test('switching results restores per-result drafts with their original revision',async()=>{
    const f=fixture();await f.api.description('first');f.elements.resultMemoText.value='first draft';f.api.changed();
    await f.api.description('second');assert.equal(f.elements.resultMemoTitle.textContent,'second');
    f.elements.resultMemoText.value='second draft';f.api.changed();await f.api.description('first');
    assert.equal(f.elements.resultMemoText.value,'first draft');await f.api.save();assert.equal(f.writes[0].expectedRevision,'r1');
    await f.api.description('second');assert.equal(f.elements.resultMemoText.value,'second draft');
});

test('saving while switching results settles the correct draft and editor',async()=>{
    const f=fixture();await f.api.description('first');f.elements.resultMemoText.value='first draft';f.api.changed();
    const normalFetch=f.context.fetch;let finish;
    f.context.fetch=async(url,options)=>options?.method==='PUT'?new Promise(resolve=>{finish=resolve;}):normalFetch(url,options);
    const saving=f.api.save();await f.api.description('second');await f.api.description('first');
    assert.equal(f.elements.resultMemoText.disabled,true);
    finish({ok:true,json:async()=>({success:true,revision:'r2'})});await saving;
    assert.equal(f.elements.resultMemoText.disabled,false);assert.equal(f.elements.resultMemoBadge.textContent,'Saved');
    assert.equal(f.elements.resultMemoText.value,'first draft');
});

test('the actual Files preview leaves the loaded result description and draft untouched',async()=>{
    const f=fixture();await f.api.description('loaded');
    f.elements.resultMemoText.value='loaded result draft';f.api.changed();
    f.elements.filePreviewPanel={};f.elements.previewDescription={};f.state.filePreviewGeneration=0;
    let previewed;
    Object.assign(f.context,{AbortController,console:{log(){},warn(){},error(){}},clearFilePreview(){},
        renderFileManagerPreview:async result=>{previewed=result;}});
    const app=fs.readFileSync(require.resolve('../public/app.js'),'utf8');
    vm.runInContext(app.slice(app.indexOf('async function showOutputPreview('),app.indexOf('async function renderFileManagerPreview(')),f.context);
    await f.context.showOutputPreview('other');
    assert.equal(previewed,'outputs/alice/other');assert.equal(f.elements.resultMemoTitle.textContent,'loaded');
    assert.equal(f.elements.resultMemoText.value,'loaded result draft');assert.equal(f.elements.resultMemoBadge.textContent,'Unsaved changes');
});
