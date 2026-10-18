const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const yaml = require('js-yaml');
const schema = require('../public/yaml-schema.json');
const core = require('../public/workspace-core');
const appSource = fs.readFileSync(require.resolve('../public/app.js'), 'utf8');
const source = appSource.slice(appSource.indexOf('async function loadConfig()'), appSource.indexOf('// ===== Run & Preview (Tab 2)'));
function fixture() {
    let content = 'saved';
    const select = {value: 'a.yaml'};
    const state = {configData:'saved', loadedConfigFile:'a.yaml', configLoadGeneration:0,
        configSaving:false, configLoadPending:false, userId:'fixture',
        aceEditor:{getValue:() => content, setValue:v => { content = v; }}};
    const context = vm.createContext({AppState:state, document:{getElementById:() => select},
        confirm:() => true, prompt:() => 'b.yaml', showStatus() {}, refreshConfigList:async () => {},
        Workbench:{refresh() {}, dirty:() => core.dirty(content, state.configData)},
        fetch:async () => ({ok:true, text:async () => 'loaded', json:async () => ({})})});
    vm.runInContext(source, context);
    return {state, select, context, edit:v => {content = v;}, text:() => content};
}
test('inspection is versioned and does not evaluate formulas', () => {
    const r = core.inspect('nonlinear_solver: {enabled: true, tolerance: "$tol"}',schema,yaml);
    assert.equal(r.version,1); assert.equal(r.valid,true); assert.equal(r.summary.tolerance,'$tol');
    assert.equal(core.inspect('',schema,yaml).valid,false);
    assert.equal(core.inspect('- 1',schema,yaml).valid,false);
});
test('file search/sort is nonmutating and stable for unknown dates', () => {
    const input=[{name:'model10.yaml',size:10},{name:'model2.yaml',size:20},{name:'other.png',size:100}];
    assert.deepEqual(core.files(input,'MODEL','name').map(f=>f.name),['model2.yaml','model10.yaml']);
    assert.equal(core.files(input,'','size')[0].name,'other.png');
    assert.equal(input[0].name,'model10.yaml');
});

test('material library inspection counts presets without inventing solver settings', () => {
    const content = fs.readFileSync(require.resolve('../../general_materials.yaml'), 'utf8');
    const r = core.inspect(content, schema, yaml);
    assert.equal(r.valid, true);
    assert.deepEqual(r.summary, {kind:'materials', materialCount:Object.keys(yaml.load(content).material_presets).length});
    assert.ok(r.summary.materialCount > 0);
    assert.deepEqual(core.inspect('material_presets: {}', schema, yaml).summary, {kind:'materials', materialCount:0});
    const config = core.inspect('materials: {air: {mu_r: 1}}', schema, yaml);
    assert.equal(config.summary.kind, 'config');
    assert.equal(config.summary.materialCount, 1);
    assert.equal(config.summary.coordinates, 'cartesian (default)');
    assert.equal(core.inspect('material_presets: [', schema, yaml).summary, null);
});
test('cancelled load retains text and restores selected filename', async () => {
    const f=fixture(); f.edit('draft'); f.select.value='b.yaml'; f.context.confirm=()=>false;
    f.context.fetch=()=>{throw Error('must not fetch');};
    assert.equal(await f.context.loadConfig(),false);
    assert.equal(f.text(),'draft'); assert.equal(f.select.value,'a.yaml');
});
test('late loads cannot overwrite edits made in flight', async () => {
    const f=fixture(); let resolve;
    f.context.fetch=()=>new Promise(r=>{resolve=r;});
    const pending=f.context.loadConfig(); f.edit('new edits');
    resolve({ok:true,text:async ()=>'remote'});
    assert.equal(await pending,false); assert.equal(f.text(),'new edits');
    assert.equal(f.state.configData,'saved'); assert.equal(f.state.configLoadPending,false);
});
test('newest file load wins over out-of-order responses', async () => {
    const f=fixture(), pending=[];
    f.context.fetch=()=>new Promise(r=>pending.push(r));
    const first=f.context.loadConfig(); f.select.value='b.yaml'; const second=f.context.loadConfig();
    pending[1]({ok:true,text:async ()=>'second'}); assert.equal(await second,true);
    pending[0]({ok:true,text:async ()=>'first'}); assert.equal(await first,false);
    assert.equal(f.state.loadedConfigFile,'b.yaml'); assert.equal(f.text(),'second');
});
test('save targets loaded document, not a changed dropdown', async () => {
    const f=fixture(); f.edit('draft'); f.select.value='wrong.yaml'; let body;
    f.context.fetch=async (_,options)=>{body=JSON.parse(options.body); return {ok:true,json:async ()=>({})};};
    assert.equal(await f.context.saveConfig(),true); assert.equal(body.file,'a.yaml');
    assert.equal(f.state.configData,'draft');
});
test('failed save preserves dirty baseline and returns failure', async () => {
    const f=fixture(); f.edit('draft'); f.context.fetch=async ()=>({ok:false,json:async ()=>({error:'denied'})});
    assert.equal(await f.context.saveConfig(),false); assert.equal(f.state.configData,'saved');
    assert.equal(f.state.configSaving,false);
});
test('edits made while saving stay dirty; duplicate saves are rejected', async () => {
    const f=fixture(); f.edit('draft'); let resolve;
    f.context.fetch=()=>new Promise(r=>{resolve=r;});
    const pending=f.context.saveConfig(); f.edit('newer');
    assert.equal(await f.context.saveConfig(),false);
    resolve({ok:true,json:async ()=>({})}); assert.equal(await pending,true);
    assert.equal(f.state.configData,'draft'); assert.equal(core.dirty(f.text(),f.state.configData),true);
});
test('save-as establishes the new acknowledged baseline', async () => {
    const f=fixture(); f.edit('draft'); await f.context.saveConfigAs();
    assert.equal(f.state.loadedConfigFile,'b.yaml'); assert.equal(f.state.configData,'draft');
    assert.equal(f.select.value,'b.yaml'); assert.equal(f.state.configSaving,false);
});
test('workspace HTML has unique IDs and ordered dependencies', () => {
    const html=fs.readFileSync(require.resolve('../public/index.html'),'utf8');
    const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
    assert.equal(ids.length,new Set(ids).size);
    assert.ok(html.indexOf('src="workspace-core.js"') < html.indexOf('src="workspace.js"'));
    assert.ok(html.indexOf('src="workspace.js"') < html.indexOf('src="app.js"'));
    for (const id of ['workspaceProblems','workspaceReview','fileManagerSearch','fileManagerSort']) assert.ok(ids.includes(id));
});
function runFixture(choice, saveWorks = true) {
    let saved = 0;
    const state = {configData:'coordinate_system: cartesian', loadedConfigFile:'a.yaml',
        uploadedImageFilename:'model.png', userId:'fixture', selectedLibrary:null,
        aceEditor:{getValue:()=> 'coordinate_system: polar'}, yamlSchema:schema};
    function element(tag) {
        const node={children:[], listeners:{}, textContent:'', append(...children){this.children.push(...children);},
            setAttribute(){}, focus(){}, remove(){}, addEventListener(k,fn){this.listeners[k]=fn;},
            close(value){this.returnValue=value; this.listeners.close();},
            showModal(){queueMicrotask(()=>this.close(choice));},
            get firstChild(){return this.children[0];}};
        return node;
    }
    const context=vm.createContext({WorkspaceCore:core, AppState:state, jsyaml:yaml,
        document:{createElement:element,body:{append(){}},getElementById:()=>null},
        showStatus(){},confirm:()=>true, queueMicrotask,
        saveConfig:async()=>{saved++; if(saveWorks) state.configData=state.aceEditor.getValue(); return saveWorks;}});
    vm.runInContext(fs.readFileSync(require.resolve('../public/workspace'),'utf8')+'\nthis.api=Workbench;',context);
    return {api:context.api,state,saves:()=>saved};
}
test('run preflight cancels without writing or launching', async () => {
    const f=runFixture('cancel'); assert.equal(await f.api.prepareRun(),null); assert.equal(f.saves(),0);
});
test('run saved version is explicit and never rewrites the draft', async () => {
    const f=runFixture('saved'); const request=await f.api.prepareRun();
    assert.equal(request.configFile,'a.yaml'); assert.equal(f.saves(),0);
    assert.equal(f.state.configData,'coordinate_system: cartesian');
});
test('save-and-run requires a successful save', async () => {
    const bad=runFixture('save',false); assert.equal(await bad.api.prepareRun(),null); assert.equal(bad.saves(),1);
    const good=runFixture('save'); assert.equal((await good.api.prepareRun()).imageFile,'model.png'); assert.equal(good.saves(),1);
});
test('run preflight blocks missing image and in-flight configuration operations', async () => {
    for (const patch of [{uploadedImageFilename:null},{configSaving:true},{configLoadPending:true}]) {
        const f=runFixture('save');Object.assign(f.state,patch);
        assert.equal(await f.api.prepareRun(),null);assert.equal(f.saves(),0);
    }
});
test('inspection API agrees with GUI core and rejects non-text input', async t => {
    process.env.OPENMAGFDM_NO_LISTEN='1'; process.env.OPENMAGFDM_SKIP_INIT='1';
    const {app}=require('../server');
    const server=await new Promise(resolve=>{const s=app.listen(0,'127.0.0.1',()=>resolve(s));});
    t.after(()=>new Promise(resolve=>server.close(resolve)));
    const url=`http://127.0.0.1:${server.address().port}/api/config/inspect`;
    const content='coordinate_system: polar\npolar_domain: {r_start: 1, r_end: 0.2}';
    const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content})});
    assert.deepEqual(await r.json(), core.inspect(content,schema,yaml));
    const bad=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:{}})});
    assert.equal(bad.status,400);
});
