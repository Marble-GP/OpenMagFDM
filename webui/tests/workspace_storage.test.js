'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os');
const storage=require('../automation/storage');
const {run}=require('../automation/admin-cli');
const {Workspace,noLinks}=require('../automation/workspace');
const {privateDirectory}=require('../automation/access');
async function fixture(t) {
    const temp=await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()),'omfdm-storage-'));
    const base=path.join(temp,'app');await fs.mkdir(base);const privateDir=privateDirectory(base);
    t.after(async()=>{await noLinks(temp);assert.ok(path.basename(temp).startsWith('omfdm-storage-'));await fs.rm(temp,{recursive:true,force:true});await noLinks(privateDir,{missing:true});await fs.rm(privateDir,{recursive:true,force:true});});
    return {temp,base,privateDir};
}
test('workspace root covers every category and preserves a separate results override',async t=>{
    const {temp,base}=await fixture(t),data=path.join(temp,'data'),outputs=path.join(temp,'separate-results');
    await run(base,['storage','set-workspace-root',data]);
    const ws=new Workspace(base,'alice');await ws.init();
    for(const [category,folder] of Object.entries({configs:'configs',images:'uploads',libraries:'user-libs',results:'outputs'})) {
        const id=`${category}/test.txt`;await ws.write(id,'data',null);
        assert.equal(await ws.resolve(id),path.join(data,folder,'alice','test.txt'));
    }
    await run(base,['storage','set-results-root',outputs]);
    await run(base,['storage','set-workspace-root',path.join(temp,'new-data')]);
    assert.equal(await ws.resolve('results/test.txt',{create:true}),path.join(outputs,'alice','test.txt'));
});
test('whole-workspace migration combines separate results, verifies copies and retains source and private identity',async t=>{
    const {temp,base,privateDir}=await fixture(t);
    await run(base,['storage','set-results-root',path.join(temp,'external-results')]);
    const ws=new Workspace(base,'alice');await ws.init();
    const ids=['configs/model.yaml','images/model.png','libraries/copper.yaml','results/run/.inputs/model.png','results/run/conditions.json'];
    for(const id of ids)await ws.write(id,Buffer.from(id),null);
    const sources=await Promise.all(ids.map(id=>ws.resolve(id)));
    await fs.writeFile(path.join(privateDir,'accounts.json'),'{"version":1,"users":[],"owners":{"alice":"owner"}}');
    const target=path.join(temp,'all-workspaces');await run(base,['storage','migrate-workspaces',target]);
    assert.equal(storage.readSettings(base).workspaceRoot,target);assert.equal(storage.readSettings(base).resultsRoot,undefined);
    for(const [i,id] of ids.entries()) {
        assert.equal(await fs.readFile(await ws.resolve(id),'utf8'),id);
        assert.equal(await fs.readFile(sources[i],'utf8'),id,'Source must remain intact');
    }
    assert.equal(JSON.parse(await fs.readFile(path.join(privateDir,'accounts.json'),'utf8')).owners.alice,'owner');
    assert.equal((await ws.list({category:'results'})).items[0].name,'run');
});
test('migration refuses overlap, nonempty targets and a running server without changing settings',async t=>{
    const {temp,base,privateDir}=await fixture(t);const ws=new Workspace(base,'alice');await ws.init();
    await ws.write('configs/model.yaml','original',null);
    await assert.rejects(run(base,['storage','migrate-workspaces',path.join(base,'configs','nested')]),/overlap/);
    const target=path.join(temp,'nonempty');await fs.mkdir(target);await fs.writeFile(path.join(target,'keep'),'retained');
    await assert.rejects(run(base,['storage','migrate-workspaces',target]),/empty/);
    await assert.rejects(run(base,['storage','set-workspace-root',path.dirname(privateDir)]),/private server state/);
    await assert.rejects(run(base,['storage','set-results-root',path.join(base,'webui','public','results')]),/public web assets/);
    await fs.writeFile(path.join(privateDir,'server-lock.json'),JSON.stringify({pid:process.pid}));
    await assert.rejects(run(base,['storage','migrate-workspaces',path.join(temp,'unused')]),/Stop the server/);
    assert.equal(storage.readSettings(base).workspaceRoot,undefined);
    assert.equal(await fs.readFile(await ws.resolve('configs/model.yaml'),'utf8'),'original');
});
test('unsafe source links leave the old storage settings intact',async t=>{
    const {temp,base}=await fixture(t);const ws=new Workspace(base,'alice');await ws.init();
    await ws.write('configs/model.yaml','original',null);
    await fs.link(await ws.resolve('configs/model.yaml'),path.join(base,'configs','alice','hard-link.yaml'));
    await assert.rejects(run(base,['storage','migrate-workspaces',path.join(temp,'target')]),{code:'UNSAFE_PATH'});
    assert.equal(storage.readSettings(base).workspaceRoot,undefined);
});
