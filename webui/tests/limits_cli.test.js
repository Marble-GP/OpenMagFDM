'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),os=require('node:os'),path=require('node:path');
const {spawnSync}=require('node:child_process');
const {Workspace,noLinks}=require('../automation/workspace');
const storage=require('../automation/storage');
const {run}=require('../automation/admin-cli');
const {privateDirectory}=require('../automation/access');
test('CLI help never starts the server or creates state',()=>{
    for(const args of [['help'],['--help'],['users','--help'],['limits','--help']]){
        const r=spawnSync(process.execPath,[path.resolve(__dirname,'../launcher.js'),...args],{encoding:'utf8',timeout:5000});
        assert.equal(r.status,0,r.stderr);assert.match(r.stdout,/limits set/);assert.match(r.stdout,/storage migrate-results/);assert.doesNotMatch(r.stdout,/Server running/);
    }
});
test('CLI persists defaults and workspace overrides; quotas serialize creation and preserve replacement',async t=>{
    const base=await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()),'omfdm-limits-'));
    const directory=privateDirectory(base);
    t.after(async()=>{await noLinks(base);await fs.rm(base,{recursive:true,force:true});await noLinks(directory,{missing:true});await fs.rm(directory,{recursive:true,force:true});});
    await run(base,['limits','set-default','configs','1']);
    await run(base,['limits','set','alice','images','0']);
    await run(base,['limits','set','bob','configs','2']);
    storage.configure(base);
    const alice=new Workspace(base,'alice'),bob=new Workspace(base,'bob');await alice.init();await bob.init();
    const writes=await Promise.allSettled(['a','b'].map(n=>alice.write(`configs/${n}.yaml`,'x',null)));
    assert.equal(writes.filter(r=>r.status==='fulfilled').length,1);
    assert.equal(writes.find(r=>r.status==='rejected').reason.code,'QUOTA_EXCEEDED');
    const file=writes.find(r=>r.status==='fulfilled').value;await alice.write(file.id,'updated',file.revision);
    await assert.rejects(alice.write('images/a.png','x',null),{code:'QUOTA_EXCEEDED'});
    for(const n of ['a','b'])await bob.write(`configs/${n}.yaml`,'x',null);
    await run(base,['limits','set','bob','configs','unlimited']);storage.configure(base);await bob.write('configs/c.yaml','x',null);
    await run(base,['limits','set','bob','configs','default']);storage.configure(base);
    await assert.rejects(bob.write('configs/d.yaml','x',null),{code:'QUOTA_EXCEEDED'});assert.equal((await bob.walk('configs')).length,3);
    await assert.rejects(run(base,['limits','set-default','configs','-1']));
    await fs.writeFile(path.join(directory,'server-lock.json'),JSON.stringify({pid:process.pid}));
    await assert.rejects(run(base,['limits','set-default','images','10']),/Stop the server/);
});
