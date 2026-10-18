const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const {locate,validate}=require('../automation/solver-location');
test('solver auto-detection and explicit overrides do not silently fall back',async()=>{
    const base=await fs.mkdtemp(path.join(os.tmpdir(),'omfdm-location-'));
    try {
        const name=process.platform==='win32'?'MagFDMsolver.exe':'MagFDMsolver';
        const built=path.join(base,'build','Release',name),adjacent=path.join(base,name);
        await fs.mkdir(path.dirname(built),{recursive:true});await fs.writeFile(built,'fixture',{mode:0o700});
        assert.equal(locate(base,'','').path,built);
        await fs.writeFile(adjacent,'fixture',{mode:0o700});assert.equal(locate(base,'','').path,adjacent);
        assert.equal(locate(base,built,'').source,'saved');
        assert.equal(locate(base,built,adjacent).source,'environment');
        assert.equal(locate(base,built,path.join(base,'missing')).available,false);
        assert.equal(validate(built),built);
        assert.throws(()=>validate(base),{code:'INVALID_SOLVER'});
        assert.throws(()=>validate('relative.exe'),{code:'INVALID_SOLVER'});
    }finally{assert.ok(path.basename(base).startsWith('omfdm-location-'));await fs.rm(base,{recursive:true,force:true});}
});
