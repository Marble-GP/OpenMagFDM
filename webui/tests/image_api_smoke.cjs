'use strict';
// Isolated HTTP acceptance for the Jimp/Multer security upgrade.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');
const {spawn} = require('node:child_process');
const {once} = require('node:events');
const {Jimp} = require('jimp');
async function main() {
    const repo = path.resolve(__dirname,'../..');
    const root = await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()),'omfdm-image-test-'));
    const base = path.join(root,'app');
    await fs.mkdir(path.join(base,'webui'),{recursive:true});
    for(const file of ['launcher.js','server.js']) await fs.copyFile(path.join(repo,'webui',file),path.join(base,'webui',file));
    for(const dir of ['automation','public']) await fs.cp(path.join(repo,'webui',dir),path.join(base,'webui',dir),{recursive:true});
    for(const file of ['sample_config.yaml','general_materials.yaml']) await fs.copyFile(path.join(repo,file),path.join(base,file));
    const probe = net.createServer().listen(0,'127.0.0.1'); await once(probe,'listening');
    const port = probe.address().port; await new Promise(r=>probe.close(r));
    const packagedIndex=process.argv.indexOf('--exe');
    let executable=process.execPath, args=[path.join(base,'webui/launcher.js')];
    const env={...process.env,PORT:String(port),LOCALAPPDATA:path.join(root,'private'),HOME:path.join(root,'private'),NODE_PATH:path.join(repo,'webui/node_modules')};
    if(packagedIndex>=0) {
        assert.ok(process.argv[packagedIndex+1],'--exe needs a file');
        executable=path.join(base,process.platform==='win32'?'webui-server.exe':'webui-server');args=[];
        await fs.copyFile(path.resolve(process.argv[packagedIndex+1]),executable);
        if(process.platform!=='win32')await fs.chmod(executable,0o700);
        await fs.cp(path.join(repo,'webui/public'),path.join(base,'public'),{recursive:true});
        delete env.NODE_PATH;
        env.PATH=(env.PATH||'').split(path.delimiter).filter(p=>path.resolve(p)!==path.dirname(process.execPath)).join(path.delimiter);
    }
    const child = spawn(executable,args,{cwd:base,windowsHide:true,stdio:['ignore','ignore','pipe'],env});
    let stderr='';child.stderr.on('data',bytes=>{stderr=(stderr+bytes).slice(-4096);});
    const url = `http://127.0.0.1:${port}`;
    const request = (endpoint,options={})=>fetch(url+endpoint,{...options,signal:AbortSignal.timeout(5000)});
    const json = (endpoint,body)=>request(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const form = (bytes,name='model.png')=>{const data=new FormData();data.set('userId','image_test');data.set('image',new Blob([bytes]),name);return data;};
    try {
        const deadline = Date.now()+15000;
        while(true){try{if((await request('/api/auth/me')).ok)break;}catch{}assert.ok(Date.now()<deadline && child.exitCode===null,`Image test server startup: ${stderr}`);await new Promise(r=>setTimeout(r,100));}
        const image = new Jimp({width:32,height:32,color:0xffffffff});
        image.setPixelColor(0xff0000ff,16,16);
        for(const [mime,ext] of [['image/png','png'],['image/jpeg','jpg'],['image/bmp','bmp']]) {
            const bytes = await image.getBuffer(mime);
            const upload = await request('/api/upload-image',{method:'POST',body:form(bytes,`model.${ext}`)});
            assert.equal(upload.status,200,await upload.text());
            const detect = await request('/api/materials/detect',{method:'POST',body:form(bytes,`model.${ext}`)});
            assert.equal(detect.status,200,await detect.text());
        }
        const common={userId:'image_test',filename:'model.png'};
        for(const [endpoint,body] of [
            ['/api/preprocess-polar/detect',common],
            ['/api/preprocess-polar/warp',{...common,center_x:16,center_y:16,r_start_px:1,r_end_px:14,theta_start:0,theta_end:360,nr:16,ntheta:32,r_orientation:'horizontal',output_filename:'warped.png'}],
            ['/api/preprocess-filter/quantize',{...common,output_filename:'quantized.png',nTargets:2,despeckleRadius:0}]
        ]) {const r=await json(endpoint,body);assert.equal(r.status,200,await r.text());}
        for(const filename of ['warped.png','quantized.png']) {
            const r=await request(`/uploads/image_test/${filename}`);assert.equal(r.status,200);
            const decoded=await Jimp.fromBuffer(Buffer.from(await r.arrayBuffer()));assert.ok(decoded.bitmap.width>0);
        }
        const enormous=await json('/api/preprocess-polar/warp',{...common,center_x:16,center_y:16,r_start_px:1,r_end_px:14,theta_start:0,theta_end:360,nr:100000,ntheta:100000});
        assert.equal(enormous.status,400,'Oversized output must be rejected before allocating an image');
        const asf = Buffer.alloc(55);Buffer.from('3026b2758e66cf11a6d900aa0062ce6c','hex').copy(asf);
        for(const endpoint of ['/api/upload-image','/api/materials/detect']) {
            const r=await request(endpoint,{method:'POST',body:form(asf)});assert.ok(r.status>=400);assert.match(await r.text(),/Only PNG, JPEG and BMP/);
        }
        const malformed=await request('/api/upload-image',{method:'POST',headers:{'Content-Type':'multipart/form-data; boundary=incomplete'},body:'--incomplete\r\nContent-Disposition: form-data; name="image"; filename="a.png"\r\n\r\ntruncated'});
        assert.ok(malformed.status>=400);
        await fs.mkdir(path.join(base,'uploads','other'),{recursive:true});
        await fs.writeFile(path.join(base,'uploads','other','secret.png'),await image.getBuffer('image/png'));
        const traversal=await json('/api/preprocess-polar/detect',{...common,filename:'../other/secret.png'});assert.ok(traversal.status>=400);
        assert.equal((await request('/api/auth/me')).status,200,'Server remains responsive after rejected inputs');
        const leftovers=(await fs.readdir(path.join(base,'uploads'),{withFileTypes:true})).filter(entry=>entry.isFile());
        assert.equal(leftovers.length,0,'Rejected uploads must not accumulate temporary files');
        console.log('PASS: PNG/JPEG/BMP upload and material detection, polar/quantization output, malformed images/multipart and foreign image path rejection.');
    } finally {
        if(child.exitCode===null&&child.signalCode===null){const closed=once(child,'close');child.kill();await closed;}
        assert.ok(path.basename(root).startsWith('omfdm-image-test-'));
        await fs.rm(root,{recursive:true,force:true});
    }
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});
