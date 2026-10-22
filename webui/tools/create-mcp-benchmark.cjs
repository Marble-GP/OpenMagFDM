'use strict';
// Small deterministic integration fixture, not a solver performance benchmark.
const fs = require('node:fs');
const path = require('node:path');
const {pipeline} = require('node:stream/promises');
const crypto = require('node:crypto');
const yaml = require('js-yaml');
const { Jimp } = require('jimp');
const yazl = require('yazl');

async function fixture() {
    const bc = {type:'dirichlet',value:0};
    const model = {
        coordinate_system:'cartesian', mesh:{dx:0.001,dy:0.001},
        boundary_conditions:{left:{...bc},right:{...bc},top:{...bc},bottom:{...bc}},
        materials:{air:{rgb:[255,255,255],mu_r:1,jz:0},
            coil:{rgb:[255,0,0],mu_r:1,jz:1e6},body:{rgb:[128,128,128],mu_r:2,jz:0}},
        nonlinear_solver:{enabled:false},transient:{enabled:false},export:{format:'csv'}
    };
    const warning = structuredClone(model);
    warning.materials.body = {rgb:[128,128,128],'B-H':'mu0 * (1000 / (1 + ($H/1000)^2)) * $H',jz:0};
    warning.nonlinear_solver = {enabled:true,solver_type:'newton-krylov',max_iterations:2,tolerance:0,anderson:{enabled:false}};
    const bitmap = new Jimp({width:32,height:32,color:0xffffffff});
    for(let y=12;y<20;y++)for(let x=12;x<20;x++)bitmap.setPixelColor(0xff0000ff,x,y);
    for(let y=10;y<22;y++)for(let x=8;x<11;x++)bitmap.setPixelColor(0x808080ff,x,y);
    return {model,warning,png:await bitmap.getBuffer('image/png')};
}

async function main() {
    const {model,warning,png} = await fixture();
    const entries = {
        'configs/mcp_acceptance_linear.yaml':Buffer.from(yaml.dump(model,{noRefs:true})),
        'configs/mcp_acceptance_warning.yaml':Buffer.from('# Diagnostic only: deliberately fails nonlinear convergence. Do not use as a production setting.\n'+yaml.dump(warning,{noRefs:true})),
        'images/mcp_acceptance_32.png':png,
        'libraries/mcp_acceptance_materials.yaml':Buffer.from(yaml.dump({material_presets:{air:{mu_r:1},test_body:{mu_r:2}}}))
    };
    const manifest = {purpose:'OpenMagFDM MCP acceptance test, not a performance or physical-accuracy benchmark',
        expected:{linear:'completed and converged',warning:'completed with warning, not converged',materials:2},
        files:Object.entries(entries).map(([id,b])=>({id,bytes:b.length,sha256:crypto.createHash('sha256').update(b).digest('hex')}))};
    const destination = path.resolve(process.argv[2] || '../artifacts/openmagfdm-mcp-acceptance.zip');
    await fs.promises.mkdir(path.dirname(destination),{recursive:true});
    const zip = new yazl.ZipFile();
    const writing = pipeline(zip.outputStream,fs.createWriteStream(destination,{flags:'wx'}));
    for(const [id,data] of Object.entries(entries))zip.addBuffer(data,id);
    zip.addBuffer(Buffer.from(JSON.stringify(manifest,null,2)),'manifest.json');
    zip.end();await writing;
    console.log(destination);
    console.log(JSON.stringify(manifest,null,2));
}
if(require.main===module)main().catch(e=>{console.error(e.message);process.exitCode=1;});
module.exports={fixture};
