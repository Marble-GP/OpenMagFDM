'use strict';
const {McpServer} = require('@modelcontextprotocol/server');
const z = require('zod');
const fs = require('node:fs');
const yaml = require('js-yaml');
const {Workspace, WorkspaceError, fail} = require('./workspace');
const core = require('../public/workspace-core');
const schema = require('../public/yaml-schema.json');
const id=z.string().min(1).max(1024), revision=z.string().regex(/^[a-f0-9]{64}$/), expected=revision.nullable();
const file=z.object({id,revision}).strict();
const target=z.object({id,revision:expected}).strict();
const number=z.number().finite();
const page={offset:z.number().int().min(0).default(0),limit:z.number().int().min(1).max(200).default(50)};
const definitions = [
    ['workspace_status','read','Show the assigned workspace, permissions, and running analyses.',{}],
    ['files_list','read','List one directory level ONLY in this workspace. Use prefix to enter a returned directory, and files_stat for edit revisions. Paging can change while files are created.',{category:z.enum(['configs','libraries','images','results']).optional(),prefix:id.optional(),...page}],
    ['files_stat','read','Get size and revision. For deletion, inspect this exact target first; directories include descendants.',{id}],
    ['files_read','read','Read a bounded text or binary segment of a workspace file. Never accepts OS paths.',{id,offset:page.offset,length:z.number().int().min(1).max(524288).default(262144),text:z.boolean().default(false)}],
    ['yaml_inspect','read','Editor-level YAML checks; does not certify physics or resolve external references.',{content:z.string().max(1024*1024)}],
    ['yaml_template','read','Generate a conservative editable starter configuration or material library, without saving it.',{kind:z.enum(['cartesian','polar','materials'])}],
    ['yaml_write','edit','Create/update configuration or materials. expectedRevision=null means create only. Replacing also requires replace permission.',{id,content:z.string().max(100*1024),expectedRevision:expected}],
    ['files_rename','edit','Rename an unchanged file or result directory within its category. Never overwrites another target.',{id,targetId:id,revision}],
    ['files_delete','delete','PERMANENTLY delete this exact unchanged target. Obtain its revision/size with files_stat first. No recycle bin; active output cannot be deleted.',{id,revision}],
    ['image_info','read','Inspect PNG/JPEG/BMP dimensions and up to 256 dominant RGB colors.',{id}],
    ['image_process','images','Detect polar geometry or produce polar/quantized PNG using the same algorithms as the GUI. Omit outputId for an expiring preview.',{
        id,revision,operation:z.enum(['detect_polar','warp','quantize']),shape:z.enum(['auto','circular','rectangular']).optional(),outputId:id.optional(),expectedRevision:expected.optional(),
        warp:z.object({center_x:number,center_y:number,r_start_px:number.min(0),r_end_px:number.positive(),theta_start:number,theta_end:number,nr:z.number().int().min(2).max(16384),ntheta:z.number().int().min(2).max(16384),r_orientation:z.enum(['horizontal','vertical']).default('horizontal'),r_outer_physical:number.positive().default(1)}).strict().optional(),
        filter:z.object({rareThreshold:number.min(0).max(1).optional(),nTargets:z.number().int().min(1).max(64).optional(),minTargetDist:number.min(0).max(442).optional(),despeckleRadius:z.number().int().min(0).max(10).optional(),bilateralSigmaSpatial:number.min(0).max(8).optional(),bilateralSigmaColor:number.min(1).max(150).optional(),minIslandSize:z.number().int().min(0).max(1024).optional()}).strict().optional()
    }],
    ['transfer_begin','read','Begin a bounded image or ZIP upload (requires images or edit permission). Transfers expire after one hour.',{kind:z.enum(['image','backup']),size:z.number().int().positive().max(1024*1024*1024)}],
    ['transfer_append','read','Append at most 256 KiB of base64 bytes at the acknowledged offset.',{transferId:z.uuid(),offset:page.offset,base64:z.string().max(349528)}],
    ['transfer_read','read','Download the next bounded archive chunk.',{transferId:z.uuid(),offset:page.offset}],
    ['transfer_release','read','Discard this connection\'s temporary upload/download and free its quota; does not remove published files.',{transferId:z.uuid()}],
    ['image_import','images','Validate and publish a completed image upload to the assigned images folder.',{transferId:z.uuid(),id,expectedRevision:expected}],
    ['backup_export','read','Export selected versioned files/directories to a streamed ZIP (maximum 1 GiB).',{files:z.array(file).min(1).max(200)}],
    ['backup_preview','read','Inspect a completed ZIP upload and get exact target revisions before import.',{transferId:z.uuid()}],
    ['backup_import','edit','Import the inspected ZIP into this workspace only. Replacement requires replace permission. An error reports already-applied entries.',{transferId:z.uuid(),files:z.array(target).min(1).max(10000)}],
    ['analysis_start','run','Start an async analysis from fixed versioned inputs. File references inside YAML must be workspace IDs. Retrying the same key does not launch twice.',{config:file,image:file,library:file.optional(),idempotencyKey:z.string().min(1).max(128)}],
    ['analysis_list','read','List this workspace\'s automation analyses, including interrupted records.',{}],
    ['analysis_get','read','Get owned job progress, convergence/warnings and bounded log tail.',{jobId:z.uuid()}],
    ['analysis_cancel','cancel','Request cancellation of a job in this workspace.',{jobId:z.uuid()}],
    ['analysis_resume','run','Resume held queued jobs in this workspace after a server restart. Interrupted computations are not resumed.',{}],
    ['result_summary','read','Get conditions, recorded convergence/warnings and available Az steps. Unknown convergence is never reported as success.',{id,...page}],
    ['results_table','read','Read a bounded row/column window of a CSV field or waveform; no full-matrix buffering.',{id,row:page.offset,rows:z.number().int().min(1).max(200).default(50),column:page.offset,columns:z.number().int().min(1).max(200).default(20)}],
    ['result_description','edit','Set a result description without changing calculation data.',{id,content:z.string().max(8192),expectedRevision:expected}]
];
const meta=z.object({id,name:z.string(),kind:z.enum(['file','directory']),size:z.number(),modified:z.string(),revision}).passthrough();
const job=z.object({jobId:z.string(),resultId:id,status:z.string(),progress:z.number(),completed:z.boolean(),converged:z.boolean(),warning:z.boolean()}).passthrough();
const transfer=z.object({transferId:z.uuid(),size:z.number(),chunkBytes:z.number(),expires:z.string()}).passthrough();
const outputs={
    workspace_status:z.object({workspace:z.string(),permissions:z.array(z.string()),jobs:z.array(job)}),
    files_stat:meta,files_rename:meta,yaml_write:meta,image_import:meta,result_description:meta,
    files_read:meta.extend({offset:z.number(),nextOffset:z.number().nullable(),encoding:z.enum(['utf8','base64']),content:z.string()}),
    files_list:z.object({items:z.array(z.object({id,kind:z.enum(['file','directory']),name:z.string(),size:z.number().nullable(),modified:z.string()})),total:z.number(),nextOffset:z.number().nullable()}),
    files_delete:z.object({deleted:id,permanent:z.literal(true),bytes:z.number()}),
    yaml_template:z.object({content:z.string()}),
    yaml_inspect:z.object({version:z.number(),valid:z.boolean(),diagnostics:z.array(z.object({row:z.number(),column:z.number(),type:z.string(),text:z.string()}).passthrough())}).passthrough(),
    image_info:z.object({width:z.number(),height:z.number(),uniqueColors:z.number(),colors:z.array(z.object({rgb:z.array(z.number()),count:z.number()})),truncated:z.boolean()}),
    image_process:z.object({id:id.optional(),revision:revision.optional()}).passthrough(),
    transfer_begin:transfer,backup_export:transfer,
    transfer_append:z.object({transferId:z.uuid(),received:z.number(),ready:z.boolean()}),
    transfer_read:z.object({base64:z.string(),size:z.number(),nextOffset:z.number().nullable()}),
    transfer_release:z.object({released:z.uuid()}),
    backup_preview:z.object({files:z.array(z.object({id,size:z.number(),expectedRevision:expected})),permanentReplacement:z.boolean()}),
    backup_import:z.object({applied:z.array(id)}),
    analysis_start:job,analysis_get:job.extend({logTail:z.array(z.string())}),analysis_cancel:job.extend({logTail:z.array(z.string())}),analysis_list:z.object({jobs:z.array(job)}),analysis_resume:z.object({jobs:z.array(job)}),
    result_summary:z.object({id,status:z.string(),converged:z.boolean().nullable(),conditions:z.unknown(),steps:z.array(z.number()),totalSteps:z.number(),nextOffset:z.number().nullable(),warnings:z.array(z.string())}),
    results_table:z.object({row:z.number(),column:z.number(),rows:z.array(z.array(z.string())),nextRow:z.number().nullable()})
};
function template(kind) {
    if(kind==='materials')return schema.snippets?.lib_file_template?.snippet || 'material_presets:\n  air:\n    mu_r: 1.0\n';
    const bc={type:'dirichlet',value:0};
    return '# Starter model: set RGB materials and physical dimensions before running.\n'+yaml.dump({coordinate_system:kind,mesh:{dx:0.001,dy:0.001},
        ...(kind==='polar'?{polar_domain:{r_start:0.01,r_end:0.1,r_orientation:'horizontal',theta_range:'2*pi'},polar_boundary_conditions:{inner:bc,outer:bc,theta_min:{type:'periodic',value:1},theta_max:{type:'periodic',value:1}}}:{boundary_conditions:{left:bc,right:bc,top:bc,bottom:bc}}),
        materials:{air:{rgb:[255,255,255],mu_r:1,jz:0}},nonlinear_solver:{enabled:false},transient:{enabled:false}}, {noRefs:true});
}
async function table(ws,input) {
    if(!input.id.startsWith('results/')||!input.id.endsWith('.csv'))fail('INVALID_INPUT','Expected a results CSV file');
    const stream=fs.createReadStream(await ws.resolve(input.id));
    const rows=[];let index=0,pending=''; stream.setEncoding('utf8');
    const row = line => {
        if(index++<input.row)return;
        rows.push(line.replace(/\r$/,'').split(',').slice(input.column,input.column+input.columns));
    };
    try {
        for await(const chunk of stream) {
            pending+=chunk;
            let newline;
            while((newline=pending.indexOf('\n'))>=0) {
                if(newline>4*1024*1024)fail('LIMIT_EXCEEDED','CSV row exceeds limit');
                row(pending.slice(0,newline));pending=pending.slice(newline+1);
                if(rows.length>=input.rows)break;
            }
            if(rows.length>=input.rows)break;
            if(pending.length>4*1024*1024)fail('LIMIT_EXCEEDED','CSV row exceeds limit');
        }
        if(pending && rows.length<input.rows)row(pending);
    } finally {stream.destroy();}
    return {row:input.row,column:input.column,rows,nextRow:rows.length===input.rows?input.row+rows.length:null};
}
async function execute(name,input,actor,services) {
    if(!(await services.backend.workspaceEnabled?.(actor.user) ?? true)) fail('FORBIDDEN','Workspace is unassigned or disabled',403);
    const ws=new Workspace(services.base,actor.user,{guard:actor.guard,busy:id=>services.jobs.busy(actor.user,id)});
    switch(name) {
        case 'workspace_status': return {workspace:actor.user,permissions:services.access.state.connections.find(c=>c.id===actor.id).permissions,jobs:services.jobs.list(actor.user)};
        case 'files_list': return ws.list(input);
        case 'files_stat': return ws.describe(input.id);
        case 'files_read': return ws.read(input.id,input);
        case 'yaml_inspect': return core.inspect(input.content,schema,yaml);
        case 'yaml_template': return {content:template(input.kind)};
        case 'yaml_write': {
            if(!/^(configs|libraries)\/[^/]+\.ya?ml$/.test(input.id))fail('INVALID_INPUT','Expected a configuration or material YAML file');
            if(Buffer.byteLength(input.content)>100*1024)fail('LIMIT_EXCEEDED','YAML limit is 100 KiB');
            yaml.load(input.content);return ws.write(input.id,Buffer.from(input.content),input.expectedRevision);
        }
        case 'files_rename': return ws.rename(input.id,input.targetId,input.revision);
        case 'files_delete': return ws.remove(input.id,input.revision);
        case 'image_info': return services.images.info(ws,input.id);
        case 'image_process': return services.images.process(ws,input);
        case 'transfer_begin': return services.transfers.begin(actor,ws,input.kind,input.size);
        case 'transfer_append': return services.transfers.append(actor,input);
        case 'transfer_read': return services.transfers.read(actor,input);
        case 'transfer_release': return services.transfers.release(actor,input.transferId);
        case 'image_import': return services.transfers.image(actor,ws,input);
        case 'backup_export': return services.transfers.export(actor,ws,input);
        case 'backup_preview': return services.transfers.preview(actor,ws,input);
        case 'backup_import': return services.transfers.import(actor,ws,input);
        case 'analysis_start': return services.jobs.start(actor,input);
        case 'analysis_list': return {jobs:services.jobs.list(actor.user)};
        case 'analysis_get': return services.jobs.get(actor.user,input.jobId);
        case 'analysis_cancel': return services.jobs.cancel(actor.user,input.jobId,actor.guard);
        case 'analysis_resume': return {jobs:await services.jobs.resume(actor.user)};
        case 'result_summary': {
            if(!/^results\/[^/]+$/.test(input.id))fail('INVALID_INPUT','Expected a result directory ID');
            await ws.resolve(input.id);
            const state=services.jobs.list(actor.user).find(job=>job.resultId===input.id);
            let conditions=null;
            try {const data=await ws.read(`${input.id}/conditions.json`,{text:true,length:524288});if(data.nextOffset!==null)fail('LIMIT_EXCEEDED','Conditions exceed 512 KiB');conditions=JSON.parse(data.content);}
            catch(e){if(e.code!=='ENOENT')throw e;}
            const steps=new Set();let offset=0;
            do {
                const list=await ws.list({prefix:`${input.id}/Az`,offset,limit:200});
                for(const file of list.items){const match=/^step_(\d+)\.(csv|tiff)$/.exec(file.name);if(match)steps.add(Number(match[1]));}
                offset=list.nextOffset;
            }while(offset!==null);
            const ordered=[...steps].sort((a,b)=>a-b);
            return {id:input.id,status:state?.status||'unknown',converged:state?state.converged:null,conditions,
                steps:ordered.slice(input.offset,input.offset+input.limit),totalSteps:ordered.length,nextOffset:input.offset+input.limit<ordered.length?input.offset+input.limit:null,
                warnings:[...(!state?['No persisted convergence record: do not infer convergence from output files.']:state.warning||!state.converged?['These outputs are not certified as converged.']:[]),...(!conditions?['conditions.json is not available yet.']:[])]};
        }
        case 'results_table': return table(ws,input);
        case 'result_description': {
            if(!/^results\/[^/]+$/.test(input.id))fail('INVALID_INPUT','Expected a result directory ID');
            return ws.write(`${input.id}/description.txt`,Buffer.from(input.content),input.expectedRevision);
        }
        default: fail('NOT_FOUND','Unknown operation',404);
    }
}
function createToolServer(invoke) {
    const server=new McpServer({name:'OpenMagFDM',version:'1.7.0'},{instructions:'Operate only on assigned workspace IDs. Never supply OS paths. Inspect revisions before writes; deletion is permanent. Editor validation is not physical validation. Nonconverged outputs are diagnostic, not converged solutions.'});
    for(const [name,permission,description,shape] of definitions) server.registerTool(name,{
        description,inputSchema:z.object(shape).strict(),outputSchema:outputs[name],
        annotations:{readOnlyHint:permission==='read'&&!name.startsWith('transfer_'),destructiveHint:['delete','edit','images'].includes(permission),openWorldHint:false}
    },input=>invoke(name,input,permission));
    return server;
}
function serverFor(actor,services) {
    return createToolServer(async(name,input,permission)=>{
        try {
            actor.guard(permission);
            const data=await execute(name,input,actor,services);
            services.access.audit(actor,name,input.id||input.jobId||null);
            return {content:[{type:'text',text:JSON.stringify(data)}],structuredContent:data};
        } catch(e) {
            const code=e instanceof WorkspaceError?e.code:(e.code==='ENOENT'?'NOT_FOUND':'OPERATION_FAILED');
            const data={error:{code,message:e instanceof WorkspaceError?e.message:'Operation failed; check the input and local server diagnostics',...(e.applied?{applied:e.applied}:{})}};
            services.access.audit(actor,name,input.id||input.jobId||null,code);
            return {isError:true,content:[{type:'text',text:JSON.stringify(data)}],structuredContent:data};
        }
    });
}
module.exports={definitions,execute,createToolServer,serverFor,template};
