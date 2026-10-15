'use strict';
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const {pipeline} = require('node:stream/promises');
const yazl = require('yazl');
const {Workspace, roots, fail, segments} = require('./workspace');

// Adapters retain the GUI backup workflow while using the MCP streaming service.
// Preview and commit are separate: opening a backup never replaces existing data.
function installBackupRoutes(app, base, runtime, upload) {
    const context = req => {
        const user = req.body?.userId || req.query.userId || 'default';
        const ws = new Workspace(base, user, {busy: id => runtime().jobs.busy(user, id)});
        return {ws, actor: {id:`gui:${ws.user}`, user:ws.user, name:'WebUI', guard:()=>{}}, transfers:runtime().transfers};
    };
    const wrap = fn => async (req, res) => {
        res.set('Cache-Control','no-store');
        try { await fn(req,res,context(req)); }
        catch(e) {
            if (!res.headersSent) res.status(e.statusCode || 400).json({success:false, error:e.statusCode?e.message:'Backup operation failed', code:e.code || 'BACKUP_FAILED', applied:e.applied || []});
            else res.destroy();
        }
    };
    app.get('/api/backup-manifest',wrap(async(req,res,{ws})=>{
        const categories={};let totalSize=0;
        for(const category of Object.keys(roots)) {
            categories[category]=(await ws.walk(category)).map(f=>({path:f.id.slice(category.length+1),size:f.size}));
            for(const file of categories[category])totalSize+=file.size;
        }
        res.json({success:true,categories,totalSize});
    }));
    const exportBackup=wrap(async(req,res,{ws,actor,transfers})=>{
        let entries=req.body?.entries;
        if(req.method==='GET') {
            entries=[];
            for(const category of String(req.query.include||'configs,images,libraries').split(',')) {
                if(!Object.hasOwn(roots,category))fail('INVALID_INPUT','Unknown backup category');
                entries.push(...(await ws.walk(category)).map(f=>f.id));
            }
        }
        if(!Array.isArray(entries)||!entries.length||entries.length>10000||new Set(entries).size!==entries.length)
            fail('INVALID_INPUT','Select 1..10000 unique files');
        const files=[];
        for(const id of entries) {const meta=await ws.describe(id);files.push({id,revision:meta.revision});}
        const download=await transfers.export(actor,ws,{files});
        const record=transfers.get(actor,download.transferId);record.busy=true;
        try {
            await new Promise((resolve,reject)=>res.download(record.filename,`omfdm-backup-${new Date().toISOString().slice(0,10)}.zip`,e=>e?reject(e):resolve()));
        } finally {record.busy=false;await transfers.release(actor,download.transferId);}
    });
    app.get('/api/export',exportBackup);app.post('/api/export',exportBackup);
    app.post('/api/import',upload.single('backup'),wrap(async(req,res,{ws,actor,transfers})=>{
        if(!req.file)fail('INVALID_INPUT','Select a backup file');
        let transfer;
        try {
            let filename=req.file.path, size=req.file.size;
            const handle=await fsp.open(filename,'r');const magic=Buffer.alloc(2);
            try {await handle.read(magic,0,2,0);} finally {await handle.close();}
            if(magic.toString()!=='PK') {
                // Keep the old JSON import format, but bound it and convert once
                // to a streamed ZIP so the same target/path checks apply.
                if(size>16*1024*1024)fail('LIMIT_EXCEEDED','Legacy JSON backup limit is 16 MiB; use ZIP for larger backups');
                const legacy=JSON.parse(await fsp.readFile(filename,'utf8'));
                if(legacy.format!=='omfdm-backup'||!Array.isArray(legacy.files)||legacy.files.length>10000)fail('INVALID_INPUT','Not an OpenMagFDM backup');
                const zip=new yazl.ZipFile(),converted=filename+'.zip';
                try {
                    for(const file of legacy.files) {
                        const category={config:'configs',upload:'images',library:'libraries'}[file.category]||file.category;
                        const id=`${category}/${file.name}`;segments(id);await ws.resolve(id,{create:true});
                        zip.addBuffer(Buffer.from(file.content,file.encoding==='base64'?'base64':'utf8'),id);
                    }
                    const writing=pipeline(zip.outputStream,fs.createWriteStream(converted,{flags:'wx'}));
                    zip.end();await writing;filename=converted;size=(await fsp.stat(filename)).size;
                } catch(e) {zip.outputStream.destroy();throw e;}
            }
            transfer=await transfers.begin(actor,ws,'backup',size);let offset=0;
            for await(const chunk of fs.createReadStream(filename,{highWaterMark:256*1024})) {
                await transfers.append(actor,{transferId:transfer.transferId,offset,base64:chunk.toString('base64')});offset+=chunk.length;
            }
            const preview=await transfers.preview(actor,ws,transfer);
            res.json({success:true,transferId:transfer.transferId,...preview});
        } catch(e) {if(transfer)await transfers.release(actor,transfer.transferId);throw e;}
        finally {await fsp.rm(req.file.path,{force:true});await fsp.rm(req.file.path+'.zip',{force:true});}
    }));
    app.post('/api/import/commit',wrap(async(req,res,{ws,actor,transfers})=>{
        if(!Array.isArray(req.body.files)||req.body.files.length>10000)fail('INVALID_INPUT','Expected inspected targets');
        const result=await transfers.import(actor,ws,{transferId:req.body.transferId,files:req.body.files});
        await transfers.release(actor,req.body.transferId);
        const written={};for(const id of result.applied){const cat=id.split('/')[0];written[cat]=(written[cat]||0)+1;}
        res.json({success:true,written,skipped:[]});
    }));
    app.delete('/api/import/:transferId',wrap(async(req,res,{actor,transfers})=>{
        await transfers.release(actor,req.params.transferId);res.json({success:true});
    }));
}
module.exports={installBackupRoutes};
