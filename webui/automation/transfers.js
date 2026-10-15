'use strict';
const fs = require('node:fs/promises');
const fsSync = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { pipeline } = require('node:stream/promises');
const { Transform } = require('node:stream');
const yauzl = require('yauzl');
const yazl = require('yazl');
const yaml = require('js-yaml');
const { fail, noLinks, locked, segments, roots } = require('./workspace');
const { inspectImage } = require('./images');
const MAX_ARCHIVE = 1024 * 1024 * 1024;
class Transfers {
    constructor() {
        this.records = new Map();
        this.timer = setInterval(() => { void this.cleanup(); }, 60000); this.timer.unref();
    }
    async close() { clearInterval(this.timer); await this.cleanup(true); }
    async cleanup(all = false) {
        for (const [id,r] of this.records) if (!r.busy && (all || Date.now() > r.expires))
            await locked(`transfer:${id}`,async()=>{
                if(r.busy)return;
                try { await noLinks(r.filename); await fs.rm(r.filename, {force:true}); } catch {}
                this.records.delete(id);
            });
    }
    get(actor, id) {
        const r = this.records.get(id);
        if (!r || r.actor !== actor.id || r.user !== actor.user || Date.now() > r.expires) fail('NOT_FOUND','Transfer not found or expired',404);
        return r;
    }
    async release(actor,id) {
        return locked(`transfer:${id}`,async()=>{
            const r=this.get(actor,id);
            if(r.busy)fail('BUSY','Transfer is in use',409);
            await noLinks(r.filename);await fs.rm(r.filename,{force:true});this.records.delete(id);
            return {released:id};
        });
    }
    async begin(actor, ws, kind, size = 0) {
        return locked('mcp-transfer-allocation',()=>this.allocate(actor,ws,kind,size));
    }
    async allocate(actor, ws, kind, size) {
        await this.cleanup();
        if (this.records.size >= 32 || [...this.records.values()].filter(r=>r.user===actor.user).length >= 4) fail('LIMIT_EXCEEDED','Too many active transfers',429);
        actor.guard(kind === 'download' ? 'read' : kind === 'image' ? 'images' : 'edit');
        if (!Number.isSafeInteger(size) || size<0 || size>MAX_ARCHIVE || (kind==='image' && size>10*1024*1024)) fail('LIMIT_EXCEEDED','Transfer is too large');
        const id=crypto.randomUUID();
        const filename=await ws.resolve(`results/.mcp-transfers/${id}.bin`,{create:true,internal:true});
        await fs.mkdir(path.dirname(filename),{recursive:true});
        actor.guard(kind === 'download' ? 'read' : kind === 'image' ? 'images' : 'edit');
        await fs.writeFile(filename,Buffer.alloc(0),{flag:'wx',mode:0o600});
        const r={id,filename,kind,size,received:0,actor:actor.id,user:actor.user,expires:Date.now()+3600000,ready:kind==='download',busy:false};
        this.records.set(id,r); return {transferId:id,size,chunkBytes:256*1024,expires:new Date(r.expires).toISOString()};
    }
    async append(actor,input) {
        return locked(`transfer:${input.transferId}`,async()=>{
            const r=this.get(actor,input.transferId); actor.guard(r.kind==='image'?'images':'edit');
            if(r.kind==='download'||r.ready) fail('INVALID_STATE','Transfer is not writable');
            if(input.offset!==r.received) fail('OFFSET_CONFLICT','Resume from the returned received offset',409);
            if(!/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(input.base64)) fail('INVALID_INPUT','Invalid base64');
            const b=Buffer.from(input.base64,'base64');
            if(b.length>256*1024||b.length===0||r.received+b.length>r.size) fail('LIMIT_EXCEEDED','Invalid chunk length');
            await noLinks(r.filename); actor.guard(r.kind==='image'?'images':'edit');
            await fs.appendFile(r.filename,b); r.received+=b.length; r.ready=r.received===r.size;
            return {transferId:r.id,received:r.received,ready:r.ready};
        });
    }
    async read(actor,input) {
        return locked(`transfer:${input.transferId}`,()=>this.readChunk(actor,input));
    }
    async readChunk(actor,input) {
        const r=this.get(actor,input.transferId); actor.guard('read');
        if(!r.ready||r.kind!=='download') fail('INVALID_STATE','Download is not ready');
        await noLinks(r.filename); const handle=await fs.open(r.filename,'r');
        try {
            const b=Buffer.alloc(Math.min(256*1024,Math.max(0,r.size-input.offset)));
            const {bytesRead}=await handle.read(b,0,b.length,input.offset);
            return {base64:b.subarray(0,bytesRead).toString('base64'),size:r.size,nextOffset:input.offset+bytesRead<r.size?input.offset+bytesRead:null};
        } finally {await handle.close();}
    }
    async image(actor,ws,input) {
        return locked(`transfer:${input.transferId}`,()=>this.publishImage(actor,ws,input));
    }
    async publishImage(actor,ws,input) {
        const r=this.get(actor,input.transferId);
        if(r.kind!=='image'||!r.ready) fail('INVALID_STATE','Image transfer is not ready');
        await noLinks(r.filename);
        const b=await fs.readFile(r.filename); inspectImage(b);
        if(!/^images\/[^/]+\.(png|jpe?g|bmp)$/i.test(input.id)) fail('INVALID_INPUT','Invalid image filename');
        return ws.write(input.id,b,input.expectedRevision,{permission:'images'});
    }
    async export(actor,ws,input) {
        const transfer=await this.begin(actor,ws,'download'); const r=this.get(actor,transfer.transferId); r.busy=true; r.ready=false;
        const zip=new yazl.ZipFile();
        try {
            return await locked(ws.lockKey,async()=>{
            let total=0;
            for(const entry of input.files) {
                if(ws.busy(entry.id))fail('BUSY','Wait until the analysis finishes before exporting its output',409);
                await ws.checkRevision(entry.id,entry.revision);
                const meta=await ws.describe(entry.id);
                const files=meta.kind==='directory'?await ws.walk(entry.id):[meta];
                for(const f of files) {
                    total+=f.size; if(total>MAX_ARCHIVE) fail('LIMIT_EXCEEDED','Selected files exceed the 1 GiB backup limit');
                    zip.addFile(await ws.resolve(f.id),f.id,{compress:false});
                }
            }
            let bytes=0;
            const limit=new Transform({transform(chunk,encoding,done){bytes+=chunk.length; done(bytes>MAX_ARCHIVE?Error('Archive size limit exceeded'):null,chunk);}});
            const writing=pipeline(zip.outputStream,limit,fsSync.createWriteStream(r.filename));
            zip.end(); await writing;
            actor.guard('read');
            r.size=bytes;r.ready=true;return {...transfer,size:bytes};
            });
        } catch(e) {zip.outputStream.destroy();throw e;} finally {r.busy=false;}
    }
    async entries(actor,input,consume) {
        return locked(`transfer:${input.transferId}`,()=>this.readEntries(actor,input,consume));
    }
    async readEntries(actor,input,consume) {
        const r=this.get(actor,input.transferId);
        if(r.kind!=='backup'||!r.ready) fail('INVALID_STATE','Backup upload is not ready');
        await noLinks(r.filename); r.busy=true;
        try {
            const zip=await new Promise((resolve,reject)=>yauzl.open(r.filename,{lazyEntries:true,validateEntrySizes:true,strictFileNames:true},(e,z)=>e?reject(e):resolve(z)));
            return await new Promise((resolve,reject)=>{
                let count=0,total=0; const names=new Set(); let ended=false;
                const stop=e=>{if(ended)return;ended=true;zip.close();reject(e);};
                zip.on('error',stop);zip.on('end',()=>{ended=true;resolve();});
                zip.on('entry',entry=>{
                    void (async()=>{
                        const name=entry.fileName;
                        if(++count>10000||(total+=entry.uncompressedSize)>MAX_ARCHIVE) fail('LIMIT_EXCEEDED','Expanded backup exceeds limits');
                        if((entry.externalFileAttributes>>>16 & 0xf000)===0xa000 || entry.generalPurposeBitFlag&1) fail('UNSAFE_ARCHIVE','Links and encrypted ZIP entries are not accepted');
                        if(name!=='manifest.json') {
                            const parts=segments(name.replace(/\/$/,''));
                            if(!Object.hasOwn(roots,parts[0]))fail('UNSAFE_ARCHIVE','Unknown archive category');
                        }
                        if(name.endsWith('/')) return;
                        if(names.has(name.toLowerCase())) fail('UNSAFE_ARCHIVE','Duplicate archive target');names.add(name.toLowerCase());
                        await consume(name,entry,()=>new Promise((resolve,reject)=>zip.openReadStream(entry,(e,s)=>e?reject(e):resolve(s))));
                    })().then(()=>zip.readEntry(),stop);
                });zip.readEntry();
            });
        } finally {r.busy=false;}
    }
    async preview(actor,ws,input) {
        const files=[];
        await this.entries(actor,input,async(name,entry)=>{
            if(name==='manifest.json')return;
            await ws.resolve(name,{create:true});
            const current=await ws.describe(name).catch(e=>{if(e.code!=='ENOENT')throw e;return null;});
            files.push({id:name,size:entry.uncompressedSize,expectedRevision:current?.revision??null});
        });
        return {files,permanentReplacement:true};
    }
    async import(actor,ws,input) {
        actor.guard('edit');const applied=[];
        const expected=new Map(input.files.map(f=>[f.id,f.revision]));
        // Validate the complete manifest before writing the first file.
        const preview=await this.preview(actor,ws,input);
        if(preview.files.length!==expected.size) fail('REVISION_CONFLICT','Import manifest does not match the archive',409);
        for(const f of preview.files) {
            if(!expected.has(f.id)||expected.get(f.id)!==f.expectedRevision)fail('REVISION_CONFLICT','Import target changed',409);
            if(f.expectedRevision!==null)actor.guard('replace');
        }
        await this.entries(actor,input,async(name,entry,stream)=>{
            if(name==='manifest.json')return;
            const size=entry.uncompressedSize;
            // Small editable inputs are validated in memory. Result artifacts stream to staging.
            if(name.startsWith('configs/')||name.startsWith('libraries/')||name.startsWith('images/')) {
                const max=name.startsWith('images/')?10*1024*1024:100*1024;
                if(size>max)fail('LIMIT_EXCEEDED','Imported input exceeds its category limit');
                const chunks=[];let received=0;for await(const b of await stream()){received+=b.length;if(received>max)fail('LIMIT_EXCEEDED','Expanded input too large');chunks.push(b);}
                const buffer=Buffer.concat(chunks);
                if(name.startsWith('images/')){actor.guard('images');inspectImage(buffer);}else yaml.load(buffer.toString('utf8'));
                await ws.write(name,buffer,expected.get(name),{permission:name.startsWith('images/')?'images':'edit'});
            } else {
                await locked(ws.lockKey,async()=>{
                    await ws.checkRevision(name,expected.get(name)); if(ws.busy(name))fail('BUSY','Active result cannot be replaced',409);
                    const p=await ws.resolve(name,{create:true});await fs.mkdir(path.dirname(p),{recursive:true});
                    const tmp=path.join(path.dirname(p),`.import-${crypto.randomUUID()}`);
                    try {
                        await pipeline(await stream(),fsSync.createWriteStream(tmp,{flags:'wx'}));
                        await ws.resolve(name,{create:true});actor.guard('edit');if(expected.get(name)!==null)actor.guard('replace');
                        await fs.rename(tmp,p);
                    } finally {await fs.rm(tmp,{force:true});}
                });
            }
            applied.push(name);
        }).catch(e=>{e.applied=applied;throw e;});
        return {applied};
    }
}
module.exports={Transfers};
