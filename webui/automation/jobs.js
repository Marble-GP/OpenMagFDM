'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const yaml = require('js-yaml');
const { Workspace, fail, locked, noLinks, digest, userKey } = require('./workspace');
const { atomicJSON, hash } = require('./access');
const REFERENCE_KEYS = new Set(['initial_az_path', 'value_profile', 'gamma_profile']);
class AnalysisJobs {
    constructor(base, access, backend) {
        this.base = base; this.access = access; this.backend = backend; this.records = [];
        this.filename = path.join(access.directory, 'jobs.json');
        this.stopping = false; this.pumping = null; this.lastUser = null;
        this.served = new Map(); this.sequence = 0; this.retryTimer = null;
    }
    async init() {
        try {
            await noLinks(this.filename); this.records = JSON.parse(await fs.readFile(this.filename, 'utf8'));
            if (!Array.isArray(this.records) || this.records.length > 10000) throw Error('Invalid job records');
            for (const j of this.records) {
                if (j.status === 'queued') j.status = 'held';
                if (['preparing','running','cancelling'].includes(j.status)) {
                j.status = 'interrupted'; j.completed = false; j.converged = false;
                j.message = 'The server restarted. This analysis was not automatically restarted.';
            }
            }
            await this.save();
        } catch (e) { if (e.code !== 'ENOENT') throw e; }
    }
    save() { return locked(this.filename, () => atomicJSON(this.filename, this.records)); }
    busy(user, id) {
        user = userKey(user);
        // A connected MCP client (or a different running job) owns no handles
        // to this result. Keep the broad guard only for unscoped legacy solves.
        if (id.startsWith('results/') && this.backend.legacySolverActive?.(user)) return true;
        return this.records.some(j => j.user === user && ['preparing','queued','held','running','cancelling'].includes(this.current(j).status) && (id === j.resultId || id.startsWith(`${j.resultId}/`)));
    }
    current(record) {
        const live = this.backend.jobs.get(record.jobId);
        return live ? { ...record, status: live.status, progress: live.progress, completed: live.completed, converged: live.converged, warning: live.warning, exitCode: live.exitCode, finished: live.finished, message: live.message } : record;
    }
    public(record) {
        const j = this.current(record);
        return { jobId: j.jobId, workspace:j.user, source: j.source, configName:j.configName||null, imageName:j.imageName||null, resultId: j.resultId, status: j.status, progress: j.progress || 0,
            completed: !!j.completed, converged: !!j.converged, warning: !!j.warning, exitCode: j.exitCode ?? null,
            resultPath: j.resultPath || null, created: j.created, finished: j.finished || null, message: j.message || null };
    }
    list(user) { return this.records.filter(j => j.user === user).map(j => this.public(j)).reverse(); }
    page(user,{view='active',offset=0,limit=20}={}) {
        if(!['active','history','overview','all'].includes(view)||!Number.isInteger(offset)||offset<0||!Number.isInteger(limit)||limit<1||limit>100)fail('INVALID_INPUT','Expected job view and offset/limit (1..100)');
        const all=this.list(user),active=j=>['preparing','queued','held','running','cancelling'].includes(j.status);
        const running=all.filter(active),history=all.filter(j=>!active(j));
        // A held job may finish after many newer submissions. Include recent
        // completions in the overview so clients notice that transition.
        const recent=history.slice().sort((a,b)=>(b.finished||b.created||'').localeCompare(a.finished||a.created||''));
        const selected=view==='overview'?[...running.slice(0,50),...recent.slice(0,20)]:view==='active'?running:view==='history'?recent:all;
        return {jobs:view==='overview'?selected:selected.slice(offset,offset+limit),counts:{active:running.length,history:history.length},total:selected.length,offset,limit,nextOffset:view!=='overview'&&offset+limit<selected.length?offset+limit:null};
    }
    get(user, jobId) {
        const record = this.records.find(j => j.user === user && j.jobId === jobId);
        if (!record) fail('NOT_FOUND', 'Job not found', 404);
        const live = this.backend.jobs.get(jobId);
        const scrub = line => String(line).split(this.base).join('[workspace]');
        return { ...this.public(record), logTail: (live?.log || record.logTail || []).slice(-50).map(scrub) };
    }
    async cancel(user, jobId, guard) {
        this.get(user, jobId); guard('cancel');
        const record = this.records.find(j => j.user === user && j.jobId === jobId);
        if (['queued','held'].includes(record.status)) { record.status = 'cancelled'; record.finished = new Date().toISOString(); await this.save(); return this.get(user, jobId); }
        const live = this.backend.jobs.get(jobId);
        if (record.status === 'preparing' && !live) {record.cancelRequested=true;await this.save();}
        if (live?.process && this.backend.isChildRunning(live.process)) {
            live.cancelRequested = true; live.status = 'cancelling';
            void this.backend.terminateChildProcess(live.process);
        }
        return this.get(user, jobId);
    }
    async resume(user) {
        for (const record of this.records) if (record.user === user && record.status === 'held') record.status = 'queued';
        await this.save(); await this.pump(); return this.list(user);
    }
    async pump() {
        if (this.stopping) return;
        if (this.pumping) return this.pumping;
        this.pumping = require('./identity-context').context.run(undefined,()=>this.dispatch()).finally(() => {
            this.pumping = null;
            // CLI enable/assignment and legacy process completion do not emit job events.
            // A bounded retry also handles transient backend capacity limits.
            if (!this.stopping && !this.retryTimer && this.records.some(j => j.status === 'queued')) {
                this.retryTimer = setTimeout(() => {this.retryTimer=null;void this.pump().catch(()=>{});}, 1000);
                this.retryTimer.unref?.();
            }
        });
        return this.pumping;
    }
    stop() {this.stopping=true;clearTimeout(this.retryTimer);this.retryTimer=null;}
    async dispatch() {
        while (!this.stopping) {
            const running = [...this.backend.jobs.values()].filter(j => ['running','cancelling'].includes(j.status));
            if (running.length >= (this.backend.maxConcurrentJobs || 2)) break;
            const candidates = [];
            for(const j of this.records) if(j.status==='queued' && !running.some(r=>r.userId===j.user) && !this.backend.userHasActiveSolver?.(j.user) && (await this.backend.workspaceEnabled?.(j.user) ?? true)) candidates.push(j);
            // Least recently served workspace; stable ordering preserves FIFO within it.
            const record = candidates.sort((a,b)=>(this.served.get(a.user)||0)-(this.served.get(b.user)||0))[0];
            if (!record) break;
            const ws = new Workspace(this.base, record.user);
            // Resolve again after restart/storage migration; never trust persisted absolute paths.
            let out;
            try {out=await ws.resolve(record.resultId);}catch(error){record.status='failed';record.message='Saved input directory unavailable';record.finished=new Date().toISOString();await this.save();continue;}
            try {await require('./limits').checkResults(this.base,record.user,{starting:true});}
            catch(error){if(error.code!=='QUOTA_EXCEEDED')throw error;record.status='held';record.message=error.message+' Adjust limits, then resume the queue.';await this.save();continue;}
            const prepared = {outputPath:out, configPath:path.join(out,'.inputs','config.yaml'), imagePath:path.join(out,'.inputs',path.basename(record.prepared.imagePath))};
            try {
                record.status = 'preparing'; await this.save();
                await this.backend.submitAnalysisJob({configFile:'snapshot',imageFile:'snapshot',userId:record.user}, {
                    jobId:record.jobId, source:record.source, prepared,
                    guard:()=>{if(record.cancelRequested)fail('CANCELLED','Cancelled before solver launch',409);},
                    onFinalized: async job => {
                        Object.assign(record, {status:job.status,progress:job.progress,completed:job.completed,converged:job.converged,warning:job.warning,exitCode:job.exitCode,finished:job.finished,message:job.message,logTail:(job.log||[]).slice(-50)});
                        await this.save();
                        setImmediate(() => { void this.pump().catch(() => {}); });
                    }
                });
                Object.assign(record, this.public(record)); this.lastUser = record.user;
                this.served.set(record.user,++this.sequence); await this.save();
            } catch (error) {
                if (['JOB_LIMIT','USER_JOB_LIMIT','SOLVER_ALREADY_RUNNING','SOLVER_PROCESS_LIMIT','SOLVER_START_LIMIT'].includes(error.code)) { record.status='queued'; await this.save(); break; }
                record.status=record.cancelRequested?'cancelled':'failed'; record.message=error.message; record.finished=new Date().toISOString(); await this.save();
            }
        }
    }

    async start(actor, input) {
        actor.guard('run');
        const requestHash = hash(JSON.stringify(input));
        const key = `${actor.id}:${input.idempotencyKey}`;
        return locked(`${this.filename}:${actor.user}`, async () => {
            const prior = this.records.find(j => j.key === key);
            if (prior) {
                if (prior.requestHash !== requestHash) fail('IDEMPOTENCY_CONFLICT', 'This key was used with different inputs', 409);
                return this.public(prior);
            }
            if (this.records.length >= 10000) fail('JOB_HISTORY_LIMIT', 'The durable job ledger is full; retain it and contact the application administrator', 409);
            const ws = new Workspace(this.base, actor.user, { guard: actor.guard });
            await ws.init();
            const jobId = crypto.randomUUID(); const resultId = `results/output_mcp_${jobId}`;
            const out = await ws.resolve(resultId, { create: true });
            const record = { key, requestHash, jobId, user: actor.user, source: actor.name, configName:input.config.id.split('/').pop(), imageName:input.image.id.split('/').pop(), resultId, status: 'preparing', created: new Date().toISOString(), browserRequestHash: input.browserRequestHash };
            // Record BEFORE spawning. An uncertain/restarted request never launches twice.
            this.records.push(record); await this.save();
            try {
                await locked(ws.lockKey, async () => {
                    actor.guard('run');
                    for (const [entry, category] of [[input.config,'configs'],[input.image,'images'],[input.library,'libraries']]) {
                        if (!entry) continue;
                        if (!entry.id.startsWith(`${category}/`)) fail('INVALID_INPUT', `Expected ${category} input`);
                        await ws.checkRevision(entry.id, entry.revision);
                    }
                    const config = await ws.readUnlocked(input.config.id, { text: true, length: 512 * 1024 });
                    if (config.nextOffset !== null) fail('LIMIT_EXCEEDED', 'Configuration too large');
                    let content = config.content;
                    if (input.library) {
                        const lib = await ws.readUnlocked(input.library.id, { text: true, length: 512 * 1024 });
                        if (lib.nextOffset !== null) fail('LIMIT_EXCEEDED', 'Material library too large');
                        content = this.backend.mergeLibraryIntoConfig(content, lib.content);
                    }
                    const parsed = yaml.load(content);
                    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) fail('INVALID_CONFIG', 'Expected a configuration mapping');
                    const snapshot = path.join(out, '.inputs');
                    const manifest = [], copies=[];
                    const planCopy = async (id, dest) => {
                        const from = await ws.resolve(id); const stat = await fs.stat(from);
                        if (!stat.isFile() || stat.size > 512 * 1024 * 1024) fail('LIMIT_EXCEEDED', 'Input must be a regular file of at most 512 MiB');
                        const sha256=await digest(from);
                        const after=await fs.stat(from);
                        if(after.size!==stat.size||after.mtimeMs!==stat.mtimeMs||after.ctimeMs!==stat.ctimeMs)fail('REVISION_CONFLICT','Input changed while planning its snapshot',409);
                        copies.push({id,dest,sha256});
                        manifest.push({ id, path:path.relative(out,dest).replace(/\\/g,'/'), sha256, size: stat.size });
                    };
                    await planCopy(input.config.id,path.join(snapshot,'source-config.yaml'));
                    if(input.library)await planCopy(input.library.id,path.join(snapshot,'materials.yaml'));
                    let visited = 0; const seen = new WeakSet();
                    const visit = async obj => {
                        if (!obj || typeof obj !== 'object') return;
                        if (seen.has(obj)) fail('INVALID_CONFIG', 'Recursive/aliased configuration objects are not supported for automation');
                        seen.add(obj);
                        for (const [k,v] of Object.entries(obj)) {
                            if (++visited > 50000) fail('LIMIT_EXCEEDED', 'Configuration is too complex');
                            if (REFERENCE_KEYS.has(k) && v !== '') {
                                if (typeof v !== 'string') fail('INVALID_PATH', 'File references must be workspace IDs');
                                // Require explicit category IDs; do not interpret paths relative to process cwd.
                                const dest = path.join(snapshot, `reference-${manifest.length}.bin`);
                                await planCopy(v, dest); obj[k] = path.relative(out,dest).replace(/\\/g,'/');
                            } else if (v && typeof v === 'object') await visit(v);
                        }
                    };
                    await visit(parsed);
                    const imagePath = path.join(snapshot, 'model' + path.extname(input.image.id));
                    await planCopy(input.image.id, imagePath);
                    const configPath = path.join(snapshot, 'config.yaml');
                    const effectiveConfig=yaml.dump(parsed, { lineWidth: -1 });
                    const metadata={ version: 1, inputs: input, files: manifest, effectiveConfigHash: hash(effectiveConfig), solverVersion: await this.backend.getSolverVersion?.() || 'unknown' };
                    const bytes=manifest.reduce((sum,file)=>sum+file.size,0)+Buffer.byteLength(effectiveConfig)+Buffer.byteLength(JSON.stringify(metadata,null,2));
                    await require('./limits').checkResults(this.base,actor.user,{newResults:1,bytes,starting:true});
                    await fs.mkdir(out, { recursive: false });
                    try {
                        await fs.mkdir(snapshot);
                        for(const file of copies) {
                            await fs.copyFile(await ws.resolve(file.id),file.dest,fs.constants.COPYFILE_EXCL);
                            if(await digest(file.dest)!==file.sha256)fail('REVISION_CONFLICT','Input changed while creating its snapshot',409);
                        }
                        await fs.writeFile(configPath,effectiveConfig,{flag:'wx'});
                        await atomicJSON(path.join(out,'run-inputs.json'),metadata);
                        actor.guard('run');
                    } catch(error) {
                        await noLinks(out);await fs.rm(out,{recursive:true,force:true});throw error;
                    }
                    actor.guard('run');
                    record.prepared = {configPath, imagePath, outputPath:out};
                    record.resultPath = `outputs/${record.user}/${path.basename(out)}`;
                    record.status = 'queued';
                    await this.save();
                });
                await this.pump();
                return this.public(record);
            } catch (e) {
                record.status = 'failed'; record.message = 'Analysis preparation failed'; record.finished = new Date().toISOString(); await this.save(); throw e;
            }
        });
    }
}
module.exports = { AnalysisJobs, REFERENCE_KEYS };
