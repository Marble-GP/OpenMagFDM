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
        if (id.startsWith('results/') && this.backend.userHasActiveSolver?.(user)) return true;
        return this.records.some(j => j.user === user && ['preparing','queued','held','running','cancelling'].includes(this.current(j).status) && (id === j.resultId || id.startsWith(`${j.resultId}/`)));
    }
    current(record) {
        const live = this.backend.jobs.get(record.jobId);
        return live ? { ...record, status: live.status, progress: live.progress, completed: live.completed, converged: live.converged, warning: live.warning, exitCode: live.exitCode, finished: live.finished, message: live.message } : record;
    }
    public(record) {
        const j = this.current(record);
        return { jobId: j.jobId, source: j.source, resultId: j.resultId, status: j.status, progress: j.progress || 0,
            completed: !!j.completed, converged: !!j.converged, warning: !!j.warning, exitCode: j.exitCode ?? null,
            resultPath: j.resultPath || null, created: j.created, finished: j.finished || null, message: j.message || null };
    }
    list(user) { return this.records.filter(j => j.user === user).map(j => this.public(j)).reverse(); }
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
            const record = { key, requestHash, jobId, user: actor.user, source: actor.name, resultId, status: 'preparing', created: new Date().toISOString(), browserRequestHash: input.browserRequestHash };
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
                    await fs.mkdir(out, { recursive: false });
                    const snapshot = path.join(out, '.inputs'); await fs.mkdir(snapshot);
                    const manifest = [];
                    const copy = async (id, dest) => {
                        const from = await ws.resolve(id); const stat = await fs.stat(from);
                        if (!stat.isFile() || stat.size > 512 * 1024 * 1024) fail('LIMIT_EXCEEDED', 'Input must be a regular file of at most 512 MiB');
                        await fs.copyFile(from, dest, fs.constants.COPYFILE_EXCL);
                        manifest.push({ id, path:path.relative(out,dest).replace(/\\/g,'/'), sha256: await digest(dest), size: stat.size });
                    };
                    await copy(input.config.id,path.join(snapshot,'source-config.yaml'));
                    if(input.library)await copy(input.library.id,path.join(snapshot,'materials.yaml'));
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
                                await copy(v, dest); obj[k] = path.relative(out,dest).replace(/\\/g,'/');
                            } else if (v && typeof v === 'object') await visit(v);
                        }
                    };
                    await visit(parsed);
                    const imagePath = path.join(snapshot, 'model' + path.extname(input.image.id));
                    await copy(input.image.id, imagePath);
                    const configPath = path.join(snapshot, 'config.yaml');
                    await fs.writeFile(configPath, yaml.dump(parsed, { lineWidth: -1 }), { flag: 'wx' });
                    await atomicJSON(path.join(out, 'run-inputs.json'), { version: 1, inputs: input, files: manifest, effectiveConfigHash: await digest(configPath), solverVersion: await this.backend.getSolverVersion?.() || 'unknown' });
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
