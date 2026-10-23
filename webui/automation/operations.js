'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const {Workspace, userKey, segments, fail, noLinks} = require('./workspace');
const {atomicJSON} = require('./access');
function deletionError(error) {
    if (error.statusCode) return error.message;
    return ({EACCES:'Permission denied. Check filesystem permissions and retry.',
        EPERM:'Permission denied or file in use. Close programs using these files and check permissions.',
        EBUSY:'File is in use. Close programs using it and retry.',
        ENOTEMPTY:'Directory changed or a file is in use. Refresh and retry.',
        ENOENT:'File no longer exists. Refresh the list.',
        EROFS:'Storage is read-only. Check the workspace storage mount.'})[error.code] || 'Deletion failed. Check the server log and filesystem permissions.';
}

class FileOperations {
    constructor(base, directory, busy) {
        this.base = base; this.directory = path.join(directory, 'operations'); this.busy = busy;
        this.records = new Map(); this.pending = new Set();
        this.ready = this.init();
    }
    async init() {
        await noLinks(this.directory, {missing:true});
        await fs.mkdir(this.directory, {recursive:true, mode:0o700});
        for (const name of await fs.readdir(this.directory)) {
            if (!/^[a-f0-9-]{36}\.json$/.test(name)) continue;
            await noLinks(path.join(this.directory, name));
            const record = JSON.parse(await fs.readFile(path.join(this.directory, name), 'utf8'));
            if (record.status === 'running') {
                record.status = 'interrupted'; record.current = null;
                record.message = 'Server restarted during deletion. Some targets may already be removed; refresh files before retrying.';
                await this.save(record);
            }
            this.records.set(record.id, record);
        }
    }
    save(record) { return atomicJSON(path.join(this.directory, `${record.id}.json`), record); }
    list(user) { return [...this.records.values()].filter(r => r.user === userKey(user)).sort((a,b)=>a.created.localeCompare(b.created)).slice(-100); }
    get(user, id) {
        const record = this.records.get(id);
        if (!record || record.user !== userKey(user)) fail('NOT_FOUND', 'Operation not found', 404);
        return record;
    }
    async start(user, files) {
        await this.ready; user = userKey(user);
        if (!Array.isArray(files) || !files.length || files.length > 100) fail('INVALID_INPUT', 'Select 1..100 files');
        for (const file of files) {
            const parts = segments(file.id);
            if (parts.length !== 2 || !['results','images','configs','libraries'].includes(parts[0]) || !/^[a-f0-9]{64}$/.test(file.revision)) fail('INVALID_INPUT', 'Expected file ID and revision');
        }
        if (this.pending.has(user)) fail('BUSY', 'A deletion is already running for this workspace', 409);
        this.pending.add(user);
        const record = {id:crypto.randomUUID(), user, status:'running', created:new Date().toISOString(), total:files.length, targets:files.map(file=>file.id), current:null, results:[]};
        try { await this.save(record); this.records.set(record.id, record); }
        catch (error) { this.pending.delete(user); throw error; }
        void this.run(record, files);
        return record;
    }
    async run(record, files) {
        const ws = new Workspace(this.base, record.user, {busy:id => this.busy(record.user, id)});
        try {
            for (const file of files) {
                // Polling reads progress from memory. Persist only acceptance
                // and completion, not two full JSON rewrites per target.
                record.current = file.id;
                const started = performance.now();
                try {
                    await ws.remove(file.id, file.revision);
                    record.results.push({id:file.id, success:true, elapsedMs:performance.now()-started});
                } catch (error) {
                    record.results.push({id:file.id, success:false, error:deletionError(error), code:error.code, elapsedMs:performance.now()-started});
                }
            }
            record.status = 'complete';
        } catch { record.status = 'interrupted'; record.message = 'Deletion was interrupted; inspect remaining files before retrying.'; }
        finally {
            record.current = null; record.finished = new Date().toISOString(); this.pending.delete(record.user);
            await this.save(record).catch(() => {});
        }
    }
}
module.exports = {FileOperations, deletionError};
