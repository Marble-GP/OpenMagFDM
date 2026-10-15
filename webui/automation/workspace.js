'use strict';
// The only filesystem boundary exposed to automation. IDs are logical, never OS paths.
const fs = require('node:fs/promises');
const fsSync = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { pipeline } = require('node:stream/promises');
const roots = Object.freeze({ configs: 'configs', libraries: 'user-libs', images: 'uploads', results: 'outputs' });
const queues = new Map();
class WorkspaceError extends Error {
    constructor(code, message, status = 400) { super(message); this.code = code; this.statusCode = status; }
}
function fail(code, message, status) { throw new WorkspaceError(code, message, status); }
function userKey(user) {
    if (typeof user !== 'string' || !/^[A-Za-z0-9_-]{1,128}$/.test(user)) fail('INVALID_USER', 'Invalid workspace identity');
    return process.platform === 'win32' ? user.toLowerCase() : user;
}
function segments(value, internal = false) {
    if (typeof value !== 'string' || !value || value.length > 1024 || /[\\:\x00-\x1f\x7f%]/.test(value)) fail('INVALID_PATH', 'Use a workspace file ID, not a filesystem path');
    const parts = value.split('/');
    for (const s of parts) {
        if (!s || s === '.' || s === '..' || /[. ]$/.test(s) || /[<>"|?*]/.test(s) || (!internal && s.startsWith('.')) || /^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)/i.test(s)) fail('INVALID_PATH', 'Invalid workspace file ID');
    }
    return parts;
}
async function noLinks(target, { missing = false } = {}) {
    const absolute = path.resolve(target);
    const parsed = path.parse(absolute);
    let current = parsed.root;
    for (const piece of absolute.slice(parsed.root.length).split(path.sep).filter(Boolean)) {
        current = path.join(current, piece);
        try {
            const st = await fs.lstat(current);
            if (st.isSymbolicLink()) fail('UNSAFE_PATH', 'Links and junctions are not allowed');
            if (st.isFile() && st.nlink > 1) fail('UNSAFE_PATH', 'Hard-linked files are not allowed');
            const real = await fs.realpath(current);
            if (path.relative(current, real) !== '') fail('UNSAFE_PATH', 'Redirected filesystem paths are not allowed');
        } catch (e) { if (missing && e.code === 'ENOENT') return; throw e; }
    }
}
function locked(key, fn) {
    const previous = queues.get(key) || Promise.resolve();
    const pending = previous.catch(() => {}).then(fn);
    queues.set(key, pending);
    return pending.finally(() => { if (queues.get(key) === pending) queues.delete(key); });
}
async function digest(filename) {
    const hash = crypto.createHash('sha256');
    await pipeline(fsSync.createReadStream(filename), hash);
    return hash.digest('hex');
}
class Workspace {
    constructor(base, user, { guard = () => {}, busy = () => false } = {}) {
        this.base = path.resolve(base); this.user = userKey(user); this.guard = guard; this.busy = busy;
        this.lockKey = `${this.base}:${this.user}`;
    }
    async resolve(id, { create = false, internal = false } = {}) {
        const [category, ...tail] = segments(id, internal);
        if (!Object.hasOwn(roots, category) || !tail.length) fail('INVALID_PATH', 'A category and filename are required');
        const root = path.join(this.base, roots[category], this.user);
        const target = path.join(root, ...tail);
        if (!path.relative(root, target) || path.relative(root, target).startsWith('..') || path.isAbsolute(path.relative(root, target))) fail('INVALID_PATH', 'Invalid target');
        await noLinks(target, { missing: create });
        return target;
    }
    async init() {
        for (const folder of Object.values(roots)) {
            const root = path.join(this.base, folder, this.user);
            await noLinks(root, { missing: true });
            await fs.mkdir(root, { recursive: true });
        }
    }
    async describe(id) {
        const p = await this.resolve(id); const stat = await fs.lstat(p);
        if (!stat.isFile() && !stat.isDirectory()) fail('UNSAFE_PATH', 'Only regular files and directories are supported');
        if (stat.isFile()) return { id, name: path.basename(p), kind: 'file', size: stat.size, modified: stat.mtime.toISOString(), revision: await digest(p) };
        const entries = await this.walk(id);
        const hash = crypto.createHash('sha256');
        hash.update(JSON.stringify([stat.mtimeMs, stat.ctimeMs]));
        let size = 0;
        for (const e of entries) { hash.update(JSON.stringify([e.id, e.size, e.modified, e.changed])); size += e.size; }
        return { id, name: path.basename(p), kind: 'directory', size, modified: stat.mtime.toISOString(), revision: hash.digest('hex'), count: entries.length };
    }
    async walk(prefix) {
        const entries = []; let visited = 0;
        const visit = async (dir, id, depth = 0, hidden = false) => {
            if (depth > 64) fail('LIMIT_EXCEEDED', 'Directory nesting exceeds 64 levels', 413);
            await noLinks(dir);
            const children = (await fs.readdir(dir, { withFileTypes: true })).sort((a,b) => a.name.localeCompare(b.name));
            for (const e of children) {
                if (++visited > 10000) fail('LIMIT_EXCEEDED', 'Directory exceeds the 10000-entry operation limit', 413);
                const nextId = `${id}/${e.name}`; const isHidden = hidden || e.name.startsWith('.');
                const p = await this.resolve(nextId, {internal: isHidden});
                const st = await fs.lstat(p);
                if (st.isDirectory()) await visit(p, nextId, depth + 1, isHidden);
                else if (st.isFile()) {
                    if (!isHidden) entries.push({ id: nextId, name: e.name, size: st.size, modified: st.mtime.toISOString(), changed: st.ctimeMs, kind: 'file' });
                }
                else fail('UNSAFE_PATH', 'Non-regular workspace entry');
                if (entries.length > 10000) fail('LIMIT_EXCEEDED', 'Directory exceeds the 10000-entry operation limit', 413);
            }
        };
        if (Object.hasOwn(roots, prefix)) {
            const root = path.join(this.base, roots[prefix], this.user);
            try { await visit(root, prefix); } catch (e) { if (e.code !== 'ENOENT') throw e; }
        } else await visit(await this.resolve(prefix), prefix);
        return entries;
    }
    async list({ category, prefix, offset = 0, limit = 50 } = {}) {
        const categories = category ? [category] : Object.keys(roots);
        const items = []; let total = 0;
        for (const c of prefix ? [prefix.split('/')[0]] : categories) {
            if (!Object.hasOwn(roots, c)) fail('INVALID_PATH', 'Unknown category');
            if (category && prefix && category !== c) fail('INVALID_PATH', 'Prefix and category must agree');
            const parent = prefix || c;
            const dir = prefix ? await this.resolve(prefix, {create: true}) : path.join(this.base, roots[c], this.user);
            await noLinks(dir, {missing: true});
            let handle;
            try { handle = await fs.opendir(dir); } catch (e) { if (e.code === 'ENOENT') continue; throw e; }
            for await (const e of handle) {
                if (e.name.startsWith('.')) continue;
                if (++total > 100000) fail('LIMIT_EXCEEDED', 'Directory exceeds the 100000-entry listing limit', 413);
                if (total <= offset || items.length >= limit) continue;
                const id = `${parent}/${e.name}`, p = await this.resolve(id), stat = await fs.lstat(p);
                if (!stat.isDirectory() && !stat.isFile()) fail('UNSAFE_PATH', 'Non-regular workspace entry');
                items.push({id, name:e.name, kind:stat.isDirectory()?'directory':'file', size:stat.isFile()?stat.size:null, modified:stat.mtime.toISOString()});
            }
        }
        return {items, total, nextOffset:offset + items.length < total ? offset + items.length : null};
    }
    async read(id, options = {}) {
        return locked(this.lockKey, () => this.readUnlocked(id, options));
    }
    // Internal only: callers already holding lockKey (input snapshot creation).
    async readUnlocked(id, { offset = 0, length = 256 * 1024, text = false } = {}) {
        if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isInteger(length) || length < 1 || length > 512 * 1024) fail('LIMIT_EXCEEDED', 'Read length must be 1..524288 bytes');
        const meta = await this.describe(id);
        if (meta.kind !== 'file') fail('INVALID_PATH', 'Expected a file');
        const p = await this.resolve(id); const h = await fs.open(p, 'r');
        try {
            const b = Buffer.alloc(Math.min(length, Math.max(0, meta.size - offset)));
            const { bytesRead } = await h.read(b, 0, b.length, offset);
            return { ...meta, offset, nextOffset: offset + bytesRead < meta.size ? offset + bytesRead : null, encoding: text ? 'utf8' : 'base64', content: b.subarray(0, bytesRead).toString(text ? 'utf8' : 'base64') };
        } finally { await h.close(); }
    }
    async checkRevision(id, expected) {
        if (expected !== null && (typeof expected !== 'string' || !/^[a-f0-9]{64}$/.test(expected))) fail('PRECONDITION_REQUIRED', 'A revision (or null for creation) is required', 428);
        let current = null;
        try { current = await this.describe(id); } catch (e) { if (e.code !== 'ENOENT') throw e; }
        if ((current?.revision ?? null) !== expected) fail('REVISION_CONFLICT', 'The target changed; retrieve its current revision before retrying', 409);
        return current;
    }
    async write(id, buffer, expectedRevision, { permission = 'edit', authorize = this.guard, internal = false } = {}) {
        return locked(this.lockKey, async () => {
            const p = await this.resolve(id, { create: true, internal });
            const current = await this.checkRevision(id, expectedRevision);
            if (current?.kind === 'directory') fail('INVALID_PATH', 'Cannot replace a directory');
            authorize(permission); if (current) authorize('replace');
            if (this.busy(id)) fail('BUSY', 'An active analysis owns this output', 409);
            const category = id.split('/')[0];
            if (!current && ['configs','images'].includes(category) && (await this.walk(category)).length >= 20) fail('QUOTA_EXCEEDED', 'File limit reached; no files were removed', 409);
            await noLinks(path.dirname(p), { missing: true }); await fs.mkdir(path.dirname(p), { recursive: true });
            const temporary = path.join(path.dirname(p), `.write-${crypto.randomUUID()}`);
            try {
                await fs.writeFile(temporary, buffer, { flag: 'wx', mode: 0o600 });
                await this.resolve(id, { create: true, internal });
                authorize(permission); if (current) authorize('replace');
                await fs.rename(temporary, p);
            } finally { await fs.rm(temporary, { force: true }); }
            return this.describe(id);
        });
    }
    async rename(id, targetId, expectedRevision) {
        return locked(this.lockKey, async () => {
            if (id.split('/')[0] !== targetId.split('/')[0]) fail('INVALID_PATH', 'Renaming must stay in the same category');
            const original = await this.checkRevision(id, expectedRevision); await this.checkRevision(targetId, null);
            if (!original) fail('NOT_FOUND', 'Target not found', 404);
            if (this.busy(id)) fail('BUSY', 'An active analysis owns this output', 409);
            const from = await this.resolve(id), to = await this.resolve(targetId, { create: true });
            await noLinks(path.dirname(to)); this.guard('edit');
            await fs.rename(from, to); return this.describe(targetId);
        });
    }
    async remove(id, expectedRevision) {
        return locked(this.lockKey, async () => {
            const meta = await this.checkRevision(id, expectedRevision);
            if (!meta) fail('NOT_FOUND', 'Target not found', 404);
            if (this.busy(id)) fail('BUSY', 'An active analysis owns this output', 409);
            const p = await this.resolve(id);
            if (meta.kind === 'directory') await this.walk(id); // Reject links anywhere in recursive target.
            this.guard('delete');
            await fs.rm(p, { recursive: meta.kind === 'directory', force: false });
            return { deleted: id, permanent: true, bytes: meta.size };
        });
    }
}
module.exports = { Workspace, WorkspaceError, fail, roots, userKey, segments, noLinks, locked, digest };
