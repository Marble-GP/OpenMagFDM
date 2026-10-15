'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const { execFileSync } = require('node:child_process');
const { fail, userKey, noLinks, locked } = require('./workspace');
const PERMISSIONS = Object.freeze(['read', 'edit', 'images', 'run', 'cancel', 'delete', 'replace']);
const hash = text => crypto.createHash('sha256').update(text).digest('hex');
const random = () => crypto.randomBytes(32).toString('base64url');
function privateDirectory(base) {
    const home = process.platform === 'win32' ? process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local') : path.join(os.homedir(), '.local', 'share');
    return path.join(home, 'OpenMagFDM', 'automation', hash(path.resolve(base)).slice(0, 20));
}
async function restrict(directory) {
    await noLinks(directory, { missing: true });
    await fs.mkdir(directory, { recursive: true, mode: 0o700 });
    if (process.platform === 'win32') {
        const identity = execFileSync('whoami.exe', ['/user', '/fo', 'csv', '/nh'], { encoding: 'utf8', windowsHide: true });
        const sid = identity.match(/S-1-5-[0-9-]+/);
        if (!sid) fail('PRIVATE_STORAGE', 'Cannot determine the current Windows user');
        execFileSync('icacls.exe', [directory, '/inheritance:r', '/grant:r', `*${sid[0]}:(OI)(CI)F`, '*S-1-5-18:(OI)(CI)F'], { windowsHide: true, stdio: 'pipe' });
    } else await fs.chmod(directory, 0o700);
}
async function atomicJSON(filename, data) {
    await noLinks(filename, { missing: true });
    const temp = `${filename}.${crypto.randomUUID()}.tmp`;
    try { await fs.writeFile(temp, JSON.stringify(data, null, 2), { flag: 'wx', mode: 0o600 }); await fs.rename(temp, filename); }
    finally { await fs.rm(temp, { force: true }); }
}
function localRequest(req) {
    const remote = req.socket?.remoteAddress;
    if (!['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(remote)) fail('LOCAL_ONLY', 'Local connections only', 403);
    let host;
    try { host = new URL(`http://${req.headers.host}`); } catch { fail('INVALID_HOST', 'Invalid Host', 403); }
    if (!['localhost', '127.0.0.1', '[::1]'].includes(host.hostname) || host.username || host.password || host.pathname !== '/') fail('INVALID_HOST', 'Invalid Host', 403);
    if (req.headers.origin) {
        let origin;
        try { origin = new URL(req.headers.origin); } catch { fail('INVALID_ORIGIN', 'Invalid Origin', 403); }
        if (origin.origin !== `http://${req.headers.host}` || req.headers.origin !== origin.origin) fail('INVALID_ORIGIN', 'Cross-origin access denied', 403);
    }
    if (req.headers['sec-fetch-site'] && !['same-origin', 'none'].includes(req.headers['sec-fetch-site'])) fail('INVALID_ORIGIN', 'Cross-site access denied', 403);
}
class Access {
    constructor(base, { directory = privateDirectory(base) } = {}) {
        this.base = path.resolve(base); this.directory = directory;
        this.filename = path.join(directory, 'settings.json');
        this.state = { version: 1, enabled: false, port: 3001, connections: [] };
        this.sessions = new Map(); this.pairingCode = random(); this.failures = 0; this.blockedUntil = 0;
        this.events = []; this.ready = false; this.auditWrites = new Set(); this.auditDropped = 0; this.auditError = null;
    }
    async init() {
        const relative = path.relative(this.base, path.resolve(this.directory));
        if (!relative || (!relative.startsWith('..') && !path.isAbsolute(relative))) fail('PRIVATE_STORAGE', 'Credential storage must be outside the application data root');
        await restrict(this.directory);
        try {
            await noLinks(this.filename);
            const state = JSON.parse(await fs.readFile(this.filename, 'utf8'));
            if (state.version !== 1 || typeof state.enabled !== 'boolean' || !Number.isInteger(state.port) || state.port < 1024 || state.port > 65535 || !Array.isArray(state.connections)) throw Error('Invalid settings');
            if (state.connections.length > 100) throw Error('Invalid connection count');
            const ids = new Set();
            for (const c of state.connections) {
                c.user = userKey(c.user); this.permissions(c.permissions);
                if (!/^[a-f0-9-]{36}$/.test(c.id) || ids.has(c.id) || !/^[a-f0-9]{64}$/.test(c.tokenHash) || typeof c.revoked !== 'boolean') throw Error('Invalid connection');
                ids.add(c.id);
            }
            this.state = state;
        } catch (e) { if (e.code !== 'ENOENT') throw e; await atomicJSON(this.filename, this.state); }
        await atomicJSON(path.join(this.directory, 'pairing.json'), { code: this.pairingCode });
        this.ready = true;
    }
    permissions(values) {
        if (!Array.isArray(values) || values.some(p => !PERMISSIONS.includes(p))) fail('INVALID_PERMISSION', 'Unknown permission');
        return [...new Set(['read', ...values])];
    }
    async change(fn) {
        return locked(this.filename, async () => {
            const next = structuredClone(this.state);
            const result = await fn(next);
            await atomicJSON(this.filename, next);
            this.state = next;
            return result;
        });
    }
    summary() {
        return { enabled: this.state.enabled, port: this.state.port, permissions: PERMISSIONS, auditDropped: this.auditDropped, auditError: this.auditError,
            connections: this.state.connections.map(({ tokenHash, ...c }) => c) };
    }
    async create({ name, user, permissions = ['read'] }) {
        user = userKey(user);
        if (typeof name !== 'string' || name.length < 1 || name.length > 80) fail('INVALID_INPUT', 'Connection name must be 1..80 characters');
        const token = random(); const connection = { id: crypto.randomUUID(), name, user, permissions: this.permissions(permissions), tokenHash: hash(token), created: new Date().toISOString(), lastAccess: null, revoked: false };
        await this.change(async next => {
            if (next.connections.length >= 100) fail('LIMIT_EXCEEDED', 'At most 100 connection profiles are allowed');
            // The bridge reads this OS-private profile. It is never served over HTTP.
            await atomicJSON(path.join(this.directory, `${connection.id}.json`), { token, id: connection.id });
            next.connections.push(connection);
        });
        return { id: connection.id, token };
    }
    async update(id, changes) {
        return this.change(next => {
            const c = next.connections.find(c => c.id === id);
            if (!c) fail('NOT_FOUND', 'Connection not found', 404);
            if (changes.permissions) c.permissions = this.permissions(changes.permissions);
            if (changes.revoked === true) c.revoked = true;
            return { id: c.id, revoked: c.revoked };
        });
    }
    authenticate(header) {
        if (!this.ready || !this.state.enabled) fail('MCP_DISABLED', 'MCP is disabled in the WebUI', 503);
        const token = /^Bearer ([A-Za-z0-9_-]{43})$/.exec(header || '')?.[1];
        if (!token) fail('UNAUTHORIZED', 'A valid connection token is required', 401);
        const h = hash(token);
        const c = this.state.connections.find(c => c.tokenHash.length === h.length && crypto.timingSafeEqual(Buffer.from(c.tokenHash), Buffer.from(h)));
        if (!c || c.revoked) fail('UNAUTHORIZED', 'Connection is not authorized', 401);
        c.lastAccess = new Date().toISOString();
        return this.actor(c.id);
    }
    actor(id) {
        const c = this.state.connections.find(c => c.id === id);
        if (!c || c.revoked) fail('UNAUTHORIZED', 'Connection is not authorized', 401);
        const actor = { id: c.id, user: c.user, name: c.name };
        actor.guard = permission => {
            const current = this.state.connections.find(c => c.id === id);
            if (!this.ready || !this.state.enabled || !current || current.revoked || current.user !== actor.user) fail('UNAUTHORIZED', 'Connection disabled or revoked', 403);
            if (!current.permissions.includes(permission)) fail('FORBIDDEN', `Permission required: ${permission}`, 403);
        };
        return actor;
    }
    pair(code) {
        if (Date.now() < this.blockedUntil) fail('RATE_LIMITED', 'Wait before retrying pairing', 429);
        if (!this.pairingCode || typeof code !== 'string' || hash(code) !== hash(this.pairingCode)) {
            if (++this.failures >= 5) { this.blockedUntil = Date.now() + 60000; this.failures = 0; }
            fail('UNAUTHORIZED', 'Incorrect pairing code', 401);
        }
        this.pairingCode = random(); // One use; --mcp-admin-code can retrieve the replacement privately.
        const filename = path.join(this.directory, 'pairing.json'), codeToSave = this.pairingCode;
        this.pairingWrite = locked(filename, () => atomicJSON(filename, {code: codeToSave})).catch(() => {this.auditError = 'Pairing code persistence failed';});
        const token = random(), csrf = random();
        for (const [key, value] of this.sessions) if (value.expires < Date.now()) this.sessions.delete(key);
        if (this.sessions.size >= 16) this.sessions.delete(this.sessions.keys().next().value);
        this.sessions.set(hash(token), { csrf, expires: Date.now() + 24 * 3600 * 1000 });
        return { token, csrf };
    }
    admin(req, mutate = false) {
        localRequest(req);
        const token = /(?:^|;\s*)omfdm_admin=([A-Za-z0-9_-]+)/.exec(req.headers.cookie || '')?.[1];
        const session = token && this.sessions.get(hash(token));
        if (!session || session.expires < Date.now()) fail('ADMIN_REQUIRED', 'Pair this browser using the server confirmation code', 401);
        if (mutate && req.headers['x-omfdm-csrf'] !== session.csrf) fail('CSRF', 'Invalid management request', 403);
        return session;
    }
    audit(actor, operation, target, code = 'OK') {
        const event = { at: new Date().toISOString(), client: actor.name, user: actor.user, operation, target, code };
        this.events.push(event); if (this.events.length > 500) this.events.shift();
        // A bounded on-disk history: one current file plus one rotation, no payloads or credentials.
        if (this.auditWrites.size >= 128) { this.auditDropped++; return; }
        const writing = locked(`${this.directory}:audit`, async () => {
            const filename = path.join(this.directory, 'audit.jsonl');
            await noLinks(filename, { missing: true });
            const stat = await fs.stat(filename).catch(() => null);
            if (stat?.size > 1024 * 1024) {
                const old = `${filename}.1`; await noLinks(old, { missing: true });
                await fs.rm(old, { force: true }); await fs.rename(filename, old);
            }
            await fs.appendFile(filename, JSON.stringify(event) + '\n', { mode: 0o600 });
        }).catch(() => { this.auditError = 'Audit file write failed'; });
        this.auditWrites.add(writing);
        void writing.finally(() => this.auditWrites.delete(writing));
    }
    async flush() { await Promise.allSettled([...this.auditWrites, this.pairingWrite]); }
}
module.exports = { Access, PERMISSIONS, privateDirectory, atomicJSON, localRequest, hash };
