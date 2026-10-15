'use strict';
// GUI adapters for the same revision-checked operations used by MCP.
// The legacy GUI workspace selector is not an authentication mechanism.
const {Workspace, fail} = require('./workspace');
function installFileRoutes(app, base, busy) {
    const workspace = req => {
        const user = req.body?.userId || req.query.userId || 'default';
        return new Workspace(base, user, {busy: id => busy(user, id)});
    };
    const revision = req => {
        const value = req.body?.revision ?? req.headers['if-match'];
        if (typeof value !== 'string' || !/^[a-f0-9]{64}$/.test(value))
            fail('PRECONDITION_REQUIRED', 'Retrieve the file revision before changing or deleting it', 428);
        return value;
    };
    const route = fn => async (req, res) => {
        res.set('Cache-Control', 'no-store');
        try { res.json({success: true, ...await fn(req, workspace(req))}); }
        catch (e) { res.status(e.statusCode || (e.code === 'ENOENT' ? 404 : 500)).json({success: false, code: e.code || 'OPERATION_FAILED', error: e.statusCode ? e.message : 'File operation failed'}); }
    };
    app.get('/api/workspace/file', route(async (req, ws) => {
        try { return {file: await ws.describe(req.query.id)}; }
        catch (e) { if (e.code === 'ENOENT') return {file: null}; throw e; }
    }));
    app.delete('/api/user-files', route((req, ws) => ws.remove(`${req.body.category}/${req.body.name}`, revision(req))));
    app.delete('/api/user-outputs/bulk', route(async (req, ws) => {
        const files = req.body.files;
        if (!Array.isArray(files) || !files.length || files.length > 100)
            fail('PRECONDITION_REQUIRED', 'Supply 1..100 result names and their revisions', 428);
        const results = [];
        for (const file of files) {
            try { await ws.remove(`results/${file.name}`, file.revision); results.push({name: file.name, success: true}); }
            catch (e) { results.push({name: file.name, success: false, error: e.statusCode ? e.message : 'Deletion failed'}); }
        }
        return {results};
    }));
    for (const [url, category, param] of [
        ['/api/images/:filename', 'images', 'filename'],
        ['/api/config/:filename', 'configs', 'filename'],
        ['/api/material-libraries/:filename', 'libraries', 'filename'],
        ['/api/user-outputs/:folderName', 'results', 'folderName']
    ]) app.delete(url, route((req, ws) => ws.remove(`${category}/${req.params[param]}`, revision(req))));
    app.put('/api/user-outputs/:folderName/rename', route(async (req, ws) => {
        if (typeof req.body.newName !== 'string' || req.body.newName.includes('/') || req.body.newName.length > 100)
            fail('INVALID_INPUT', 'Invalid result folder name');
        const file = await ws.rename(`results/${req.params.folderName}`, `results/${req.body.newName}`, revision(req));
        return {newName: file.name, revision: file.revision};
    }));
    app.get('/api/user-outputs/:folderName/description', route(async (req, ws) => {
        const id = `results/${req.params.folderName}/description.txt`;
        await ws.resolve(`results/${req.params.folderName}`);
        try {
            const value = await ws.read(id, {text: true, length: 32768});
            if (value.nextOffset !== null) fail('LIMIT_EXCEEDED', 'Description is too large');
            return {description: value.content, revision: value.revision};
        } catch (e) { if (e.code === 'ENOENT') return {description: '', revision: null}; throw e; }
    }));
    app.put('/api/user-outputs/:folderName/description', route(async (req, ws) => {
        const {description, expectedRevision} = req.body;
        if (typeof description !== 'string' || Buffer.byteLength(description) > 32768)
            fail('INVALID_INPUT', 'Description must be at most 32 KiB');
        if (expectedRevision === undefined) fail('PRECONDITION_REQUIRED', 'Load the description before editing', 428);
        await ws.resolve(`results/${req.params.folderName}`);
        return ws.write(`results/${req.params.folderName}/description.txt`, Buffer.from(description), expectedRevision);
    }));
}
module.exports = {installFileRoutes};
