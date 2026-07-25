const assert = require('node:assert/strict');
const test = require('node:test');
const { EventEmitter, once } = require('node:events');
const http = require('node:http');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs').promises;
const path = require('node:path');

process.env.OPENMAGFDM_NO_LISTEN = '1';
process.env.OPENMAGFDM_SKIP_INIT = '1';
process.env.PORT = '0';

const server = require('../server');
const lifecycle = server._test;

function fakeChild({ closeOnTerm = true, closeOnKill = true } = {}) {
    const child = new EventEmitter();
    child.pid = 12345;
    child.exitCode = null;
    child.signalCode = null;
    child.killCalls = [];
    child.stdout = { pause() {}, resume() {} };
    child.stderr = { pause() {}, resume() {} };
    child.kill = (signal) => {
        child.killCalls.push(signal);
        const shouldClose = signal === 'SIGTERM' ? closeOnTerm : closeOnKill;
        if (shouldClose) {
            setImmediate(() => {
                if (child.signalCode !== null) return;
                child.signalCode = signal;
                child.emit('close', null, signal);
            });
        }
        return true;
    };
    return child;
}

function postJson(httpServer, pathname, payload) {
    const body = JSON.stringify(payload);
    const address = httpServer.address();
    return new Promise((resolve, reject) => {
        const request = http.request({
            host: '127.0.0.1',
            port: address.port,
            path: pathname,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(body)
            }
        }, (response) => {
            let responseBody = '';
            response.setEncoding('utf8');
            response.on('data', chunk => { responseBody += chunk; });
            response.on('end', () => resolve({
                statusCode: response.statusCode,
                body: JSON.parse(responseBody)
            }));
        });
        request.once('error', reject);
        request.end(body);
    });
}

test('per-user claims reject duplicates and only identity-matched deletes succeed', () => {
    lifecycle.runningProcesses.clear();
    const first = fakeChild();
    const second = fakeChild();

    assert.equal(lifecycle.claimRunningProcess('alice', first, 'legacy'), true);
    assert.equal(lifecycle.claimRunningProcess('alice', second, 'stream'), false);
    assert.equal(lifecycle.deleteRunningProcessIfSame('alice', second), false);
    assert.equal(lifecycle.runningProcesses.get('alice').process, first);
    assert.equal(lifecycle.deleteRunningProcessIfSame('alice', first), true);
});

test('solver preparation and legacy response retention are synchronously bounded', () => {
    const finishPreparation = lifecycle.beginSolverStartOperation('preparing-user');
    assert.throws(
        () => lifecycle.beginSolverStartOperation('preparing-user'),
        error => error.code === 'USER_SOLVER_START_LIMIT' && error.statusCode === 409
    );
    finishPreparation();
    assert.equal(lifecycle.activeSolverStartOperations.size, 0);
    assert.equal(lifecycle.activeSolverStartOperationsByUser.size, 0);

    const response = new EventEmitter();
    const initialResponses = lifecycle.activeLegacyResponses.size;
    lifecycle.claimLegacyResponsePermit(response);
    assert.equal(lifecycle.activeLegacyResponses.size, initialResponses + 1);
    response.emit('finish');
    assert.equal(lifecycle.activeLegacyResponses.size, initialResponses);

    const bounded = lifecycle.boundedLegacyResponseText('x'.repeat(1024 * 1024));
    assert.ok(bounded.length < 1024 * 1024);
    assert.match(bounded, /earlier solver output omitted/);
});

test('job logs and retained job records remain bounded', () => {
    let log = [];
    for (const line of ['one', 'two', 'three', 'four']) {
        log = lifecycle.appendCappedLog(log, line, 3, 100);
    }
    assert.deepEqual(log, ['two', 'three', 'four']);
    assert.equal(lifecycle.appendCappedLog([], 'x'.repeat(50), 3, 20)[0].length, 20);
    log = lifecycle.appendCappedLog(['1234567890'], 'abcdefghij', 10, 100, 15);
    assert.deepEqual(log, ['abcdefghij']);

    const now = Date.now();
    const active = fakeChild();
    const records = new Map([
        ['expired', { finished: new Date(now - 10_000).toISOString(), process: null }],
        ['old', { finished: new Date(now - 900).toISOString(), process: null }],
        ['new', { finished: new Date(now - 100).toISOString(), process: null }],
        ['active', { finished: null, process: active }]
    ]);

    lifecycle.pruneJobRecords(records, now, 1_000, 2);
    assert.deepEqual(Array.from(records.keys()).sort(), ['active', 'new']);
});

test('output reservations are unique and unused reservations are removable', async () => {
    const userId = `lifecycle-output-${process.pid}`;
    const first = await lifecycle.prepareUserOutputDirectory(userId);
    const second = await lifecycle.prepareUserOutputDirectory(userId);
    assert.notEqual(first, second);

    await lifecycle.removeEmptyOutputDirectory(first);
    await lifecycle.removeEmptyOutputDirectory(second);
    await assert.rejects(fs.access(first), error => error.code === 'ENOENT');
    await assert.rejects(fs.access(second), error => error.code === 'ENOENT');
    await fs.rmdir(path.dirname(first)).catch(error => {
        if (error.code !== 'ENOENT' && error.code !== 'ENOTEMPTY') throw error;
    });
});

test('termination escalates from SIGTERM to SIGKILL and observes close', async () => {
    const child = fakeChild({ closeOnTerm: false, closeOnKill: true });
    const result = await lifecycle.terminateChildProcess(child, { graceMs: 5, forceWaitMs: 100 });

    assert.deepEqual(child.killCalls, ['SIGTERM', 'SIGKILL']);
    assert.deepEqual(result, { exited: true, forced: true });
});

test('premature response close terminates its solver child', async () => {
    const response = new EventEmitter();
    response.writableEnded = false;
    response.destroyed = false;
    const child = fakeChild();
    const childClosed = once(child, 'close');

    lifecycle.terminateOnPrematureResponseClose(response, child, 'test stream');
    response.emit('close');
    await childClosed;

    assert.deepEqual(child.killCalls, ['SIGTERM']);
});

test('SSE backpressure resumes paused pipes when child exit precedes drain', () => {
    const response = new EventEmitter();
    response.destroyed = false;
    const child = fakeChild();
    let pauseCalls = 0;
    let resumeCalls = 0;
    child.stdout = {
        pause() { pauseCalls++; },
        resume() { resumeCalls++; }
    };
    child.stderr = {
        pause() { pauseCalls++; },
        resume() { resumeCalls++; }
    };
    const controller = lifecycle.createSseBackpressureController(response, child, 1000);

    controller.pause();
    assert.equal(controller.isPaused(), true);
    child.exitCode = 0;
    child.emit('exit', 0, null);

    assert.equal(controller.isPaused(), false);
    assert.equal(controller.isWaitingForDrain(), true);
    assert.equal(pauseCalls, 2);
    assert.equal(resumeCalls, 2);
    response.emit('close');
    assert.equal(controller.isWaitingForDrain(), false);
});

test('SSE backpressure timeout destroys a stalled response and terminates the child', async () => {
    const response = new EventEmitter();
    response.destroyed = false;
    response.destroy = () => {
        response.destroyed = true;
        response.emit('close');
    };
    const child = fakeChild();
    const childClosed = once(child, 'close');
    const controller = lifecycle.createSseBackpressureController(response, child, 5);

    controller.pause();
    await childClosed;

    assert.equal(response.destroyed, true);
    assert.equal(controller.isPaused(), false);
    assert.equal(controller.isWaitingForDrain(), false);
    assert.deepEqual(child.killCalls, ['SIGTERM']);
});

test('interactive and job endpoints reject cross-mode overlap for one user', async (t) => {
    lifecycle.jobs.clear();
    lifecycle.runningProcesses.clear();
    const httpServer = server.startServer();
    if (!httpServer.listening) await once(httpServer, 'listening');
    t.after(async () => {
        lifecycle.jobs.clear();
        lifecycle.runningProcesses.clear();
        if (httpServer.listening) {
            await new Promise(resolve => httpServer.close(resolve));
        }
    });

    const jobChild = fakeChild();
    lifecycle.jobs.set('active-job', {
        userId: 'shared-user',
        process: jobChild,
        finished: null
    });
    const interactiveResponse = await postJson(httpServer, '/api/solve', {
        userId: 'shared-user'
    });
    assert.equal(interactiveResponse.statusCode, 409);

    lifecycle.jobs.clear();
    const interactiveChild = fakeChild();
    lifecycle.claimRunningProcess('shared-user', interactiveChild, 'stream');
    const jobResponse = await postJson(httpServer, '/api/jobs', {
        userId: 'shared-user',
        configFile: 'config.yaml',
        imageFile: 'image.png'
    });
    assert.equal(jobResponse.statusCode, 409);
});

test('shutdown terminates and unregisters all tracked solver children', async () => {
    lifecycle.runningProcesses.clear();
    lifecycle.trackedSolverProcesses.clear();
    const child = fakeChild();
    lifecycle.registerTrackedSolverProcess(child, { purpose: 'stream', userId: 'shutdown-user' });
    lifecycle.claimRunningProcess('shutdown-user', child, 'stream');
    const finishStartOperation = lifecycle.beginSolverStartOperation();
    // Model Node's normal exit -> stdio close window. Shutdown must wait for
    // close-based registry/finalizer cleanup even though exitCode is already set.
    child.exitCode = 0;
    let closeEmitted = false;
    let startOperationFinished = false;
    setTimeout(() => {
        closeEmitted = true;
        child.emit('close', 0, null);
    }, 10);
    setTimeout(() => {
        startOperationFinished = true;
        finishStartOperation();
    }, 20);

    const result = await server.shutdownServer('lifecycle-test');

    assert.equal(result.clean, true);
    assert.equal(result.timedOut, false);
    assert.equal(result.requestedChildren, 1);
    assert.equal(closeEmitted, true);
    assert.equal(startOperationFinished, true);
    assert.equal(result.remainingStartOperations, 0);
    assert.equal(lifecycle.trackedSolverProcesses.size, 0);
    assert.equal(lifecycle.runningProcesses.has('shutdown-user'), false);
});

test('shutdown reports a child that survives every termination attempt', () => {
    const serverPath = require.resolve('../server');
    const script = `
        const { EventEmitter } = require('node:events');
        const server = require(${JSON.stringify(serverPath)});
        const child = new EventEmitter();
        child.pid = 99999;
        child.exitCode = null;
        child.signalCode = null;
        child.stdout = { pause() {}, resume() {} };
        child.stderr = { pause() {}, resume() {} };
        child.kill = () => true;
        server._test.registerTrackedSolverProcess(child, { purpose: 'job' });
        server.shutdownServer('survivor-test').then(result => {
            console.log(JSON.stringify(result));
            server._test.trackedSolverProcesses.clear();
        });
    `;
    const result = spawnSync(process.execPath, ['-e', script], {
        encoding: 'utf8',
        env: {
            ...process.env,
            OPENMAGFDM_NO_LISTEN: '1',
            OPENMAGFDM_SKIP_INIT: '1',
            CHILD_TERMINATION_GRACE_MS: '5',
            CHILD_FORCE_WAIT_MS: '5',
            SHUTDOWN_TIMEOUT_MS: '30',
            FINAL_SHUTDOWN_FORCE_WAIT_MS: '5'
        }
    });

    assert.equal(result.status, 0, result.stderr);
    const summary = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1));
    assert.equal(summary.clean, false);
    assert.equal(summary.timedOut, true);
    assert.equal(summary.remainingRunningChildren, 1);
});

test('version lookup is single-flight and then served from cache', () => {
    const serverPath = require.resolve('../server');
    const script = `
        const server = require(${JSON.stringify(serverPath)});
        const first = server._test.getSolverVersion();
        const second = server._test.getSolverVersion();
        const during = server._test.trackedSolverProcesses.size;
        Promise.all([first, second]).then(async versions => {
            const afterProbe = server._test.trackedSolverProcesses.size;
            const cached = await server._test.getSolverVersion();
            console.log(JSON.stringify({ versions, cached, during, afterProbe,
                afterCache: server._test.trackedSolverProcesses.size }));
        });
    `;
    const result = spawnSync(process.execPath, ['-e', script], {
        encoding: 'utf8',
        env: {
            ...process.env,
            OPENMAGFDM_NO_LISTEN: '1',
            OPENMAGFDM_SKIP_INIT: '1',
            SOLVER_PATH: process.execPath
        }
    });

    assert.equal(result.status, 0, result.stderr);
    const summary = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1));
    assert.equal(summary.during, 1);
    assert.equal(summary.afterProbe, 0);
    assert.equal(summary.afterCache, 0);
    assert.deepEqual(summary.versions, [summary.cached, summary.cached]);
    assert.match(summary.cached, /^\d+\.\d+\.\d+$/);
});
