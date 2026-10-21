const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const app = fs.readFileSync(require.resolve('../public/app.js'), 'utf8');
const ui = fs.readFileSync(require.resolve('../public/results-workspace.js'), 'utf8');
const html = fs.readFileSync(require.resolve('../public/index.html'), 'utf8');
const slice = (from, to) => app.slice(app.indexOf(from), app.indexOf(to));
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
const tick = () => new Promise(resolve => setImmediate(resolve));

class Element {
    constructor(tag = 'div') {
        this.tag = tag; this.children = []; this.dataset = {}; this.attributes = {};
        this.value = ''; this.textContent = ''; this.style = {}; this.events = {};
        this.isConnected = true; this.disabled = false;
        const classes = new Set();
        this.classList = { add: x => classes.add(x), remove: x => classes.delete(x),
            contains: x => classes.has(x), toggle(x, on = !classes.has(x)) { on ? classes.add(x) : classes.delete(x); } };
    }
    setAttribute(k, v) { this.attributes[k] = String(v); }
    getAttribute(k) { return this.attributes[k]; }
    appendChild(child) { this.children.push(child); return child; }
    get options() { return this.children; }
    set innerHTML(value) { this._html = value; this.children = []; if (this.tag === 'select') this.value = ''; }
    get innerHTML() { return this._html || ''; }
    querySelector(selector) {
        if (selector === '.plot-source') return this.label;
        if (selector === '.custom-select-option-main') return this.children.find(c => c.className === 'custom-select-option-main');
        return null;
    }
    querySelectorAll() { return this.children; }
    closest() { return this.content; }
    focus() { this.focused = true; }
    scrollIntoView() { this.scrolled = true; }
    addEventListener(type, fn) { (this.events[type] ||= []).push(fn); }
    dispatchEvent(event) { for (const fn of this.events[event.type] || []) fn(event); }
}

function fixture() {
    const nodes = {};
    for (const match of html.matchAll(/\bid="([^"]+)"/g)) nodes[match[1]] = new Element(match[1] === 'resultSelect' ? 'select' : 'div');
    const controls = new Element(); const palette = new Element();
    palette.dataset.plotType = 'b_magnitude';
    const plots = [];
    const state = {
        resultsData: {}, currentStep: 1, totalSteps: 1, userId: 'test', analysisConditions: null,
        resultLoadGeneration: 0, resultLoadController: null, dashboardRenderGeneration: 0,
        containerRenderControllers: new Map(), containerRenderTokens: new Map(),
    };
    const document = {
        getElementById: id => nodes[id] || null,
        createElement: tag => new Element(tag),
        querySelector: s => s === '.dashboard-controls' ? controls : s === '#dashboard-grid .grid-stack-item' ? plots[0] || null : null,
        querySelectorAll: s => s === '.palette-item' ? [palette] : s === '#dashboard-grid .plot-source' ? plots.map(p => p.label)
            : s === '#resultSelectOptions .custom-select-option' ? nodes.resultSelectOptions.children : [],
    };
    const counters = { cacheClears: 0, renders: 0, purges: 0 };
    const context = vm.createContext({
        AppState: state, document, console: { log() {}, warn() {}, error() {} }, AbortController, Event,
        Workbench: { schedule() {} }, switchTab: tab => { state.currentTab = tab; },
        updateCustomSelectDisplay() {}, pauseAnimation() {}, showStatus() {},
        invalidateDashboardRenders() { state.dashboardRenderGeneration++; },
        invalidateFieldLoads() {}, clearFieldDataCache() { counters.cacheClears++; }, clearCoarseningMaskCache() {},
        updateExportFormatBadge() {}, updatePolarControls() {},
        loadQuickPreviewFromResult: async () => {}, loadResultLog: async () => {},
        updateAllPlots: async () => { counters.renders++; },
        fetch: async url => ({ ok: true, json: async () => url.includes('detect-steps') ? { steps: 6 } : { coordinate_system: 'polar' } }),
        getCurrentResultPath: () => state.resultsData.currentResult || '',
        snapshotPolarView: () => ({ isPolar: false }),
        purgePlotlyTree: () => { counters.purges++; },
        Plotly: { newPlot: async () => ({}) },
    });
    vm.runInContext(ui + '\nthis.ResultsWorkspace = ResultsWorkspace;', context);
    vm.runInContext(slice('function createAbortError', 'async function loadResultLog'), context);
    function plot(type = 'b_magnitude') {
        const content = new Element(); content.dataset.plotType = type; content.label = new Element();
        const container = new Element(); container.content = content; container.id = 'container-test';
        nodes[container.id] = container; plots.push(content);
        const token = {}; container._activeRenderToken = token;
        return { content, container, scope: 'dashboard', token, containerId: container.id, step: 2,
            resultPath: 'outputs/test/result-A', analysisConditions: { coordinate_system: 'polar' },
            polarView: { isPolar: true, cartesianTransform: true, fullModel: false } };
    }
    return { context, state, nodes, controls, palette, plots, counters, plot, api: context.ResultsWorkspace };
}

test('empty Results offers a loader and disables field controls; adding a plot guides to the loader', () => {
    const f = fixture(); f.api.refresh();
    assert.match(f.nodes.resultLoadStatus.textContent, /No result loaded/);
    assert.equal(f.nodes.loadResultBtn.disabled, true);
    assert.equal(f.nodes.reloadResultBtn.disabled, true);
    assert.equal(f.controls.inert, true);
    assert.equal(f.nodes.resultsEmptyLoad.hidden, false);
    assert.equal(f.api.canAddPlot(), false);
    assert.equal(f.state.currentTab, 'dashboard');
    assert.equal(f.nodes.resultSelectTrigger.focused, true);
    f.api.listReady(0);
    assert.match(f.nodes.resultLoadStatus.textContent, /import a backup/);
    f.api.listFailed(new Error('offline'));
    assert.match(f.nodes.resultLoadStatus.textContent, /offline/);
});

test('selected result never masquerades as loaded data; filenames are text, not HTML', () => {
    const f = fixture();
    f.state.resultsData.currentResult = 'outputs/u/<img onerror=alert(1)>';
    f.state.analysisConditions = { coordinate_system: 'polar' };
    f.state.currentStep = 4; f.state.totalSteps = 12;
    f.nodes.resultSelect.value = 'outputs/u/B'; f.api.refresh();
    assert.equal(f.nodes.loadedResultName.textContent, '<img onerror=alert(1)>');
    assert.equal(f.nodes.loadedResultName.innerHTML, '');
    assert.match(f.nodes.resultSelectionHint.textContent, /Selected, not loaded: B/);
    assert.equal(f.nodes.loadedResultStep.textContent, '4 / 12');
    assert.equal(f.nodes.resultsEmptyAdd.hidden, false);
    assert.equal(f.controls.inert, false);
});

test('load notices are generation-safe and preserve an error beside the retained result', () => {
    const f = fixture(); const a = { resultPath: 'A' }, b = { resultPath: 'B' };
    f.api.beginLoad(a); f.api.beginLoad(b); f.api.loadFailed(a, new Error('old')); f.api.finishLoad(a);
    assert.match(f.nodes.resultLoadStatus.textContent, /Loading B/);
    f.state.resultsData.currentResult = 'A';
    f.api.loadFailed(b, new Error('offline')); f.api.finishLoad(b);
    assert.match(f.nodes.resultLoadStatus.textContent, /Could not load B: offline/);
    assert.match(f.nodes.resultLoadStatus.textContent, /Loaded result: A/);
    assert.equal(f.nodes.loadedResultName.textContent, 'A');
});

test('plot provenance is based on the completed render snapshot, and ignores stale tokens', () => {
    const f = fixture(), c = f.plot();
    f.state.currentStep = 9;
    f.api.plotStatus(c, 'loading');
    assert.equal(c.content.label.dataset.state, 'loading');
    f.api.plotStatus(c, 'ready');
    assert.match(c.content.label.textContent, /result-A · Step 2 · polar · arc view/);
    assert.doesNotMatch(c.content.label.textContent, /Step 9/);
    c.container._activeRenderToken = {};
    f.api.plotStatus(c, 'unavailable');
    assert.equal(c.content.label.dataset.state, 'ready');
    f.api.invalidatePlots();
    assert.equal(c.content.label.dataset.state, 'loading');
    const css = fs.readFileSync(require.resolve('../public/workspace.css'), 'utf8');
    assert.match(css, /\.plot-source\[data-state=loading\] \+ \.plot-container \{ visibility:hidden/);
});

test('timeline provenance counts finite rendered points, not the field step or detected step count', () => {
    const f = fixture(), c = f.plot('flux_linkage_time');
    f.state.totalSteps = 99;
    f.api.plotStatus(c, 'ready', [{ x: [1, 2, 3, 4], y: [0.1, null, 0.3, 0.4] }]);
    assert.match(c.content.label.textContent, /Timeline · up to 3 points\/series · steps 1–4/);
    assert.doesNotMatch(c.content.label.textContent, /99|Step 2/);
    f.api.plotStatus(c, 'ready', [{ x: [1, 2, 3, 4, 5], y: [1, 2, 3, 4, 5] }]);
    assert.match(c.content.label.textContent, /5 points\/series · steps 1–5/);
    f.api.plotStatus(c, 'unavailable');
    assert.match(c.content.label.textContent, /Unavailable \/ error/);
    assert.doesNotMatch(c.content.label.textContent, /5 points/);
});

test('reload targets the loaded result, refreshes cached fields and retains a valid current step', async () => {
    const f = fixture(); f.state.resultsData.currentResult = 'A'; f.state.currentStep = 4;
    f.nodes.resultSelect.value = 'B';
    await f.api.reload();
    assert.equal(f.nodes.resultSelect.value, 'A');
    assert.equal(f.state.resultsData.currentResult, 'A');
    assert.equal(f.state.currentStep, 4);
    assert.equal(f.counters.cacheClears, 1);
    assert.equal(f.counters.renders, 1);
    assert.equal(f.nodes.loadedResultStep.textContent, '4 / 6');
    assert.equal(f.nodes.resultLoadStatus.dataset.state, 'ready');
});

test('real load failure restores committed result, reports the error in Results and leaves controls usable', async () => {
    const f = fixture(); f.state.resultsData.currentResult = 'A'; f.nodes.resultSelect.value = 'B';
    f.context.fetch = async () => ({ ok: false });
    await f.context.loadSelectedResult();
    assert.equal(f.state.resultsData.currentResult, 'A');
    assert.equal(f.nodes.resultSelect.value, 'A');
    assert.match(f.nodes.resultLoadStatus.textContent, /Could not load B/);
    assert.equal(f.nodes.reloadResultBtn.disabled, false);
    assert.equal(f.controls.inert, false);
    assert.equal(f.counters.cacheClears, 0);
    assert.equal(f.counters.renders, 1);
});

test('overlapping real loads only commit the latest request and its status', async () => {
    const f = fixture(); const old = deferred();
    f.nodes.resultSelect.value = 'A';
    f.context.fetch = url => url.includes('result=A') ? old.promise
        : Promise.resolve({ ok: true, json: async () => url.includes('detect-steps') ? { steps: 8 } : { coordinate_system: 'cartesian' } });
    const a = f.context.loadSelectedResult();
    f.nodes.resultSelect.value = 'B'; await f.context.loadSelectedResult();
    old.resolve({ ok: true, json: async () => ({ steps: 20 }) }); await a;
    assert.equal(f.state.resultsData.currentResult, 'B');
    assert.equal(f.state.totalSteps, 8);
    assert.equal(f.nodes.loadedResultName.textContent, 'B');
    assert.equal(f.nodes.loadResultBtn.disabled, false);
    assert.equal(f.nodes.resultLoadStatus.dataset.state, 'ready');
    assert.equal(f.counters.renders, 1);
});

test('real Plotly commit publishes provenance only after completion and cannot publish an obsolete step', async () => {
    const f = fixture(); const c = f.plot();
    vm.runInContext(slice('function createDashboardRenderContext', 'function createFilePreviewRenderContext'), f.context);
    vm.runInContext(slice('function isRenderContextCurrent', 'function discardStaleRender'), f.context);
    f.state.resultsData.currentResult = 'A'; f.state.currentStep = 1;
    const context = f.context.createDashboardRenderContext(c.containerId, 'A', 1);
    const wait = deferred(); f.context.Plotly.newPlot = () => wait.promise;
    const render = f.context.newPlotForRender(c.container, [], {}, {}, context);
    assert.equal(c.content.label.dataset.state, 'loading');
    f.state.currentStep = 2; wait.resolve({});
    await assert.rejects(render, { name: 'AbortError' });
    assert.equal(c.content.label.dataset.state, 'loading');
    assert.equal(f.counters.purges, 1);
    const next = f.context.createDashboardRenderContext(c.containerId, 'A', 2);
    await f.context.newPlotForRender(c.container, [], {}, {}, next);
    assert.equal(c.content.label.dataset.state, 'ready');
    assert.match(c.content.label.textContent, /Step 2/);
});

test('list refresh preserves a mid-fetch choice and ignores stale responses; failures preserve options', async () => {
    const f = fixture();
    vm.runInContext(slice('let resultListGeneration', '// Initialize custom select event handlers'), f.context);
    const response = rows => ({ ok: true, json: async () => ({ results: rows }) });
    const a = deferred(), b = deferred(); let calls = 0;
    f.context.fetch = url => url.startsWith('/api/results') ? (++calls === 1 ? a.promise : b.promise)
        : Promise.resolve({ ok: true, json: async () => ({ description: '' }) });
    const old = f.context.refreshResultsList(), latest = f.context.refreshResultsList();
    f.nodes.resultSelect.value = 'B';
    b.resolve(response([{ path: 'B', name: 'B', steps: 5 }])); await latest;
    a.resolve(response([{ path: 'A', name: 'A', steps: 1 }])); await old;
    assert.equal(f.nodes.resultSelect.value, 'B');
    assert.equal(f.nodes.resultSelectDisplay.textContent, 'B (5 steps)');
    const before = f.nodes.resultSelectOptions.children;
    f.context.fetch = async () => { throw new Error('offline'); };
    await f.context.refreshResultsList();
    assert.equal(f.nodes.resultSelectOptions.children, before);
    assert.match(f.nodes.resultLoadStatus.textContent, /offline/);
});

test('removing the last plot restores the empty-state actions', () => {
    const f = fixture(); f.state.resultsData.currentResult = 'A'; f.plot(); f.api.refresh();
    assert.equal(f.nodes.emptyCanvas.hidden, true);
    f.plots.length = 0; f.api.refresh();
    assert.equal(f.nodes.emptyCanvas.hidden, false);
    assert.equal(f.nodes.resultsEmptyAdd.hidden, false);
});

test('palette supports click/Enter/Space while rejecting additions before a result loads', async () => {
    const f = fixture(); const canvas = new Element();
    const original = f.context.document.querySelector;
    f.context.document.querySelector = s => s === '#dashboard-grid' ? canvas : original(s);
    f.context.plotDefinitions = { b_magnitude: { name: '|B|' } };
    let added = 0;
    f.context.addPlotWidget = () => { if (f.api.canAddPlot()) added++; };
    vm.runInContext(slice('function setupPaletteDragDrop', '// Add plot widget to dashboard'), f.context);
    f.context.setupPaletteDragDrop();
    assert.equal(f.palette.tabIndex, 0);
    f.palette.dispatchEvent({ type: 'click' }); assert.equal(added, 0);
    f.state.resultsData.currentResult = 'A';
    for (const key of ['Enter', ' ']) f.palette.dispatchEvent({ type: 'keydown', key, preventDefault() {} });
    f.palette.dispatchEvent({ type: 'click' });
    assert.equal(added, 3);
    await tick();
});

test('single Results-owned loader, accessible controls and dependency order remain intact', () => {
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
    assert.equal(new Set(ids).size, ids.length);
    const start = html.indexOf('id="tab-dashboard"'), end = html.indexOf('id="tab-files"');
    for (const id of ['resultSelect', 'resultLoadStatus', 'polarControls', 'resultMetaBadge', 'emptyCanvas']) {
        assert.ok(html.indexOf(`id="${id}"`) > start && html.indexOf(`id="${id}"`) < end, id);
    }
    assert.ok(html.indexOf('src="results-workspace.js"') < html.indexOf('src="app.js"'));
    assert.match(html, /id="resultSelectTrigger"[^>]+aria-haspopup="listbox"/);
    assert.match(html, /id="resultLoadStatus" role="status" aria-live="polite"/);
});

test('result picker keyboard navigation selects without implicitly loading and closes on Escape', () => {
    const f = fixture(); const { resultSelectContainer: container, resultSelectTrigger: trigger,
        resultSelectOptions: options, resultSelect: select } = f.nodes;
    const listeners = {};
    f.context.document.addEventListener = (type, fn) => { (listeners[type] ||= []).push(fn); };
    container.contains = target => [container, trigger, options, ...options.children].includes(target);
    for (const value of ['', 'A', 'B']) {
        const option = new Element(); option.setAttribute('data-value', value);
        option.closest = () => option; options.appendChild(option);
    }
    vm.runInContext(slice('function initCustomSelect', 'function createAbortError'), f.context);
    f.context.initCustomSelect();
    trigger.dispatchEvent({ type: 'keydown', key: 'ArrowDown', preventDefault() {} });
    assert.equal(trigger.getAttribute('aria-expanded'), 'true');
    assert.equal(options.children[0].focused, true);
    options.dispatchEvent({ type: 'keydown', key: 'End', target: options.children[0], preventDefault() {} });
    assert.equal(options.children[2].focused, true);
    options.dispatchEvent({ type: 'keydown', key: 'Enter', target: options.children[2], preventDefault() {} });
    assert.equal(select.value, 'B');
    assert.equal(trigger.getAttribute('aria-expanded'), 'false');
    assert.equal(f.state.resultsData.currentResult, undefined);
    trigger.dispatchEvent({ type: 'click', stopPropagation() {} });
    for (const fn of listeners.keydown) fn({ key: 'Escape' });
    assert.equal(trigger.getAttribute('aria-expanded'), 'false');
    assert.equal(trigger.focused, true);
});

test('missing data uses the real message lifecycle and never remains labelled as plotted data', () => {
    const f = fixture(), c = f.plot();
    vm.runInContext(slice('function preparePlotlyContainer', 'function invalidateDashboardRenders'), f.context);
    f.context.assertRenderContextCurrent = () => {};
    f.api.plotStatus(c, 'ready');
    f.context.showPlotMessage(c.container, '<p>No data available</p>', c);
    assert.equal(c.content.label.dataset.state, 'unavailable');
    assert.equal(c.container.getAttribute('aria-busy'), 'false');
    assert.match(c.container.innerHTML, /No data available/);
});

test('overlapping Plotly updates serialize commits and skip superseded pending steps',async()=>{
    const f=fixture(),c=f.plot();
    vm.runInContext(slice('function createDashboardRenderContext','function createFilePreviewRenderContext'),f.context);
    vm.runInContext(slice('function isRenderContextCurrent','function discardStaleRender'),f.context);
    f.state.resultsData.currentResult='A';f.state.currentStep=1;
    const wait=deferred(),calls=[];
    c.container._fullLayout={};c.container._preservePlot=true;
    f.context.Plotly.react=(container,traces)=>{calls.push(traces[0].step);return calls.length===1?wait.promise:Promise.resolve({});};
    const render=step=>{f.state.currentStep=step;const ctx=f.context.createDashboardRenderContext(c.containerId,'A',step);return f.context.newPlotForRender(c.container,[{step}],{},{},ctx);};
    const first=render(1),second=render(2),third=render(3);
    const firstRejected=assert.rejects(first,{name:'AbortError'}),secondRejected=assert.rejects(second,{name:'AbortError'});
    assert.deepEqual(calls,[1]);wait.resolve({});
    await Promise.all([firstRejected,secondRejected,third]);
    assert.deepEqual(calls,[1,3]);assert.equal(f.counters.purges,0);
    assert.match(c.content.label.textContent,/Step 3/);assert.equal(c.container._plotlyPending,undefined);
});
