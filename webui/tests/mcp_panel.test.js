const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require.resolve('../public/mcp-panel.js'), 'utf8');

class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.events = {}; this.textContent = ''; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(type, fn) { this.events[type] = fn; }
}
function fixture(available = true) {
    const ids = Object.fromEntries(['mcpActivity', 'mcpMessage', 'mcpExternalChanges', 'resultSelect']
        .map(id => [id, new Element('div')]));
    ids.resultSelect.options = available ? [{ value: 'outputs/user_A/output_mcp_test' }] : [];
    ids.resultSelect.value = 'outputs/user_A/old-result';
    const calls = [];
    let displayed = 'old-result';
    const context = vm.createContext({
        document: { getElementById: id => ids[id], createElement: tag => new Element(tag), addEventListener() {} },
        AppState: { userId: 'user_A' },
        fetch: async url => {
            assert.equal(url, '/api/automation/activity?userId=user_A');
            return { ok: true, json: async () => ({ jobs: [{ source: 'test-agent', status: 'completed', progress: 100,
                resultId: 'results/output_mcp_test', jobId: 'test-job' }], events: [] }) };
        },
        refreshResultsList: async () => { calls.push('refresh'); },
        updateCustomSelectDisplay: value => { displayed = value; calls.push('display'); },
        switchTab: name => { calls.push(name); },
        loadSelectedResult: async () => {
            assert.equal(ids.resultSelect.value, 'outputs/user_A/output_mcp_test');
            assert.equal(displayed, ids.resultSelect.value);
            calls.push('load');
        },
    });
    vm.runInContext(source + '\nthis.panel = MCPPanel;', context);
    return { ids, calls, context };
}
function findButton(node, text) {
    if (node.tag === 'button' && node.textContent === text) return node;
    for (const child of node.children || []) { const match = findButton(child, text); if (match) return match; }
}
test('MCP activity summarizes finished jobs and routes to the dedicated manager', async () => {
    const f = fixture();
    await f.context.panel.activity();
    const button = findButton(f.ids.mcpActivity, 'Open job manager');
    assert.ok(button);
    await button.events.click();
    assert.deepEqual(f.calls, ['jobs']);
    assert.equal(findButton(f.ids.mcpActivity, 'Open Results'), undefined);
    assert.equal(f.ids.mcpMessage.textContent, '');
    assert.equal(button.disabled, false);
});
test('MCP activity keeps file events collapsed and leaves the current result untouched', async () => {
    const f = fixture(false);
    await f.context.panel.activity();
    const details=f.ids.mcpActivity.children.find(n=>n.tag==='details');
    assert.ok(details);assert.ok(!details.open);
    assert.deepEqual(f.calls, []);
    assert.equal(f.ids.resultSelect.value, 'outputs/user_A/old-result');
});
