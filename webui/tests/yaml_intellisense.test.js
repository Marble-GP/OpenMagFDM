const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const yaml = require('js-yaml');
const schema = require('../public/yaml-schema.json');
const api = require('../public/yaml-intellisense');
const check = text => api.analyze(text, schema, yaml);
function completion(text) {
    const offset = text.indexOf('|CURSOR|');
    assert.ok(offset >= 0);
    const before = text.slice(0, offset).split('\n');
    return api.complete(text.replace('|CURSOR|', ''), {row: before.length - 1, column: before.at(-1).length}, schema, yaml);
}
const captions = text => completion(text).items.map(i => i.caption);

test('root candidates exclude leaf properties and hidden experiments', () => {
    const names = captions('|CURSOR|');
    assert.ok(names.includes('materials'));
    for (const key of ['r_start', 'mu_r', 'enabled', 'jacobian', 'dx']) assert.ok(!names.includes(key));
    assert.ok(!captions('nonlinear_solver:\n  |CURSOR|').includes('jacobian'));
});
test('known keys in the wrong hierarchy are warned', () => {
    assert.equal(check('omp:\n  enabled: true')[0].row, 1);
    assert.equal(check('mu_r: 1')[0].type, 'warning');
    assert.deepEqual(check('nonlinear_solver:\n  jacobian: tangent\n  line_search_objective: energy'), []);
});
test('mesh and rectangle spacing have different descriptions and types', () => {
    const a = completion('mesh:\n  d|CURSOR|').items.find(i => i.caption === 'dx');
    const b = completion('transient:\n  slides:\n    - kind: rectangle\n      d|CURSOR|').items.find(i => i.caption === 'dx');
    assert.match(a.docHTML, /Cartesian cell width/);
    assert.match(b.docHTML, /pixels\/step/);
    assert.deepEqual(check('mesh: {dx: 0.001, dy: "0.002"}\ntransient:\n  slides:\n    - kind: rectangle\n      dx: 0.5\n      dy: "$speed * $step"'), []);
});
test('Robin alpha and EW alpha explanations are distinct', () => {
    assert.match(completion('polar_boundary_conditions:\n  inner:\n    al|CURSOR|').items.find(i => i.caption === 'alpha').docHTML, /Robin/);
    assert.match(completion('nonlinear_solver:\n  eisenstat_walker:\n    al|CURSOR|').items.find(i => i.caption === 'alpha').docHTML, /Exponent/);
});
test('arrays and indentless arrays get coil-item context', () => {
    for (const indent of ['', '  ']) {
        assert.ok(captions(`flux_linkage:\n${indent}- name: U\n${indent}  ma|CURSOR|`).includes('material_a'));
        assert.ok(captions(`flux_linkage:\n${indent}- name: U\n${indent}  material_a: coil\n${indent}- ma|CURSOR|`).includes('material_b'));
    }
    assert.ok(captions('flux_linkage:\n  -\n    ma|CURSOR|').includes('material_a'));
});
test('arbitrary material names do not leak keys into their name slot', () => {
    assert.deepEqual(captions('materials:\n  |CURSOR|'), []);
    assert.ok(captions('materials:\n  "coil U":\n    |CURSOR|').includes('mu_r'));
    assert.ok(captions('materials:\n  mesh:\n    |CURSOR|').includes('rgb'));
});
test('material values are local RGB-bearing materials, not presets', () => {
    const text = 'materials:\n  coil_U: {rgb: [1, 2, 3]}\n  steel: {mu_r: 100}\nmaterial_presets:\n  iron: {mu_r: 100}\nflux_linkage:\n  - name: U\n    material_a: co|CURSOR|';
    assert.deepEqual(captions(text), ['coil_U']);
    assert.deepEqual(check(text.replace('co|CURSOR|', 'coil_U')), []);
});
test('both single-sided conventions remain valid', () => {
    for (const key of ['material_a', 'material_b']) {
        assert.deepEqual(check(`materials: {coil: {rgb: [1, 2, 3]}}\nflux_linkage:\n  - name: U\n    ${key}: coil`), []);
    }
});
test('unknown material diagnosis identifies the right array item', () => {
    const notes = check('materials: {coil: {rgb: [1, 2, 3]}}\nflux_linkage:\n- name: U\n  material_a: coil\n- name: V\n  material_b: missing');
    assert.equal(notes.length, 1);
    assert.equal(notes[0].row, 5);
    assert.match(notes[0].text, /flux_linkage.1.material_b/);
});
test('quoted names use safe insertion and escaped documentation', () => {
    const text = 'materials:\n  "<coil U>": {rgb: [1, 2, 3]}\nflux_linkage:\n- material_a: |CURSOR|';
    const item = completion(text).items[0];
    assert.equal(item.value, '"<coil U>"');
    assert.ok(!item.docHTML.includes('<coil U>'));
    assert.equal(yaml.load(item.value), '<coil U>');
});
test('comment/flow/quoted-value edits do not guess completion ranges', () => {
    for (const text of ['# mat|CURSOR|', 'materials: {co|CURSOR|', 'flux_linkage:\n- material_a: "co|CURSOR|', 'mesh:\n  dx: 1 # a|CURSOR|'])
        assert.deepEqual(captions(text), []);
});
test('block scalar body is not mistaken for YAML mappings', () => {
    const text = 'variables:\n  f: |\n    omp:\n      enabled: true\nmaterials:\n  air:\n    |CURSOR|';
    assert.ok(captions(text).includes('rgb'));
    assert.deepEqual(captions('variables:\n  f: |\n    ty|CURSOR|'), []);
    assert.deepEqual(check(text.replace('|CURSOR|', 'mu_r: 1')), []);
});
test('parser marks syntax errors at the original row', () => {
    const note = check('mesh:\n  dx: [1, 2\n  dy: 1')[0];
    assert.equal(note.type, 'error');
    assert.ok(note.row > 0);
    assert.match(note.text, /Syntax error/);
});
test('type validation accepts quoted numbers and substitution but not invalid types', () => {
    assert.deepEqual(check('variables: {n: 10}\nomp: {threads: "$n"}\nnonlinear_solver: {max_iterations: "60", enabled: "true"}'), []);
    assert.ok(check('omp: {threads: 2.5}').some(n => n.type === 'error'));
    assert.ok(check('nonlinear_solver: {enabled: 12}').some(n => n.type === 'error'));
});
test('material formulas and tables are not interpreted as config sections', () => {
    assert.deepEqual(check('materials:\n  iron:\n    mu_r: "100 / (1 + $H)"\n    jz: "$current * sin($step)"\n    B-H: [[0, 1], [0, 100]]'), []);
});
test('enum and cross-field radius checks', () => {
    assert.ok(check('coordinate_system: sphere').some(n => n.type === 'error'));
    assert.ok(check('coordinate_system: polar\npolar_domain: {r_start: 1, r_end: 0.5}').some(n => /Must satisfy/.test(n.text)));
    assert.deepEqual(check('coordinate_system: polar\npolar_domain: {r_start: "$inner", r_end: "$outer"}'), []);
    assert.deepEqual(check('coordinate_system: polar\npolar_domain: {r_start: 0, r_end: 1}'), []);
});
test('EW comparison and Anderson limits include solver defaults', () => {
    assert.ok(check('nonlinear_solver:\n  eisenstat_walker: {eta_min: 0.5}').some(n => /eta_min <=/.test(n.text)));
    assert.ok(check('nonlinear_solver:\n  anderson: {depth: 0, beta: 2}').length >= 2);
    assert.deepEqual(check('nonlinear_solver:\n  anderson: {depth: 1, beta: 0}\n  eisenstat_walker: {eta_min: 1e-6, eta_max: 0.1}'), []);
});
test('aliases and merge keys are inspected without modifying documents', () => {
    const text = 'materials:\n  iron: &iron {mu_r: 100}\n  steel: {<<: *iron, rgb: [1, 2, 3]}';
    assert.deepEqual(check(text), []);
    assert.ok(check('materials: &m {loop: *m}').length > 0);
});
test('empty document and invalid root are safe', () => {
    assert.deepEqual(check(''), []);
    assert.equal(check('- 1')[0].type, 'error');
});
test('shipped analysis configs and benchmark formulas produce no diagnostics', () => {
    for (const file of ['../../sample_config.yaml', '../../benchmarks/solver_speed/ieej_12step.yaml',
        '../../benchmarks/solver_speed/ieej_12step_tangent_energy.yaml']) {
        assert.deepEqual(check(fs.readFileSync(require.resolve(file), 'utf8')), [], file);
    }
    assert.deepEqual(check('dx: 0.001\ndy: 0.001\ntransient:\n  slides:\n    - angle_rad: pi/720'), []);
});
test('browser entry is loaded before app and shares Node behavior', () => {
    const source = fs.readFileSync(require.resolve('../public/yaml-intellisense'), 'utf8');
    const context = vm.createContext({});
    vm.runInContext(source, context);
    assert.equal(typeof context.OpenMagYaml.analyze, 'function');
    assert.equal(context.OpenMagYaml.analyze('mu_r: 1', schema, yaml)[0].type, 'warning');
    const html = fs.readFileSync(require.resolve('../public/index.html'), 'utf8');
    assert.ok(html.indexOf('src="yaml-intellisense.js"') < html.indexOf('src="app.js"'));
});
test('actual Ace adapter delegates completions and change diagnostics', async () => {
    const source = fs.readFileSync(require.resolve('../public/app.js'), 'utf8');
    const start = source.indexOf('async function initializeConfigEditor()');
    const end = source.indexOf('// ============================================================', start);
    const validateStart = source.indexOf('function validateYAML(editor)');
    const validateEnd = source.indexOf('function getContextPathForLine', validateStart);
    let text = 'materials: {coil_U: {rgb: [1, 2, 3]}}\nflux_linkage:\n- material_a: ';
    const handlers = {};
    let annotations, swatches = false;
    const editor = {
        setTheme() {}, setOptions() {}, getValue: () => text,
        session: {setMode() {}, on: (name, fn) => { handlers[name] = fn; }, setAnnotations: a => { annotations = a; }},
    };
    const context = vm.createContext({
        AppState: {}, OpenMagYaml: api, jsyaml: yaml, console,
        fetch: async () => ({json: async () => schema}),
        ace: {edit: () => editor, require() {}},
        addContextSnippets() {}, initMaterialSwatches: () => { swatches = true; },
    });
    vm.runInContext(source.slice(start, end) + '\n' + source.slice(validateStart, validateEnd), context);
    await context.initializeConfigEditor();
    assert.ok(swatches);
    assert.equal(editor.completers.length, 1);
    let items;
    editor.completers[0].getCompletions(editor, editor.session, {row: 2, column: 14}, '', (_, result) => { items = result; });
    assert.equal(items[0].value, 'coil_U');
    text = 'omp:\n  enabled: true';
    handlers.change();
    assert.equal(annotations[0].row, 1);
    assert.match(annotations[0].text, /misplaced/);
    assert.equal(editor.getValue(), text); // diagnostics do not rewrite user input
});
