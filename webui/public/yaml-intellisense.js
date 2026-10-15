/* Analysis-config assistance. No DOM, network, expression evaluation or mutations. */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory();
    else root.OpenMagYaml = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';
    const own = (o, k) => o != null && Object.prototype.hasOwnProperty.call(o, k);
    const map = v => v !== null && typeof v === 'object' && !Array.isArray(v);
    const id = path => JSON.stringify(path);
    const pattern = path => path.map((p, i) => typeof p === 'number' ||
        (i === 1 && ['materials', 'material_presets', 'variables'].includes(path[0])) ? '*' : p).join('.');
    const rootKeys = ['coordinate_system', 'variables', 'omp', 'mesh', 'polar_domain',
        'polar_boundary_conditions', 'boundary_conditions', 'materials', 'material_presets',
        'flux_linkage', 'transient', 'nonlinear_solver', 'domain_decomposition', 'export', 'image_path'];
    const extra = {
        '': ['dx', 'dy', 'polar', 'boundary'], // accepted legacy CLI aliases, not suggested
        'nonlinear_solver': ['initial_az_path', 'gmres_restart', 'jacobian', 'line_search_objective',
            'use_galerkin_coarsening', 'use_matrix_free_jv', 'use_phase6_precond_jfnk',
            'precond_update_frequency', 'precond_verbose', 'fine_finishing_iterations',
            'fine_finishing_tolerance', 'strict_convergence', 'eisenstat_walker_enabled'],
        'nonlinear_solver.eisenstat_walker': ['residual_cap'],
        'materials.*': ['coarsen', 'coarsen_ratio'],
    };
    const fields = {
        'polar_domain.theta_range': {type: 'number|string', description: 'Angular extent [rad], e.g. 2*pi or pi/3. Supports formulas; keep the image and angular boundary conditions consistent.'},
        'transient.slide_angle_rad': {type: 'number|string'},
        'transient.slide_angle_deg': {type: 'number|string'},
        'transient.slides.*.angle_rad': {type: 'number|string'},
        'transient.slides.*.angle_deg': {type: 'number|string'},
        'mesh.dx': {type: 'number', description: 'Cartesian cell width [m]. Smaller cells resolve finer geometry and increase the problem size.'},
        'mesh.dy': {type: 'number', description: 'Cartesian cell height [m]. Use the physical size represented by one image pixel.'},
        'transient.slides.*.dx': {type: 'number|string', description: 'Rectangle displacement in x [pixels/step]; a constant or formula. Fractional movement accumulates over steps.'},
        'transient.slides.*.dy': {type: 'number|string', description: 'Rectangle displacement in y [pixels/step]; a constant or formula. Negative values reverse movement.'},
        'transient.slides.*.direction': {values: ['horizontal', 'vertical'], type: 'string', description: 'Direction along which this band slides in image coordinates.'},
        'nonlinear_solver.eisenstat_walker': {description: 'Adapts the inner linear-solve tolerance to nonlinear progress. Enabled can reduce inner work; it does not relax the requested outer tolerance. Omitted: disabled.'},
        'nonlinear_solver.eisenstat_walker.alpha': {type: 'number', description: 'Exponent controlling how quickly the inner tolerance changes as the outer residual decreases. Default: 2.'},
        'nonlinear_solver.eisenstat_walker.gamma': {type: 'number', description: 'Multiplier for the adaptive inner tolerance. Smaller values request tighter inner solves. Default: 0.9.'},
        'nonlinear_solver.eisenstat_walker.eta_min': {type: 'number', description: 'Lower bound of inner tolerance (default 1e-6). Smaller is stricter, potentially more expensive.'},
        'nonlinear_solver.eisenstat_walker.eta_max': {type: 'number', description: 'Upper bound and initial inner tolerance (default 0.1). Must not be below eta_min.'},
        'image_path': {type: 'string', description: 'Model image path used by the WebUI. Does not define grid spacing.'},
    };
    function children(path, schema, includeHidden = true) {
        const p = pattern(path);
        if (!path.length) return [...rootKeys, ...(includeHidden ? extra[''] : [])];
        if (path[0] === 'polar') return children(['polar_domain', ...path.slice(1)], schema, includeHidden);
        if (path[0] === 'boundary' && path.length === 1) return ['left', 'right', 'top', 'bottom', 'inner', 'outer', 'theta_min', 'theta_max'];
        if (['materials', 'material_presets', 'variables'].includes(p)) return null; // arbitrary names
        if (p === 'variables.*') return null;
        let keys;
        if (p === 'materials.*') keys = schema.keywords.materials?.childrenProperties;
        else if (p === 'material_presets.*') keys = schema.keywords.materials?.childrenProperties?.filter(k => k !== 'rgb');
        else if (p === 'flux_linkage.*') keys = ['name', 'start', 'end', 'material_a', 'material_b'];
        else if (p === 'transient.slides.*') keys = ['name', 'kind', 'direction', 'region_start', 'region_end',
            'pixels_per_step', 'angle_rad', 'angle_deg', 'wrap_mode', 'vacuum_rgb', 'rect', 'dx', 'dy'];
        else if (/^(polar_boundary_conditions|boundary_conditions)\.[^.]+$/.test(p))
            keys = ['type', 'value', 'alpha', 'beta', 'gamma', ...(p.startsWith('polar_') ? ['value_profile', 'gamma_profile'] : [])];
        else keys = schema.keywords[path.at(-1)]?.children;
        if (!keys) return null; // do not interpret arbitrary formula/table subtrees
        return [...new Set([...keys, ...(includeHidden ? extra[p] || [] : [])])];
    }
    function info(path, schema) {
        if (path.length === 1 && ['dx', 'dy'].includes(path[0])) return info(['mesh', path[0]], schema);
        if (path[0] === 'polar') return info(['polar_domain', ...path.slice(1)], schema);
        const p = pattern(path);
        const base = own(schema.keywords, path.at(-1)) ? schema.keywords[path.at(-1)] : {};
        if (/^(polar_boundary_conditions|boundary_conditions)\.[^.]+\.(alpha|beta|gamma)$/.test(p))
            return {type: 'number', description: 'Robin boundary coefficient: alpha * Az + beta * dAz/dn = gamma. This is a boundary value, not a nonlinear-solver parameter.'};
        return {...base, ...(fields[p] || {})};
    }
    // A conservative block-style locator, not a second YAML parser. Parsed YAML
    // below is authoritative. Flow mappings get their enclosing block's location.
    function scan(text, yaml) {
        const entries = [], stack = [], counts = new Map();
        let blockIndent = null;
        for (const [row, line] of text.split(/\r?\n/).entries()) {
            const indent = line.match(/^ */)[0].length;
            if (!line.trim() || line.trimStart().startsWith('#')) continue;
            if (blockIndent !== null && indent > blockIndent) continue;
            blockIndent = null;
            const dash = line.match(/^( *)-\s*(?:#.*)?$/);
            const m = dash ? [line, dash[1], '- ', '', ''] : line.match(/^( *)(- +)?("(?:[^"\\]|\\.)*"|'(?:[^']|'')*'|[\w$][\w$ .-]*)\s*:(?=\s|$)(.*)$/);
            if (!m) continue;
            const sequence = !!m[2], keyIndent = indent + (m[2]?.length || 0);
            while (stack.length && stack.at(-1).indent >= (sequence ? indent : keyIndent)) {
                if (sequence && stack.at(-1).indent === indent && stack.at(-1).container) break;
                stack.pop();
            }
            let parent = stack.length ? stack.at(-1).path : [];
            if (sequence) {
                const n = counts.get(id(parent)) || 0;
                counts.set(id(parent), n + 1);
                parent = [...parent, n];
                stack.push({indent, path: parent});
            }
            if (dash) continue;
            let key = m[3].trim();
            if (/^['"]/.test(key)) { try { key = String(yaml.load(key)); } catch (_) { continue; } }
            const path = [...parent, key];
            const value = m[4].trim();
            entries.push({row, column: keyIndent, path, value});
            stack.push({indent: keyIndent, path, container: !value || value.startsWith('#')});
            if (/^[|>][+\-\d]*(?:\s|$)/.test(value)) blockIndent = keyIndent;
        }
        return entries;
    }
    function number(v) {
        if (typeof v === 'number') return v;
        if (typeof v === 'string' && /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(v.trim())) return Number(v);
        return null;
    }
    function validType(v, types) {
        if (typeof v === 'string' && /\$[a-zA-Z_]\w*/.test(v)) return true; // solver substitution, not evaluated here
        return types.split('|').some(t => {
            if (t === 'number' || t === 'integer') {
                const n = number(v);
                return n !== null && Number.isFinite(n) && (t !== 'integer' || Number.isInteger(n));
            }
            if (t === 'boolean') return typeof v === 'boolean' || (typeof v === 'string' && /^(true|false|yes|no|on|off)$/i.test(v));
            if (t === 'array') return Array.isArray(v);
            return typeof v === t;
        });
    }
    function analyze(text, schema, yaml) {
        const annotations = [];
        let doc;
        try { doc = yaml.load(text); }
        catch (e) { return [{row: e.mark?.line || 0, column: e.mark?.column || 0, type: 'error', text: `Syntax error: ${e.reason || e.message}`}]; }
        if (doc == null) return annotations;
        const entries = scan(text, yaml), locations = new Map(entries.map(e => [id(e.path), e]));
        function report(path, message, type = 'warning') {
            const original = path;
            while (path.length && !locations.has(id(path))) path = path.slice(0, -1);
            const where = locations.get(id(path)) || {row: 0, column: 0};
            annotations.push({row: where.row, column: where.column, type, text: `${original.join('.') || 'Config'}: ${message}`});
        }
        if (!map(doc)) { report([], 'Expected a mapping of configuration sections.', 'error'); return annotations; }
        const ancestors = new Set();
        function visit(value, path) {
            if (!map(value) && !Array.isArray(value)) return;
            if (ancestors.has(value)) { report(path, 'Recursive YAML aliases are not supported by the configuration editor.'); return; }
            ancestors.add(value);
            if (Array.isArray(value)) {
                if (['flux_linkage', 'transient.slides'].includes(pattern(path))) value.forEach((v, i) => {
                    if (!map(v)) report([...path, i], 'Expected a mapping.', 'error'); else visit(v, [...path, i]);
                });
            } else {
                const allowed = children(path, schema);
                for (const [key, v] of Object.entries(value)) {
                    const at = [...path, key];
                    if (allowed && !allowed.includes(key)) { report(at, `Unknown or misplaced key; allowed here: ${allowed.join(', ')}.`); continue; }
                    // Named material and variable entries are not keyword definitions.
                    const named = path.length === 1 && ['materials', 'material_presets', 'variables'].includes(path[0]);
                    const rule = named ? {} : info(at, schema);
                    if (rule.type && v !== null && !validType(v, rule.type)) report(at, `Expected ${rule.type}.`, 'error');
                    if (rule.values && v !== null && !String(v).includes('$') && !rule.values.includes(v))
                        report(at, `Expected one of: ${rule.values.join(', ')}.`, 'error');
                    const childKeys = children(at, schema);
                    const expectsMap = childKeys && !['flux_linkage.*', 'transient.slides.*'].includes(pattern(at));
                    if ((expectsMap || ['materials', 'variables', 'material_presets'].includes(pattern(at))) && v !== null && !map(v))
                        report(at, 'Expected a mapping.', 'error');
                    if (path[0] !== 'variables') visit(v, at);
                }
            }
            ancestors.delete(value);
        }
        visit(doc, []);
        const polar = doc.polar_domain;
        if (map(polar) && (doc.coordinate_system || 'cartesian') === 'polar') {
            const start = number(polar.r_start ?? 0), end = number(polar.r_end ?? 1);
            if (start !== null && end !== null && !(start >= 0 && start < end))
                report(['polar_domain', 'r_start'], 'Must satisfy 0 <= r_start < r_end [m].', 'error');
        }
        const ew = doc.nonlinear_solver?.eisenstat_walker;
        if (map(ew)) {
            const lo = number(ew.eta_min ?? 1e-6), hi = number(ew.eta_max ?? .1);
            if (lo !== null && hi !== null && !(lo > 0 && lo <= hi))
                report(['nonlinear_solver', 'eisenstat_walker', 'eta_min'], 'Use 0 < eta_min <= eta_max.');
        }
        const aa = doc.nonlinear_solver?.anderson;
        if (map(aa)) {
            if (own(aa, 'depth') && number(aa.depth) !== null && number(aa.depth) < 1)
                report(['nonlinear_solver', 'anderson', 'depth'], 'Must be at least 1.', 'error');
            if (own(aa, 'beta') && number(aa.beta) !== null && !(number(aa.beta) >= 0 && number(aa.beta) <= 1))
                report(['nonlinear_solver', 'anderson', 'beta'], 'Must be between 0 and 1.', 'error');
        }
        const materials = materialNames(doc);
        if (Array.isArray(doc.flux_linkage)) doc.flux_linkage.forEach((coil, i) => {
            if (!map(coil)) return;
            for (const key of ['material_a', 'material_b']) if (own(coil, key) && !String(coil[key]).includes('$') && !materials.includes(coil[key]))
                report(['flux_linkage', i, key], 'Unknown material or missing RGB. The solver skips this flux-linkage entry.');
        });
        return annotations;
    }
    function materialNames(doc) {
        return map(doc?.materials) ? Object.entries(doc.materials)
            .filter(([, v]) => map(v) && Array.isArray(v.rgb) && v.rgb.length >= 3).map(([k]) => k) : [];
    }
    function escape(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c])); }
    function complete(text, pos, schema, yaml) {
        const lines = text.split(/\r?\n/), before = (lines[pos.row] || '').slice(0, pos.column);
        // Flow-style edits, comments and quoted partial values are deliberately
        // left to manual editing; no guessed replacement range inside quotes.
        const m = before.match(/^( *)(- +)?(?:([\w-]+):\s*(\$?[\w.-]*))?$/);
        const keyOnly = before.match(/^( *)(- +)?[\w-]*$/);
        if (!m && !keyOnly) return {items: [], context: null};
        const valueMode = !!m?.[3];
        const padding = (m || keyOnly)[1] + ((m || keyOnly)[2] || '');
        const replacement = padding + '__cursor__: null';
        const probe = [...lines.slice(0, pos.row), replacement].join('\n');
        const entry = scan(probe, yaml).find(e => e.row === pos.row && e.path.at(-1) === '__cursor__');
        if (!entry) return {items: [], context: null};
        const parent = entry.path.slice(0, -1), key = valueMode ? m[3] : null;
        const allowed = children(parent, schema, false) || [];
        const items = [];
        const item = (caption, value, rule, meta) => ({caption, value, meta, score: 1000,
            docHTML: `<b>${escape(caption)}</b><br>${escape(rule.description)}<br><code>${escape(rule.example)}</code>`});
        if (valueMode && allowed.includes(key)) {
            const rule = info([...parent, key], schema);
            let values = rule.values || (rule.type === 'boolean' ? ['true', 'false'] : []);
            if (pattern(parent) === 'flux_linkage.*' && ['material_a', 'material_b'].includes(key)) {
                // Neutralize only the incomplete current value for parsing.
                const candidate = [...lines]; candidate[pos.row] = padding + key + ': null';
                try { values = materialNames(yaml.load(candidate.join('\n'))); } catch (_) { values = []; }
                for (const name of values) {
                    const value = /^[a-zA-Z_][\w-]*$/.test(name) && !/^(true|false|null|yes|no|on|off)$/i.test(name) ? name : JSON.stringify(name);
                    items.push(item(name, value, {description: 'Coil cross-section from this configuration. material_a contributes +mean(Az); material_b contributes -mean(Az). Either side alone is allowed.'}, 'material'));
                }
            } else for (const value of values) items.push(item(String(value), String(value), rule, 'value'));
        } else if (!valueMode) for (const name of allowed) items.push(item(name, name + ': ', info([...parent, name], schema), 'key'));
        return {items, context: {path: parent, valueMode, key}};
    }
    return {analyze, complete, scan, children, info};
});
