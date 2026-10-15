(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory(require('./yaml-intellisense'));
    else root.WorkspaceCore = factory(root.OpenMagYaml);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (assist) {
    'use strict';
    function dirty(current, saved) { return saved !== null && current !== saved; }
    function inspect(content, schema, yaml) {
        const diagnostics = assist.analyze(content, schema, yaml);
        let doc;
        try { doc = yaml.load(content); } catch (_) { doc = null; }
        const object = doc && typeof doc === 'object' && !Array.isArray(doc);
        const scalar = (v, fallback) => ['number', 'string', 'boolean'].includes(typeof v) ? String(v) : fallback;
        const summary = object ? {
            coordinates: scalar(doc.coordinate_system, 'cartesian (default)'),
            solver: scalar(doc.nonlinear_solver?.solver_type, 'solver default'),
            nonlinear: scalar(doc.nonlinear_solver?.enabled, 'false (default)'),
            tolerance: scalar(doc.nonlinear_solver?.tolerance, 'solver default'),
            steps: scalar(doc.transient?.total_steps, 'solver default'),
            transient: scalar(doc.transient?.enabled, 'false (default)'),
            threads: scalar(doc.omp?.threads, 'runtime default'),
            chunks: scalar(doc.transient?.parallel_chunks, 'solver default'),
            materialCount: doc.materials && typeof doc.materials === 'object' ? Object.keys(doc.materials).length : 0,
        } : null;
        return {version: 1, valid: Boolean(object) && !diagnostics.some(d => d.type === 'error'), diagnostics, summary,
            scope: 'Editor assistance only; formulas, external files and physical consistency are not fully evaluated.'};
    }
    function files(items, query, sort) {
        const q = query.trim().toLocaleLowerCase();
        return items.filter(f => !q || String(f.name).toLocaleLowerCase().includes(q)).slice().sort((a, b) => {
            if (sort === 'size') return (Number(b.size) || 0) - (Number(a.size) || 0) || String(a.name).localeCompare(String(b.name));
            if (sort === 'name') return String(a.name).localeCompare(String(b.name), undefined, {numeric: true});
            return (Date.parse(b.modified || b.mtime || '') || 0) - (Date.parse(a.modified || a.mtime || '') || 0) || String(a.name).localeCompare(String(b.name));
        });
    }
    return {dirty, inspect, files};
});
