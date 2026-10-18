/* Results navigation and provenance only. Never owns field arrays or Plotly traces. */
const ResultsWorkspace = (() => {
    let pending = null;
    let loadError = '';
    let listState = { loading: false, count: null, error: '' };
    const byId = id => document.getElementById(id);
    const leaf = path => String(path || '').split(/[\\/]/).pop();
    const timelineTypes = new Set(['force_x_time', 'force_y_time', 'torque_time',
        'energy_time', 'virtual_work', 'flux_linkage_time', 'back_emf_time']);
    // Avoid repeated live-region announcements while only the field step changes.
    const text = (id, value) => { const el = byId(id); if (el && el.textContent !== value) el.textContent = value; };
    const hidden = (id, value) => { const el = byId(id); if (el) el.hidden = value; };

    function refresh() {
        const loaded = AppState.resultsData.currentResult || '';
        const selected = byId('resultSelect')?.value || '';
        const busy = !!pending;
        text('loadedResultName', loaded ? leaf(loaded) : 'None — load a result first');
        if (byId('loadedResultName')) byId('loadedResultName').title = loaded;
        text('loadedResultCoordinates', loaded ? (AppState.analysisConditions?.coordinate_system || 'unknown') : '—');
        text('loadedResultStep', loaded ? `${AppState.currentStep} / ${AppState.totalSteps}` : '—');
        hidden('resultDisplayOptions', !loaded);
        text('resultSelectionHint', selected && selected !== loaded
            ? `Selected, not loaded: ${leaf(selected)}. Click Load selected to change the plots.`
            : 'Selecting a result does not change the plots until you load it.');

        let message = loaded ? 'Data loaded. Add plots below; missing outputs are reported in each plot.'
            : 'No result loaded. Choose a result above and click Load selected.';
        if (listState.count === 0 && !loaded) message = 'No saved results found. Run an analysis or import a backup in File Manager, then refresh the list.';
        if (listState.loading) message = 'Refreshing the result list…';
        if (listState.error) message = `Could not refresh the list: ${listState.error}. Try Refresh list again.`;
        if (loadError) message = `${loadError}${loaded ? ` Loaded result: ${leaf(loaded)}.` : ''} Select a result and try loading again.`;
        if (pending) message = `Loading ${leaf(pending.resultPath)}… Waiting for previews and plots to finish.`;
        text('resultLoadStatus', message);
        if (byId('resultLoadStatus')) byId('resultLoadStatus').dataset.state = busy ? 'loading' : (loadError || listState.error ? 'error' : 'ready');
        if (byId('loadResultBtn')) {
            byId('loadResultBtn').disabled = !selected || busy;
            byId('loadResultBtn').textContent = busy ? 'Loading…' : 'Load selected';
        }
        if (byId('reloadResultBtn')) byId('reloadResultBtn').disabled = !loaded || busy;
        if (byId('refreshResultListBtn')) byId('refreshResultListBtn').disabled = listState.loading;

        const controls = document.querySelector('.dashboard-controls');
        if (controls) {
            controls.inert = !loaded || busy;
            controls.setAttribute('aria-disabled', String(!loaded || busy));
        }
        document.querySelectorAll('.palette-item').forEach(item => item.setAttribute('aria-disabled', String(!loaded || busy)));
        const hasPlots = !!document.querySelector('#dashboard-grid .grid-stack-item');
        hidden('emptyCanvas', hasPlots);
        text('resultsEmptyTitle', busy ? 'Loading result…' : loaded ? 'Choose what you want to see' : 'Load a result to start exploring');
        text('resultsEmptyHint', loaded
            ? 'Add a |B| distribution below, or click any plot in the palette. You can add several plots side by side.'
            : 'Choose a saved result above. No results yet? Run an analysis or import a backup.');
        hidden('resultsEmptyLoad', !!loaded || busy);
        hidden('resultsEmptyAdd', !loaded || busy);
        if (typeof Workbench !== 'undefined') Workbench.schedule();
    }

    function chooseResult() {
        switchTab('dashboard');
        const trigger = byId('resultSelectTrigger');
        trigger?.scrollIntoView({ block: 'center', behavior: 'auto' });
        trigger?.focus();
    }

    async function reload() {
        const loaded = AppState.resultsData.currentResult;
        if (!loaded || pending) return;
        const select = byId('resultSelect');
        // A running result can temporarily disappear from a refreshed list.
        if (!Array.from(select.options).some(option => option.value === loaded)) {
            const option = document.createElement('option');
            option.value = loaded; option.textContent = leaf(loaded);
            select.appendChild(option);
        }
        select.value = loaded;
        updateCustomSelectDisplay(loaded);
        return loadSelectedResult({ refreshData: true, preserveStep: true });
    }

    function canAddPlot() {
        if (AppState.resultsData.currentResult && !pending) return true;
        chooseResult();
        refresh();
        return false;
    }

    function seriesSummary(traces) {
        let count = 0, min = Infinity, max = -Infinity;
        for (const trace of traces || []) {
            let n = 0;
            for (let i = 0; i < (trace.x?.length || 0); i++) {
                if (!Number.isFinite(trace.x[i]) || !Number.isFinite(trace.y?.[i])) continue;
                n++;
                min = Math.min(min, trace.x[i]); max = Math.max(max, trace.x[i]);
            }
            count = Math.max(count, n);
        }
        return count ? `Timeline · up to ${count} points/series · steps ${min}–${max}` : 'Timeline · available series';
    }

    function plotStatus(context, status, traces = null) {
        if (context?.scope !== 'dashboard') return;
        const container = context.container;
        // Also guard here: stale completions must not relabel a newer graph.
        if (!container || container._activeRenderToken !== context.token) return;
        const content = container.closest('.grid-stack-item-content');
        const label = content?.querySelector('.plot-source');
        if (!label) return;
        const scope = timelineTypes.has(content.dataset.plotType)
            ? seriesSummary(status === 'ready' ? traces : null) : `Step ${context.step}`;
        const coordinates = context.analysisConditions?.coordinate_system || 'unknown coordinates';
        const view = context.polarView?.isPolar
            ? (context.polarView.cartesianTransform ? ' · arc view' : ' · polar grid') + (context.polarView.fullModel ? ' · full model' : '') : '';
        const prefix = status === 'ready' ? '' : status === 'loading' ? 'Loading: ' : 'Unavailable / error: ';
        label.textContent = `${prefix}${leaf(context.resultPath)} · ${scope} · ${coordinates}${view}`;
        label.title = context.resultPath;
        label.dataset.state = status;
        container.setAttribute('aria-busy', String(status === 'loading'));
    }

    function invalidatePlots() {
        document.querySelectorAll('#dashboard-grid .plot-source').forEach(label => {
            label.textContent = 'Updating data…';
            label.dataset.state = 'loading';
        });
    }

    return {
        refresh, chooseResult, reload, canAddPlot, plotStatus, invalidatePlots, seriesSummary,
        listLoading() { listState.loading = true; listState.error = ''; refresh(); },
        listReady(count) { listState = { loading: false, count, error: '' }; refresh(); },
        listFailed(error) { listState.loading = false; listState.error = error.message; refresh(); },
        beginLoad(context) { pending = context; loadError = ''; refresh(); },
        loadFailed(context, error) {
            if (pending === context) loadError = `Could not load ${leaf(context.resultPath)}: ${error.message}.`;
        },
        finishLoad(context) { if (pending === context) { pending = null; refresh(); } },
    };
})();
