/* DOM adapter for the shared workspace contract. No solver ownership here. */
const Workbench = (() => {
    let mounted = false, refreshTimer, clock, startedAt = null;
    const el = id => document.getElementById(id);
    const text = (id, value) => { if (el(id) && el(id).textContent !== value) el(id).textContent = value; };
    function dirty() { return WorkspaceCore.dirty(AppState.aceEditor?.getValue() ?? '', AppState.configData); }
    function inspect(content) { return WorkspaceCore.inspect(content, AppState.yamlSchema || {keywords: {}}, jsyaml); }
    function refresh() {
        if (!mounted) return;
        text('workspaceConfig', AppState.loadedConfigFile || 'No configuration');
        text('workspaceSaved', AppState.configSaving ? 'Saving…' : AppState.configLoadPending ? 'Loading…' : dirty() ? '● Unsaved changes' : 'Saved');
        el('workspaceSaved').classList.toggle('dirty', dirty());
        text('workspaceImage', AppState.uploadedImageFilename || 'No image selected');
        text('workspaceResult', AppState.resultsData.currentResult ? `Result: ${AppState.resultsData.currentResult.split(/[\\/]/).pop()}` : 'No result loaded');
        const running = !!el('solverBtn')?.disabled;
        if (running && startedAt === null) { startedAt = performance.now(); clock = setInterval(refreshClock, 1000); }
        if (!running && startedAt !== null) { clearInterval(clock); clock = null; startedAt = null; }
        refreshClock();
        const report = inspect(AppState.aceEditor?.getValue() || '');
        const problems = el('workspaceProblems');
        problems.replaceChildren();
        text('problemCount', `(${report.diagnostics.length})`);
        if (!report.diagnostics.length) { const p = document.createElement('p'); p.textContent = 'No editor issues detected.'; problems.append(p); }
        report.diagnostics.forEach(d => {
            const button = document.createElement('button');
            button.className = `problem-item ${d.type}`;
            button.textContent = `${d.type === 'error' ? 'Error' : 'Warning'} · Line ${d.row + 1}\n${d.text}`;
            button.onclick = () => { switchTab('config'); AppState.aceEditor.gotoLine(d.row + 1, d.column, true); AppState.aceEditor.focus(); };
            problems.append(button);
        });
        const s = report.summary || {};
        const values = [
            ['Configuration', AppState.loadedConfigFile || 'None'], ['Input image', AppState.uploadedImageFilename || 'Not selected'],
            ['Material library', AppState.selectedLibrary || 'None'], ['Coordinates', s.coordinates || 'Invalid YAML'],
            ['Nonlinear / method', `${s.nonlinear || '—'} / ${s.solver || '—'}`], ['Outer tolerance', s.tolerance || '—'],
            ['Transient / steps', `${s.transient || '—'} / ${s.steps || '—'}`], ['Threads / chunks', `${s.threads || '—'} / ${s.chunks || '—'}`],
        ];
        const review = el('workspaceReview'); review.replaceChildren();
        for (const [label, value] of values) {
            const group = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
            dt.textContent = label; dd.textContent = value; group.append(dt, dd); review.append(group);
        }
        text('workspaceReviewNote', (dirty() ? 'DRAFT — changes are not saved. Run will ask which version to use. ' : 'Saved editor settings. ') +
            'Values shown before formula evaluation; this is not a complete physical validation.');
    }
    function refreshClock() {
        const status = el('solverProgressContainer')?.style.display === 'none' ? 'Idle' : (el('solverProgressText')?.textContent || 'Idle');
        const elapsed = startedAt === null ? '' : ` · ${Math.floor((performance.now() - startedAt) / 60000)}m ${Math.floor((performance.now() - startedAt) / 1000) % 60}s`;
        text('workspaceRun', (startedAt === null && status === 'Initializing…' ? 'Idle' : status) + elapsed);
    }
    function schedule() { clearTimeout(refreshTimer); refreshTimer = setTimeout(refresh, 160); }
    function chooseRunVersion() {
        return new Promise(resolve => {
            const dialog = document.createElement('dialog'); dialog.className = 'workspace-dialog';
            const title = document.createElement('h2'); title.id = 'runVersionTitle'; title.textContent = 'Which configuration should run?';
            dialog.setAttribute('aria-labelledby', title.id);
            const p = document.createElement('p'); p.textContent = 'The editor has unsaved changes. Save them before running, or explicitly use the previously saved file.';
            const buttons = document.createElement('div'); buttons.className = 'button-group';
            for (const [value, label, style] of [['cancel','Cancel','btn-secondary'], ['saved','Run saved file','btn-secondary'], ['save','Save & run','btn-primary']]) {
                const button = document.createElement('button'); button.textContent = label; button.className = style;
                button.onclick = () => dialog.close(value); buttons.append(button);
            }
            dialog.append(title, p, buttons); document.body.append(dialog);
            dialog.addEventListener('close', () => { const value = dialog.returnValue || 'cancel'; dialog.remove(); resolve(value); }, {once:true});
            dialog.showModal(); buttons.firstChild.focus();
        });
    }
    async function prepareRun() {
        if (AppState.configSaving || AppState.configLoadPending) { showStatus('solverStatus', 'Wait for the configuration operation to finish.', 'warning'); return null; }
        if (!AppState.loadedConfigFile || !AppState.uploadedImageFilename) { showStatus('solverStatus', 'Load a configuration and select an input image first.', 'warning'); return null; }
        const file = AppState.loadedConfigFile;
        const current = AppState.aceEditor.getValue();
        let content = AppState.configData;
        if (dirty()) {
            const choice = await chooseRunVersion();
            if (choice === 'cancel') return null;
            if (file !== AppState.loadedConfigFile || current !== AppState.aceEditor.getValue()) return null;
            if (choice === 'save') {
                if (!await saveConfig() || dirty()) return null;
                content = AppState.configData;
            }
        }
        const report = inspect(content || '');
        if (!report.valid && !confirm('The editor reports configuration errors. Review them in Configuration.\n\nRun this saved file anyway?')) return null;
        return {configFile: file, imageFile: AppState.uploadedImageFilename, userId: AppState.userId, materialLibraryFile: AppState.selectedLibrary || null};
    }
    function mount() {
        if (mounted) return; mounted = true;
        AppState.aceEditor.session.on('change', schedule);
        window.addEventListener('beforeunload', e => { if (dirty() || el('solverBtn')?.disabled) { e.preventDefault(); e.returnValue = ''; } });
        document.addEventListener('keydown', e => {
            const modalOpen = document.querySelector('dialog[open]') || Array.from(document.querySelectorAll('.modal')).some(m => getComputedStyle(m).display !== 'none');
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's' && AppState.currentTab === 'config' && !modalOpen) {
                e.preventDefault(); saveConfig();
            }
        });
        document.querySelectorAll('.tab').forEach((tab, i, tabs) => {
            tab.tabIndex = tab.classList.contains('active') ? 0 : -1;
            tab.addEventListener('keydown', e => {
                if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(e.key)) return;
                e.preventDefault();
                const next = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length-1 : (i + (['ArrowLeft','ArrowUp'].includes(e.key) ? -1 : 1) + tabs.length) % tabs.length;
                tabs[next].focus(); switchTab(tabs[next].dataset.tab);
            });
        });
        document.querySelectorAll('.tab-content').forEach(panel => { panel.setAttribute('role','tabpanel'); panel.setAttribute('aria-labelledby', panel.id.replace('tab-','nav-')); });
        const observer = new MutationObserver(schedule);
        for (const id of ['solverBtn','solverProgressText','uploadedImage','activeLibraryBadge','resultSelectDisplay']) if (el(id)) observer.observe(el(id), {attributes:true, childList:true, subtree:true, characterData:true});
        refresh();
    }
    return {mount, refresh, schedule, dirty, prepareRun};
})();
