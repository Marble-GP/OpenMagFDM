/* Result annotations and display preferences; no field data is retained here. */
const ResultImprovements = (() => {
    let generation = 0, current = null;
    const drafts = new Map();
    const draftKey = (user, folder) => JSON.stringify([user, folder]);
    const el = id => document.getElementById(id);
    function dirty() { return !!current && el('resultMemoText').value !== current.saved; }
    function status(message, error = false) {
        el('resultMemoStatus').textContent = message;
        el('resultMemoStatus').dataset.state = error ? 'error' : 'info';
    }
    function changed() {
        if (!current || current.busy) return;
        el('resultMemoSave').disabled = !dirty();
        el('resultMemoBadge').textContent = dirty() ? 'Unsaved changes' : 'Saved';
        el('resultMemoBadge').dataset.state = dirty() ? 'dirty' : 'saved';
        el('resultMemoDiscard').hidden = true; status('');
    }
    function keepEditing() { el('resultMemoDiscard').hidden = true; el('resultMemoText').focus(); }
    function requestReload(discard = false) {
        if (current?.busy) return;
        if (!discard && dirty()) { el('resultMemoDiscard').hidden = false; return; }
        return description(el('resultMemoTitle').textContent, {discard});
    }
    async function description(folder, {discard = false} = {}) {
        const user = AppState.userId, key = draftKey(user, folder);
        // Result data may be reloaded while notes are being edited or saved.
        // Only the explicit description reload may discard this editor's draft.
        if (!discard && current?.user === user && current.folder === folder && (dirty() || current.busy)) return;
        if (current && (dirty() || current.busy)) {
            current.value = el('resultMemoText').value;
            drafts.set(draftKey(current.user, current.folder), current);
        }
        if (discard) drafts.delete(key);
        const token = ++generation;
        current = null;
        el('resultMemoEditor').hidden = false;
        el('resultMemoTitle').textContent = folder;
        el('resultOriginalInput').href = '/api/get-original-input?result=' + encodeURIComponent('outputs/'+user+'/'+folder);
        el('resultOriginalInput').textContent = 'Original input image (initial state)';
        el('resultMemoText').value = '';
        el('resultMemoText').disabled = true;
        el('resultMemoSave').disabled = true; el('resultMemoSave').textContent = 'Save description';
        el('resultMemoReload').disabled = true;
        el('resultMemoDiscard').hidden = true;
        el('resultMemoBadge').textContent = 'Loading'; el('resultMemoBadge').dataset.state = '';
        status('Loading description…');
        try {
            const response = await fetch('/api/user-outputs/' + encodeURIComponent(folder) + '/description?userId=' + encodeURIComponent(user));
            const data = await response.json();
            if (!response.ok || !data.success) throw Error(data.error || 'Description load failed');
            if (token !== generation || user !== AppState.userId) return;
            const draft = drafts.get(key);
            current = draft || {folder, user, revision: data.revision, saved:data.description || '', busy:false};
            el('resultMemoText').value = draft ? draft.value : current.saved;
            el('resultMemoText').disabled = current.busy;
            changed();
            if (current.busy) {
                el('resultMemoBadge').textContent = 'Saving';
                el('resultMemoSave').textContent = 'Saving…';
                status('Saving changes…');
            } else if (draft) {
                status(data.revision === draft.revision ? 'Unsaved description restored for this result.'
                    : 'Unsaved description restored. The saved version changed; reload explicitly before replacing it.', data.revision !== draft.revision);
            }
            const image=await fetch(el('resultOriginalInput').href,{method:'HEAD'});
            if(token===generation&&!image.ok){
                el('resultOriginalInput').removeAttribute('href');
                el('resultOriginalInput').textContent=image.status===404?'Original input was not saved for this result.':'Original input could not be retrieved.';
            }
        } catch (error) {
            if (token === generation) { status(error.message,true); if (!current) el('resultMemoBadge').textContent = 'Unavailable'; }
        } finally { if (token === generation) el('resultMemoReload').disabled = !!current?.busy; }
    }
    async function save() {
        if (!current || current.busy || !dirty() || current.user !== AppState.userId) return;
        const target = current, token = generation, value = el('resultMemoText').value;
        target.busy = true;
        el('resultMemoSave').disabled = true; el('resultMemoText').disabled = true; el('resultMemoReload').disabled = true;
        el('resultMemoSave').textContent = 'Saving…'; el('resultMemoBadge').textContent = 'Saving';
        el('resultMemoDiscard').hidden = true; status('Saving changes…');
        try {
            const response = await fetch('/api/user-outputs/' + encodeURIComponent(target.folder) + '/description', {
                method: 'PUT', headers: {'Content-Type':'application/json'},
                body: JSON.stringify({userId:target.user, description:value, expectedRevision:target.revision})
            });
            const data = await response.json();
            if (!response.ok || !data.success) throw Error(data.error || 'Save failed; reload the description before retrying');
            target.revision = data.revision; target.saved = value;
            const key = draftKey(target.user, target.folder);
            if (drafts.get(key) === target) drafts.delete(key);
            if (current !== target || target.user !== AppState.userId) return;
            status('Description saved.');
            void refreshResultsList().catch(()=>{ if (token === generation) status('Description saved. The results list could not be refreshed.',true); });
        } catch (error) { if (current === target && target.user === AppState.userId) status(error.message + ' Your input has been kept.',true); }
        finally {
            target.busy = false;
            if (current === target && target.user === AppState.userId) {
                el('resultMemoText').disabled = false; el('resultMemoReload').disabled = false;
                el('resultMemoSave').textContent = 'Save description'; el('resultMemoSave').disabled = !dirty();
                el('resultMemoBadge').textContent = dirty() ? 'Unsaved changes' : 'Saved';
                el('resultMemoBadge').dataset.state = dirty() ? 'dirty' : 'saved';
            }
        }
    }
    function syncDescription(folder, value, revision) {
        if (!current || current.user !== AppState.userId || current.folder !== folder) return;
        // Never replace a draft from Results when the same notes were edited in Files.
        if (dirty() || current.busy) { status('Description changed in Files. Reload to use the saved version; your draft is still here.',true); return; }
        current.saved = value; current.revision = revision; el('resultMemoText').value = value; changed();
    }
    function aspect(checked) {
        AppState.polarAspectLocked = checked;
        try { localStorage.setItem('omfdm.polarAspectLocked', String(checked)); } catch {}
        refreshAllPlots();
    }
    function rename(oldName, newName) {
        const oldKey = draftKey(AppState.userId, oldName), draft = drafts.get(oldKey);
        if (draft) { drafts.delete(oldKey); draft.folder = newName; drafts.set(draftKey(AppState.userId, newName), draft); }
        if (!current || current.user !== AppState.userId || current.folder !== oldName) return;
        current.folder = newName;
        el('resultMemoTitle').textContent = newName;
        const link = el('resultOriginalInput');
        if (link.hasAttribute('href')) link.href = `/api/get-original-input?result=${encodeURIComponent('outputs/'+AppState.userId+'/'+newName)}`;
    }
    document.addEventListener('DOMContentLoaded', () => {
        try { AppState.polarAspectLocked = localStorage.getItem('omfdm.polarAspectLocked') !== 'false'; } catch {}
        el('polarAspectLocked').checked = AppState.polarAspectLocked;
    });
    return {description, save, aspect, rename, changed, requestReload, keepEditing, syncDescription};
})();

async function optionalMaterialCanvas(url, data, context, container) {
    try {
        const canvas = await loadImageToCanvas(url, context.controller.signal);
        delete container.dataset.materialBackground;
        return canvas;
    } catch (error) {
        if (error.name === 'AbortError' || !isRenderContextCurrent(context, container)) throw error;
        const canvas = document.createElement('canvas');
        canvas.width = data[0].length; canvas.height = data.length;
        container.dataset.materialBackground = 'unavailable';
        container.title = 'Material image unavailable. Field lines are calculated from Az; no material background is shown.';
        return canvas;
    }
}

async function contourCanvasInWorker(data, context) {
    const rows=data.length,cols=data[0].length;
    // Charge the cloned rows and worker pixels against the shared working budget.
    const reservation=estimateFieldDataBytes(data)*2+rows*cols*8;
    AppState.fieldWorkerReservedBytes=(AppState.fieldWorkerReservedBytes||0)+reservation;
    const previousReservation=AppState.fieldDecodeReservedBytes;
    try {reserveFieldDecodeWorkspace(0,'Contour worker');}
    catch(error){AppState.fieldWorkerReservedBytes-=reservation;throw error;}
    finally{AppState.fieldDecodeReservedBytes=previousReservation;}
    const started=performance.now();
    const result=await new Promise((resolve,reject)=>{
        const worker=new Worker('/contour-worker.js');
        const signal=context.controller.signal;
        const finish=(error,value)=>{signal.removeEventListener('abort',abort);worker.terminate();error?reject(error):resolve(value);};
        const abort=()=>finish(createAbortError('Contour superseded'));
        if(signal.aborted){abort();return;}
        signal.addEventListener('abort',abort,{once:true});
        worker.onerror=event=>finish(Error(event.message));
        worker.onmessage=event=>event.data.error?finish(Error(event.data.error)):finish(null,event.data);
        worker.postMessage({data,levels:calculateContourLineCount(cols,rows)});
    }).finally(()=>{AppState.fieldWorkerReservedBytes=Math.max(0,AppState.fieldWorkerReservedBytes-reservation);});
    if(typeof FieldMetrics!=='undefined')FieldMetrics.record('contour-worker',performance.now()-started,{step:context.step,result:context.resultPath});
    const canvas=document.createElement('canvas');canvas.width=cols;canvas.height=rows;
    canvas.getContext('2d').putImageData(new ImageData(result.pixels,cols,rows),0,0);
    return canvas;
}
