/* Result annotations and display preferences; no field data is retained here. */
const ResultImprovements = (() => {
    let generation = 0, current = null;
    const el = id => document.getElementById(id);
    async function description(folder) {
        const token = ++generation;
        current = null;
        el('resultMemoEditor').hidden = false;
        el('resultMemoTitle').textContent = folder;
        el('resultOriginalInput').href = `/api/get-original-input?result=${encodeURIComponent('outputs/'+AppState.userId+'/'+folder)}`;
        el('resultOriginalInput').textContent = 'Original input image (initial state)';
        el('resultMemoText').value = '';
        el('resultMemoText').disabled = true;
        el('resultMemoSave').disabled = true;
        el('resultMemoStatus').textContent = 'Loading description…';
        try {
            const response = await fetch(`/api/user-outputs/${encodeURIComponent(folder)}/description?userId=${encodeURIComponent(AppState.userId)}`);
            const data = await response.json();
            if (!response.ok || !data.success) throw Error(data.error || 'Description load failed');
            if (token !== generation) return;
            current = {folder, user: AppState.userId, revision: data.revision};
            el('resultMemoText').value = data.description;
            el('resultMemoText').disabled = false;
            el('resultMemoSave').disabled = false;
            el('resultMemoStatus').textContent = '';
            const image=await fetch(el('resultOriginalInput').href,{method:'HEAD'});
            if(token===generation&&!image.ok){
                el('resultOriginalInput').removeAttribute('href');
                el('resultOriginalInput').textContent=image.status===404?'Original input was not saved for this result.':'Original input could not be retrieved.';
            }
        } catch (error) { if (token === generation) el('resultMemoStatus').textContent = error.message; }
    }
    async function save() {
        if (!current) return;
        const target = current, token = generation;
        el('resultMemoSave').disabled = true;
        el('resultMemoStatus').textContent = 'Saving…';
        try {
            const response = await fetch(`/api/user-outputs/${encodeURIComponent(target.folder)}/description`, {
                method: 'PUT', headers: {'Content-Type':'application/json'},
                body: JSON.stringify({userId:target.user, description:el('resultMemoText').value, expectedRevision:target.revision})
            });
            const data = await response.json();
            if (!response.ok || !data.success) throw Error(data.error || 'Save failed; reload the description before retrying');
            if (token !== generation) return;
            target.revision = data.revision;
            el('resultMemoStatus').textContent = 'Saved';
            void refreshResultsList();
        } catch (error) { if (token === generation) el('resultMemoStatus').textContent = error.message; }
        finally { if (token === generation) el('resultMemoSave').disabled = false; }
    }
    function aspect(checked) {
        AppState.polarAspectLocked = checked;
        try { localStorage.setItem('omfdm.polarAspectLocked', String(checked)); } catch {}
        refreshAllPlots();
    }
    document.addEventListener('DOMContentLoaded', () => {
        try { AppState.polarAspectLocked = localStorage.getItem('omfdm.polarAspectLocked') !== 'false'; } catch {}
        el('polarAspectLocked').checked = AppState.polarAspectLocked;
    });
    return {description, save, aspect};
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
