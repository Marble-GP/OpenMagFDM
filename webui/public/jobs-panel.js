/* Server-owned jobs: closing this page never cancels a solve. */
const JobsPanel = (() => {
    let timer, selected = null, busy = false, stopped=false;
    const byId = id => document.getElementById(id);
    const active = new Set(['queued','held','preparing','running','cancelling']);
    async function request(url, options = {}) {
        const response = await fetch(url, {...options, signal:AbortSignal.timeout(30000)});
        const data = await response.json();
        if (!response.ok || !data.success) throw Error(data.error || 'Job service unavailable');
        return data;
    }
    async function submit(input) {
        const storageKey = `omfdm.pendingJob.${AppState.userId}`;
        const signature = JSON.stringify(input);
        let pending;
        try { pending = JSON.parse(localStorage.getItem(storageKey)); } catch {}
        if (!pending || pending.signature !== signature) pending = {signature, key:crypto.randomUUID()};
        try { localStorage.setItem(storageKey, JSON.stringify(pending)); } catch {}
        const job = await request('/api/jobs', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...input,idempotencyKey:pending.key})});
        try { localStorage.removeItem(storageKey); } catch {}
        selected = job.jobId;
        showStatus('solverStatus', `Accepted: ${job.jobId} (${job.status})`, 'success');
        await refresh();
    }
    async function cancel(id = selected) {
        if (!id) return;
        await request(`/api/jobs/${encodeURIComponent(id)}?userId=${encodeURIComponent(AppState.userId)}`,{method:'DELETE'});
        await refresh();
    }
    async function resume() {
        await request('/api/jobs/resume',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({userId:AppState.userId})});
        await refresh();
    }
    async function refresh() {
        if (busy || !AppState.userId) return;
        busy = true;
        try {
            const user = AppState.userId;
            const data = await request(`/api/jobs?userId=${encodeURIComponent(user)}`);
            if (user !== AppState.userId) return;
            byId('jobsConnection').textContent='Connected';
            const panel = byId('analysisJobs'); panel.replaceChildren();
            if (!data.jobs.length) panel.textContent = 'No jobs';
            for (const job of data.jobs.slice(0, 100)) {
                const row = document.createElement('div');
                const label = document.createElement('button'); label.className = 'btn-secondary';
                label.textContent = `${job.jobId.slice(0,8)} · ${job.status} · ${job.progress || 0}%`;
                label.onclick = () => { selected = job.jobId; void refresh(); }; row.append(label);
                if (active.has(job.status)) {
                    const button = document.createElement('button'); button.textContent='Cancel';
                    button.onclick = () => { button.disabled = true; void cancel(job.jobId).catch(e => showStatus('solverStatus',e.message,'error')); }; row.append(button);
                }
                if (job.status === 'held') {
                    const button = document.createElement('button'); button.textContent='Resume workspace queue';
                    button.onclick = () => { button.disabled=true; void resume().catch(e=>showStatus('solverStatus',e.message,'error')); }; row.append(button);
                }
                panel.append(row);
            }
            if (!selected || !data.jobs.some(j=>j.jobId===selected)) selected = data.jobs.find(j=>active.has(j.status))?.jobId || data.jobs[0]?.jobId;
            if (selected) {
                const job = await request(`/api/jobs/${selected}?userId=${encodeURIComponent(user)}`);
                if (user !== AppState.userId) return;
                byId('solverOutput').textContent = (job.logTail || []).join('\n');
                byId('solverProgressContainer').style.display='block';
                byId('solverProgressBar').style.width=`${Math.max(0,Math.min(100,job.progress || 0))}%`;
                byId('solverProgressText').textContent=`${job.status}${job.message ? ': '+job.message : ''}`;
                byId('solverProgressPercent').textContent=`${job.progress || 0}%`;
                byId('stopBtn').style.display=active.has(job.status)?'inline-block':'none';
            }
        } catch(error) { byId('jobsConnection').textContent=`Disconnected: ${error.message}. Reconnecting…`; }
        finally { busy=false; }
    }
    async function poll() { await refresh(); if(!stopped)timer=setTimeout(poll,document.hidden?8000:1500); }
    document.addEventListener('DOMContentLoaded', () => { void poll(); });
    window.addEventListener('pagehide',()=>{stopped=true;clearTimeout(timer);});
    window.addEventListener('pageshow',event=>{if(event.persisted){stopped=false;void poll();}});
    return {submit,cancel,refresh,resume};
})();
