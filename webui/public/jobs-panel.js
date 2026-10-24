/* Server-owned jobs. Workspace changes and page closure never cancel a solve. */
const JobsPanel = (() => {
    let timer, selected=null, busy=false, stopped=false, user=null, offset=0, view='active', next=null;
    let resultsFingerprint=null, detailKey=null, currentActive=null;
    const active=new Set(['queued','held','preparing','running','cancelling']);
    const el=id=>document.getElementById(id);
    const text=(id,value)=>{if(el(id))el(id).textContent=value;};
    const node=(tag,value)=>{const n=document.createElement(tag);if(value!==undefined)n.textContent=value;return n;};
    async function request(url, options={}) {
        const response=await fetch(url,{...options,signal:AbortSignal.timeout(30000)}),data=await response.json();
        if(!response.ok||!data.success)throw Error(data.error||'Job service unavailable');
        return data;
    }
    const url=(suffix='')=>`/api/jobs?userId=${encodeURIComponent(AppState.userId)}${suffix}`;
    async function submit(input) {
        const storageKey=`omfdm.pendingJob.${AppState.userId}`,signature=JSON.stringify(input);
        let pending;try{pending=JSON.parse(localStorage.getItem(storageKey));}catch{}
        if(!pending||pending.signature!==signature)pending={signature,key:crypto.randomUUID()};
        try{localStorage.setItem(storageKey,JSON.stringify(pending));}catch{}
        const job=await request('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...input,idempotencyKey:pending.key})});
        try{localStorage.removeItem(storageKey);}catch{}
        selected=job.jobId;detailKey=null;
        showStatus('solverStatus',`Accepted: ${job.jobId} (${job.status}). Track it in Jobs.`,'success');
        await refresh();
    }
    async function cancel(id=currentActive||selected) {
        if(!id)return;
        await request(`/api/jobs/${encodeURIComponent(id)}?userId=${encodeURIComponent(AppState.userId)}`,{method:'DELETE'});
        await refresh();
    }
    async function resume() {
        await request('/api/jobs/resume',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({userId:AppState.userId})});
        await refresh();
    }
    function button(label,action) {
        const b=node('button',label);b.type='button';b.className='btn-secondary';
        b.onclick=async()=>{b.disabled=true;try{await action();}catch(e){showStatus('solverStatus',e.message,'error');text('jobsConnection',e.message);}finally{b.disabled=false;}};
        return b;
    }
    async function openResult(job) {
        const workspace=AppState.userId;
        await refreshResultsList({throwOnError:true});
        if(workspace!==AppState.userId)return;
        const select=el('resultSelect'),folder=job.resultId.split('/')[1];
        const option=Array.from(select.options).find(o=>o.value.split('/').pop()===folder);
        if(!option)throw Error('No field output is available for this job yet.');
        select.value=option.value;updateCustomSelectDisplay(option.value);switchTab('dashboard');await loadSelectedResult();
    }
    function renderPage(page) {
        const panel=el('analysisJobs');panel.replaceChildren();
        el('jobsEmpty').hidden=page.jobs.length>0;
        text('jobsEmpty',view==='active'?'No active jobs. Finished jobs are available in history.':'No finished jobs.');
        for(const j of page.jobs){
            const row=node('tr'),title=node('td');
            const label=button(j.configName||j.jobId.slice(0,8),()=>{selected=j.jobId;detailKey=null;return refresh();});
            label.title=j.jobId;title.append(label,node('small',j.jobId.slice(0,8)));row.append(title,node('td',j.source||'WebUI'),node('td',j.status),node('td',`${j.progress||0}%`),node('td',new Date(j.created).toLocaleString()));
            const actions=node('td');
            if(active.has(j.status))actions.append(button('Cancel',()=>cancel(j.jobId)));
            if(j.status==='held')actions.append(button('Resume queue',resume));
            if(!active.has(j.status))actions.append(button('Open results',()=>openResult(j)));
            row.append(actions);panel.append(row);
        }
        next=page.nextOffset;
        el('jobsPrevious').disabled=offset===0;el('jobsNext').disabled=next===null;
        text('jobsPage',page.total?`${offset+1}–${Math.min(offset+20,page.total)} of ${page.total}`:'0 jobs');
    }
    async function syncResults(jobs,workspace) {
        const finished=jobs.filter(j=>!active.has(j.status));
        const fingerprint=finished.map(j=>`${j.jobId}:${j.status}:${j.finished}`).sort().join('|');
        if(fingerprint===resultsFingerprint)return;
        if(!finished.length){resultsFingerprint=fingerprint;return;}
        try {
            await refreshResultsList({throwOnError:true});
            if(workspace!==AppState.userId)return;
            resultsFingerprint=fingerprint;
            text('jobsResultNotice','Results list updated. Open a job’s results from Jobs.');
        }catch(error){text('jobsResultNotice',`Results list refresh failed: ${error.message}. Retrying automatically.`);}
    }
    async function refresh() {
        if(busy||!AppState.userId)return;
        busy=true;
        try{
            const workspace=AppState.userId;
            if(user!==workspace){user=workspace;selected=null;offset=0;view='active';resultsFingerprint=null;detailKey=null;el('jobsFilter').value=view;text('solverOutput','');text('jobDetails','Select a job to inspect its status and log.');}
            const overview=await request(url('&view=overview'));
            if(workspace!==AppState.userId)return;
            text('jobsConnection','Connected');text('jobsWorkspace',workspace);
            const count=overview.counts;
            text('jobsOverview',count.active?`${count.active} active job${count.active===1?'':'s'}. Manage the queue in Jobs.`:'No active jobs.');
            text('jobsCounts',`${count.active} active · ${count.history} finished (history is hidden by default)`);
            let page=await request(url(`&view=${view}&offset=${offset}&limit=20`));
            if(workspace!==AppState.userId)return;
            if(offset&&offset>=page.total){offset=Math.max(0,Math.floor((page.total-1)/20)*20);page=await request(url(`&view=${view}&offset=${offset}&limit=20`));if(workspace!==AppState.userId)return;}
            renderPage(page);
            const running=overview.jobs.filter(j=>active.has(j.status));
            const current=running.find(j=>j.jobId===selected)||running[0];
            currentActive=current?.jobId||null;
            el('solverProgressContainer').style.display=current?'block':'none';
            el('stopBtn').style.display=current?'inline-block':'none';
            if(current){
                if(!selected)selected=current.jobId;
                const progress=Math.max(0,Math.min(100,current.progress||0));
                el('solverProgressBar').style.width=`${progress}%`;
                text('solverProgressText',`${current.status} · ${current.configName||current.jobId.slice(0,8)}`);
                text('solverProgressPercent',`${progress}%`);
            }
            if(selected){
                const job=await request(`/api/jobs/${encodeURIComponent(selected)}?userId=${encodeURIComponent(workspace)}`);
                if(workspace!==AppState.userId)return;
                const key=`${job.jobId}:${job.status}:${job.progress}:${job.logTail?.join('\n')}`;
                if(key!==detailKey){detailKey=key;text('jobDetails',`${job.configName||job.jobId} · ${job.source||'WebUI'} · ${job.status} · ${job.progress||0}%${job.message?' — '+job.message:''}`);text('solverOutput',(job.logTail||[]).join('\n'));}
            }
            await syncResults(overview.jobs,workspace);
        }catch(error){text('jobsConnection',`Disconnected: ${error.message}. Reconnecting…`);}
        finally{busy=false;}
    }
    async function poll(){await refresh();if(!stopped)timer=setTimeout(poll,document.hidden?8000:1500);}
    document.addEventListener('DOMContentLoaded',()=>{
        el('jobsFilter').onchange=()=>{view=el('jobsFilter').value;offset=0;void refresh();};
        el('jobsPrevious').onclick=()=>{offset=Math.max(0,offset-20);void refresh();};
        el('jobsNext').onclick=()=>{if(next!==null){offset=next;void refresh();}};
        void poll();
    });
    window.addEventListener('pagehide',()=>{stopped=true;clearTimeout(timer);});
    window.addEventListener('pageshow',event=>{if(event.persisted){stopped=false;void poll();}});
    return {submit,cancel,refresh,resume,openResult};
})();
