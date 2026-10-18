'use strict';
const MCPPanel=(()=>{
    let csrf=null,busy=false,activityBusy=false,observing=false;
    const openedAt=Date.now();
    const labels={read:'Read files & results',edit:'Edit YAML & files',images:'Import & process images',run:'Start analysis',cancel:'Stop analysis',delete:'Permanently delete',replace:'Replace existing files'};
    const el=id=>document.getElementById(id);
    const node=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;};
    const message=text=>{el('mcpMessage').textContent=text;};
    async function api(url,method='GET',body) {
        const response=await fetch(url,{method,headers:{'Content-Type':'application/json',...(csrf?{'X-Omfdm-Csrf':csrf}:{})},...(body===undefined?{}:{body:JSON.stringify(body)})});
        const result=await response.json();
        if(!response.ok){const e=Error(result.error||'Request failed');e.code=result.code;throw e;}
        return result;
    }
    function permissions(container,selected=['read']) {
        container.replaceChildren();
        for(const [value,label] of Object.entries(labels)) {
            const wrapper=node('label'),checkbox=node('input');checkbox.type='checkbox';checkbox.value=value;checkbox.checked=selected.includes(value);checkbox.disabled=value==='read';
            wrapper.append(checkbox,document.createTextNode(label));container.append(wrapper);
        }
    }
    const selected=container=>Array.from(container.querySelectorAll('input:checked')).map(e=>e.value);
    function button(text,action,style='btn-secondary') {
        const b=node('button',text);b.type='button';b.className=style;
        b.addEventListener('click',async()=>{b.disabled=true;try{await action();}catch(e){message(e.message);}finally{b.disabled=false;}});return b;
    }
    async function setup(id){el('mcpSetup').value=JSON.stringify(await api(`/api/mcp/setup/${id}`),null,2);}
    async function refresh() {
        if(busy)return;busy=true;
        try {
            const status=await api('/api/mcp/status');csrf=status.csrf;
            el('mcpPairSection').hidden=true;el('mcpManagement').hidden=false;
            el('mcpAssignedUser').textContent=AppState.userId;
            el('mcpEnabled').checked=status.enabled;el('mcpPort').value=status.port;
            el('mcpBrowserStatus').textContent=status.browser?.remember?`Browser trusted until ${new Date(status.browser.expires).toLocaleString()}`:'Browser paired for this session.';
            const solver=await api('/api/mcp/solver');
            el('mcpSolverStatus').textContent=`Solver: ${solver.available?'Found':'Not found'} (${solver.source||'unavailable'}) — ${solver.path||''}${solver.activePath&&solver.activePath!==solver.path?' — restart required':''}`;
            if(document.activeElement!==el('mcpSolverPath'))el('mcpSolverPath').value=solver.savedPath||'';
            el('mcpListenerStatus').textContent=status.error||`${status.listening?'Listening':'OFF'} · ${status.url}`;
            if(status.auditError||status.auditDropped)el('mcpListenerStatus').textContent+=` · Audit warning: ${status.auditError||''} ${status.auditDropped||0} records dropped`;
            const list=el('mcpConnections');list.replaceChildren();
            for(const c of status.connections) {
                const row=node('article');row.className='mcp-connection';row.append(node('strong',`${c.name} · ${c.user}${c.revoked?' · Revoked':''}`),node('p',`Last access: ${c.lastAccess||'not yet'}`));
                if(!c.revoked){const p=node('fieldset');p.className='mcp-permissions';permissions(p,c.permissions);row.append(p);const actions=node('div');actions.className='mcp-actions';
                    actions.append(button('Save permissions',async()=>{await api(`/api/mcp/connections/${c.id}`,'PATCH',{permissions:selected(p)});message('Permissions saved.');await refresh();}),
                        button('Settings',()=>setup(c.id)),button('Test',async()=>{const r=await api(`/api/mcp/connections/${c.id}/test`,'POST',{});message(`Connected to ${r.workspace} through ${r.transport}.`);}),
                        button('Revoke',async()=>{await api(`/api/mcp/connections/${c.id}`,'PATCH',{revoked:true});message('Connection revoked. Running analyses continue.');await refresh();},'btn-danger'));row.append(actions);}
                list.append(row);
            }
            if(!status.connections.length)list.append(node('p','No connections. Create a read-only connection first.'));
        }catch(e){if(e.code==='ADMIN_REQUIRED'){el('mcpPairSection').hidden=false;el('mcpManagement').hidden=true;}else message(e.message);}
        finally{busy=false;await activity();}
    }
    async function activity() {
        if(activityBusy||typeof AppState==='undefined'||!AppState.userId)return;activityBusy=true;
        try{
            const data=await api(`/api/automation/activity?userId=${encodeURIComponent(AppState.userId)}`);
            const container=el('mcpActivity');container.replaceChildren();
            for(const j of data.jobs.slice(0,20)) {
                const row=node('article');row.className='mcp-job';row.append(node('strong',`${j.source} · ${j.status} · ${j.progress}%`),node('p',j.resultId),node('p',j.message||''));
                if(['running','cancelling'].includes(j.status))row.append(button('Stop analysis',async()=>{await api(`/api/automation/jobs/${j.jobId}/cancel`,'POST',{userId:AppState.userId});await activity();},'btn-danger'));
                row.append(button('Open Results',async()=>{
                    await refreshResultsList();
                    const select=el('resultSelect'),folder=j.resultId.split('/')[1];
                    const option=Array.from(select.options).find(o=>o.value.split('/').pop()===folder);
                    if(!option){message('Result is not available yet. Refresh after the first output is written.');return;}
                    select.value=option.value;updateCustomSelectDisplay(option.value);switchTab('dashboard');await loadSelectedResult();
                }));container.append(row);
            }
            const list=node('ul');for(const e of data.events.slice(-20).reverse())list.append(node('li',`${e.at} · ${e.client} · ${e.operation} · ${e.target||''} · ${e.code}`));container.append(list);
            if(!data.jobs.length&&!data.events.length)container.append(node('p','No agent activity in this workspace.'));
            await observeChanges(data.events);
        }catch(e){el('mcpActivity').textContent=e.message;}finally{activityBusy=false;}
    }
    async function observeChanges(events) {
        if(observing)return;observing=true;
        try {
            const changes=[];
            for(const [category,name,revision,reload] of [
                ['configs',AppState.loadedConfigFile,AppState.configRevision,async()=>{el('configFileSelect').value=AppState.loadedConfigFile;switchTab('config');await loadConfig();}],
                ['libraries',AppState.currentLibraryFile,AppState.libraryRevision,async()=>{switchTab('config');await selectLibraryFile(AppState.currentLibraryFile);}]
            ]) {
                if(!name||!revision)continue;
                const result=await api(`/api/workspace/file?userId=${encodeURIComponent(AppState.userId)}&id=${encodeURIComponent(`${category}/${name}`)}`);
                if(result.file?.revision!==revision)changes.push({name,deleted:!result.file,reload});
            }
            const imageName=AppState.uploadedImageFilename;
            if(imageName&&events.some(e=>e.code==='OK'&&Date.parse(e.at)>Math.max(openedAt,AppState.imageSelectedAt||0)&&
                (e.target===`images/${imageName}`||e.operation==='backup_import')&&!['files_read','files_stat','image_info'].includes(e.operation)))
                changes.push({name:imageName,reload:async()=>{el('imageSelect').value=imageName;loadSelectedImage();}});
            const banner=el('mcpExternalChanges');banner.replaceChildren();banner.hidden=!changes.length;
            if(changes.length)banner.append(node('strong','Saved files changed externally. Your current edits and image are unchanged.'));
            for(const change of changes) {
                const row=node('div',`${change.name}${change.deleted?' was deleted.':' has a new saved version.'}`);
                if(!change.deleted)row.append(button('Reload',async()=>{await change.reload();await activity();}));
                banner.append(row);
            }
        } finally {observing=false;}
    }
    document.addEventListener('DOMContentLoaded',()=>{
        permissions(el('mcpPermissions'));
        el('mcpPairForm').addEventListener('submit',async e=>{e.preventDefault();try{const r=await api('/api/mcp/pair','POST',{code:el('mcpPairCode').value,remember:el('mcpRemember').checked});csrf=r.csrf;el('mcpPairCode').value='';message('Browser paired.');await refresh();}catch(e){message(e.message);}});
        el('mcpForgetBrowser').onclick=async()=>{try{await api('/api/mcp/forget-browser','POST',{});csrf=null;el('mcpToken').value='';el('mcpNewSecret').hidden=true;await refresh();message('Browser forgotten. Agent connections are unchanged.');}catch(e){message(e.message);}};
        el('mcpSolverForm').addEventListener('submit',async e=>{e.preventDefault();try{await api('/api/mcp/solver','PUT',{path:el('mcpSolverPath').value.trim()});await refresh();message('Solver setting saved. Restart the server when no analyses are running.');}catch(e){message(e.message);}});
        el('mcpSettingsForm').addEventListener('submit',async e=>{e.preventDefault();try{await api('/api/mcp/settings','PUT',{enabled:el('mcpEnabled').checked,port:Number(el('mcpPort').value)});message('Access settings saved.');await refresh();}catch(e){message(e.message);}});
        el('mcpCreateForm').addEventListener('submit',async e=>{e.preventDefault();try{const c=await api('/api/mcp/connections','POST',{name:el('mcpConnectionName').value,user:AppState.userId,permissions:selected(el('mcpPermissions'))});el('mcpToken').value=c.token;el('mcpNewSecret').hidden=false;await setup(c.id);await refresh();message('Connection created. Store the token now if using HTTP directly.');}catch(e){message(e.message);}});
        el('mcpHideSecret').onclick=()=>{el('mcpToken').value='';el('mcpNewSecret').hidden=true;};
        el('mcpCopySetup').onclick=async()=>{try{await navigator.clipboard.writeText(el('mcpSetup').value);message('Settings copied.');}catch{message('Select the settings and copy them manually.');}};
        el('mcpRefreshActivity').onclick=activity;
        setInterval(()=>{if(!document.hidden)void activity();},15000);
    });
    return {refresh,activity};
})();
