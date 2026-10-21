const ApiSession = (() => {
    const originalFetch = window.fetch.bind(window);
    let identity = null;
    const ready = originalFetch('/api/auth/me').then(async response=>{
        if(response.status===401){location.replace('/login.html');throw Error('Login required');}
        if(!response.ok)throw Error('Session unavailable');
        identity=await response.json();
        if(identity.mode==='shared'){
            let workspace;try{workspace=sessionStorage.getItem('omfdm.workspace');}catch{}
            if(!identity.workspaces.includes(workspace))workspace=identity.workspaces[0];
            identity.workspace=workspace || null;
        }
        return identity;
    });
    window.fetch=async(input,options={})=>{
        const session=await ready;
        const url=new URL(typeof input==='string'?input:input.url,location.href);
        if(url.origin===location.origin&&session.mode==='shared'&&!['GET','HEAD','OPTIONS'].includes((options.method||input.method||'GET').toUpperCase())){
            const headers=new Headers(options.headers||input.headers);headers.set('X-CSRF-Token',session.csrf);options={...options,headers};
        }
        const response=await originalFetch(input,options);
        if(response.status===401&&session.mode==='shared')location.replace('/login.html');
        return response;
    };
    document.addEventListener('DOMContentLoaded',async()=>{
        const session=await ready;if(session.mode!=='shared')return;
        const bar=document.createElement('div');bar.style.cssText='padding:8px;background:#e9f0f6';
        const label=document.createElement('span');label.textContent=`${session.name} (${session.role}) `;bar.append(label);
        const select=document.createElement('select');for(const workspace of session.workspaces){const option=document.createElement('option');option.value=workspace;option.textContent=workspace;select.append(option);}select.value=session.workspace;
        select.onchange=()=>{sessionStorage.setItem('omfdm.workspace',select.value);location.reload();};bar.append(select);
        const logout=document.createElement('button');logout.textContent='Logout';logout.onclick=async()=>{await fetch('/api/auth/logout',{method:'POST'});location.replace('/login.html');};bar.append(logout);
        const change=document.createElement('button');change.textContent='Change password';bar.append(change);
        const form=document.createElement('form');form.hidden=true;
        const inputs=[];
        for(const [labelText,autocomplete] of [['Current password','current-password'],['New password (12+ characters)','new-password'],['Repeat new password','new-password']]){
            const label=document.createElement('label');label.textContent=labelText+' ';
            const input=document.createElement('input');input.type='password';input.autocomplete=autocomplete;input.required=true;label.append(input);form.append(label);inputs.push(input);
        }
        const save=document.createElement('button');save.type='submit';save.textContent='Save and sign out';form.append(save);
        const status=document.createElement('span');status.setAttribute('role','status');form.append(status);bar.append(form);
        change.onclick=()=>{form.hidden=!form.hidden;if(!form.hidden)inputs[0].focus();};
        form.onsubmit=async event=>{
            event.preventDefault();if(inputs[1].value!==inputs[2].value){status.textContent='Passwords differ';return;}
            save.disabled=true;status.textContent='Saving…';
            try {
                const response=await fetch('/api/auth/password',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({currentPassword:inputs[0].value,password:inputs[1].value})});
                const data=await response.json();if(!response.ok)throw Error(data.error||'Password change failed');
                inputs.forEach(input=>input.value='');location.replace('/login.html');
            }catch(error){status.textContent=error.message;}finally{save.disabled=false;}
        };
        if(!session.workspace){const info=document.createElement('strong');info.textContent=' No workspace assigned. Ask the administrator.';bar.append(info);}
        document.body.prepend(bar);
    });
    return {ready,get identity(){return identity;}};
})();
