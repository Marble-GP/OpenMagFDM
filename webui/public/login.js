document.getElementById('loginForm').addEventListener('submit',async event=>{
    event.preventDefault();const form=event.currentTarget,button=form.querySelector('button');button.disabled=true;
    try {const response=await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:form.elements.name.value,password:form.elements.password.value,remember:form.elements.remember.checked})});const data=await response.json();if(!response.ok)throw Error(data.error);location.replace('/');}
    catch(error){document.getElementById('status').textContent=error.message;}finally{button.disabled=false;}
});
