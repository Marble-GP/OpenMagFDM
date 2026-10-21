const FieldMetrics = (() => {
    const records=[];let timer,stop=false,running=false,frame=0;
    function record(stage,ms,details={}) {records.push({at:new Date().toISOString(),stage,ms,...details});if(records.length>4000)records.shift();}
    function refresh(){
        const element=document.getElementById('fieldCacheStatus');if(!element)return;
        const prefix=AppState.resultsData.currentResult+':';
        const fallback=[...document.querySelectorAll('[data-gpu-fallback]')].filter(e=>e.getClientRects().length).map(e=>`${e.id}: ${e.dataset.gpuFallback}`).join('; ');
        let pairs=0,fields=0;
        for(let step=1;step<=AppState.totalSteps;step++){
            const az=Object.hasOwn(AppState.dataCache,prefix+'Az:'+step),mu=Object.hasOwn(AppState.dataCache,prefix+'Mu:'+step);
            pairs+=az&&mu?1:0;fields+=Number(az)+Number(mu);
        }
        element.textContent=`Az + Mu: ${pairs}/${AppState.totalSteps} steps (${fields} fields), ${formatFieldCacheBytes(AppState.dataCacheBytes)} / ${formatFieldCacheBytes(AppState.dataCacheBudgetBytes)} · ${AppState.preloadOutcome||'On demand'} · GPU allocation estimate ${formatFieldCacheBytes(typeof FieldGPU==='undefined'?0:FieldGPU.bytes)}${fallback?' · Plotly fallback: '+fallback:''}`;
    }
    function csv(){
        const keys=[...new Set(records.flatMap(r=>Object.keys(r)))];
        const quote=value=>'"'+String(value??'').replace(/"/g,'""').replace(/^([=+@-])/,'\t$1')+'"';
        const blob=new Blob(['\ufeff'+[keys,...records.map(r=>keys.map(k=>r[k]))].map(row=>row.map(quote).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});
        const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='openmagfdm-render-metrics.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }
    async function benchmark(){
        if(running||!AppState.resultsData.currentResult||!AppState.totalSteps)return;
        const minutes=Math.max(1,Math.min(30,Number(document.getElementById('benchmarkMinutes').value)||1));
        const start=performance.now(),result=AppState.resultsData.currentResult,renderer=document.getElementById('fieldRenderer').value;
        const controls=[...document.querySelectorAll('button,input,select,textarea')].filter(e=>e.id!=='benchmarkStop');
        const disabled=controls.map(e=>e.disabled);controls.forEach(e=>e.disabled=true);document.getElementById('benchmarkStop').disabled=false;
        running=true;stop=false;let intervals=[],previous=performance.now(),iteration=0;
        const sample=now=>{intervals.push(now-previous);previous=now;if(intervals.length>10000)intervals.shift();frame=requestAnimationFrame(sample);};frame=requestAnimationFrame(sample);
        try {
            while(!stop&&performance.now()-start<minutes*60000){
                if(document.hidden||AppState.resultsData.currentResult!==result||document.getElementById('fieldRenderer').value!==renderer)throw Error('Tab hidden, result changed, or renderer fell back');
                const step=iteration%AppState.totalSteps+1,t=performance.now();
                await setStep(step);await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
                const sorted=intervals.slice().sort((a,b)=>a-b),mean=intervals.reduce((a,b)=>a+b,0)/Math.max(1,intervals.length);
                const backends=[...document.querySelectorAll('[data-field-backend]')].filter(e=>e.getClientRects().length).map(e=>`${e.id}:${e.dataset.fieldBackend}${e.dataset.gpuFallback?' ('+e.dataset.gpuFallback+')':''}`).join(';');
                const errors=[...document.querySelectorAll('[data-pending-error]')].map(e=>e.dataset.pendingError).filter(Boolean);
                record('step-visible',performance.now()-t,{result,renderer,backends,step,iteration:iteration++,fps:mean?1000/mean:null,frameP95:sorted[Math.floor((sorted.length-1)*.95)]??null,cacheBytes:AppState.dataCacheBytes,gpuBytes:typeof FieldGPU==='undefined'?0:FieldGPU.bytes,heapBytes:performance.memory?.usedJSHeapSize??null,width:innerWidth,height:innerHeight,dpr:devicePixelRatio,userAgent:navigator.userAgent,error:errors.join(';')});intervals=[];
                document.getElementById('benchmarkStatus').textContent=`${iteration} steps · ${((performance.now()-start)/60000).toFixed(1)}/${minutes} min`;
                if(errors.length)throw Error('A plot failed; previous data was retained');
            }
        }catch(error){record('invalid',0,{error:error.message});document.getElementById('benchmarkStatus').textContent=error.message;}
        finally{cancelAnimationFrame(frame);running=false;controls.forEach((e,i)=>e.disabled=disabled[i]);document.getElementById('benchmarkStop').disabled=true;refresh();}
    }
    document.addEventListener('DOMContentLoaded',()=>{timer=setInterval(refresh,1000);});
    window.addEventListener('pagehide',()=>{clearInterval(timer);stop=true;cancelAnimationFrame(frame);});
    return {record,csv,benchmark,stop:()=>{stop=true;}};
})();
