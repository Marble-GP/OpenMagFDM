const NeighborPreload = (() => {
    let timer,controller;
    function cancel(){clearTimeout(timer);controller?.abort();controller=null;}
    function schedule(step){
        cancel();const result=getCurrentResultPath(),generation=AppState.fieldLoadGeneration;
        if(!result||AppState.fieldPreloadController)return;
        const own=new AbortController();controller=own;
        timer=setTimeout(async()=>{
            try {
                for(const next of [step+1,step-1])for(const field of ['Az','Mu']){
                    if(own.signal.aborted||generation!==AppState.fieldLoadGeneration||result!==getCurrentResultPath()||AppState.currentStep!==step)return;
                    if(next<1||next>AppState.totalSteps)continue;
                    const key=`${result}:${field}:${next}`,estimate=AppState.dataCacheSizes[`${result}:${field}:${step}`];
                    if(Object.hasOwn(AppState.dataCache,key)||!estimate)continue;
                    if(AppState.dataCacheBytes+estimate>AppState.dataCacheBudgetBytes)return;
                    let lease;
                    try {
                        lease=await loadFieldDataForPreload(field,next,result,own.signal);
                        if(own.signal.aborted)return;
                        if(!cacheFieldPairAtomically([{cacheKey:key,...lease.payload}]))return;
                        lease.markCached();
                    }finally{lease?.release();}
                }
            }catch(error){if(error.name!=='AbortError')console.info('Neighbor preload stopped:',error.message);}
            finally{if(controller===own)controller=null;}
        },300);
    }
    window.addEventListener('pagehide',cancel);
    return {schedule,cancel};
})();
