'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const FluxFilter=require('../public/flux-filter');
const source=fs.readFileSync(require.resolve('../public/app.js'),'utf8');
function fixture(config={},metadataStatus=404){
    const state={plotConfigs:{p:{fluxFilter:{...config,resultPath:'results/a'}}}},captured={};
    const data={steps:Array.from({length:144},(_,i)=>i),phiNames:['Phi_A'],phiSeries:{Phi_A:Array.from({length:144},(_,i)=>Math.sin(i*Math.PI/36))}};
    const container={},controller=new AbortController(),renderContext={resultPath:'results/a',controller};
    const context=vm.createContext({FluxFilter,DOMException,Number,Promise,Worker:undefined,AppState:state,
        getCurrentResultPath:()=> 'results/a',document:{getElementById:()=>container},
        ensureDashboardRenderContext:()=>renderContext,assertRenderContextCurrent(){},isRenderContextCurrent:()=>!controller.signal.aborted,
        loadFluxLinkageData:async()=>data,fetch:async()=>({status:metadataStatus,ok:metadataStatus===200,text:async()=> 'bad quality header'}),
        fileManagerEscape:s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;'),fluxPhaseColor:()=> '#d62728',
        preparePlotlyContainer(){},getContainerSize:()=>({width:800,height:450}),
        newPlotForRender:async(c,t,l)=>{captured.traces=t;captured.layout=l;},
        showPlotMessage:(c,s)=>{captured.error=s;},discardStaleRender:()=>{captured.stale=true;}
    });
    vm.runInContext(source.slice(source.indexOf('function fluxFilterSettings('),source.indexOf('async function renderEnergyTime(')),context);
    return {context,captured,controller,state,data};
}
test('EMF defaults to raw and uses true midpoint and step difference',async()=>{
    const f=fixture();f.data.steps=[0,2,5];f.data.phiSeries.Phi_A=[0,4,10];
    await f.context.renderBackEMFTime('container-p',1);
    assert.deepEqual(Array.from(f.captured.traces[0].x),[2,4.5]);
    assert.deepEqual(Array.from(f.captured.traces[0].y),[-2,-2]);
    assert.match(f.captured.layout.yaxis.title,/per step/);assert.equal(f.captured.traces.length,1);
});
test('enabled estimate overlays raw data and reports missing convergence metadata',async()=>{
    const f=fixture({mode:'local',window:9});await f.context.renderBackEMFTime('container-p',1);
    assert.equal(f.captured.traces.length,2);assert.match(f.captured.traces[0].name,/raw/);assert.match(f.captured.traces[1].name,/estimate/);
    assert.equal(f.captured.traces[1].connectgaps,false);
    assert.match(f.captured.layout.annotations[0].text,/metadata unavailable/);
    await f.context.renderFluxLinkageTime('container-p',1);assert.equal(f.captured.traces[0].y,f.data.phiSeries.Phi_A);
});
test('metadata errors preserve raw plot and withhold estimates',async()=>{
    for(const status of [403,500,200]){
        const f=fixture({mode:'local',window:9},status);await f.context.renderBackEMFTime('container-p',1);
        assert.equal(f.captured.traces.length,1);assert.match(f.captured.layout.annotations[0].text,/Raw data retained/);
    }
});
test('period settings are scoped to result and controls explain state equivalence',()=>{
    const f=fixture({mode:'periodic',window:9,period:72});
    assert.equal(f.context.fluxFilterSettings('p','results/b').mode,'off');
    const html=f.context.fluxFilterControls('p');
    assert.match(html,/geometry, excitation and boundary conditions all repeat/);
    assert.match(html,/role="alert"/);assert.match(html,/value="72"/);
});
test('aborted filter work terminates its worker and cannot publish stale curves',async()=>{
    const f=fixture({mode:'local',window:9});let worker;
    f.context.Worker=class {constructor(){worker=this;}postMessage(){}terminate(){this.terminated=true;}};
    const result=f.context.calculateFluxEstimate(f.data,{mode:'local',window:9},null,f.controller.signal);
    f.controller.abort();await assert.rejects(result,{name:'AbortError'});assert.ok(worker.terminated);
});
test('page loads local filter module before app and worker uses the same tested implementation',()=>{
    const page=fs.readFileSync(require.resolve('../public/index.html'),'utf8');
    assert.ok(page.indexOf('src="flux-filter.js"')<page.indexOf('src="app.js"'));
    const worker=fs.readFileSync(require.resolve('../public/flux-filter-worker.js'),'utf8');
    assert.match(worker,/importScripts\('flux-filter.js'\)/);assert.match(worker,/FluxFilter.analyze/);
});
