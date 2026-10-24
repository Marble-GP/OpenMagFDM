'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element {
    constructor(){this.children=[];this.style={};this.value='';this.textContent='';}
    append(...nodes){this.children.push(...nodes);}
    replaceChildren(...nodes){this.children=nodes;}
}
function fixture(){
    const ids={};const el=id=>ids[id]||=(new Element());
    let jobs=[],refreshes=0,failResults=false,ready;
    const requests=[],state={userId:'wa',currentTab:'run'},calls=[];
    const context=vm.createContext({
        AppState:state,AbortSignal,URL,console,localStorage:{getItem(){},setItem(){},removeItem(){}},crypto:{randomUUID:()=> 'retry-key'},
        document:{getElementById:el,createElement:()=>new Element(),addEventListener:(name,fn)=>{if(name==='DOMContentLoaded')ready=fn;},hidden:false},
        window:{addEventListener(){}},setTimeout(){},clearTimeout(){},
        fetch:async(url,options)=>{requests.push(url);const parsed=new URL(url,'http://localhost');
            if(parsed.pathname!=='/api/jobs')return {ok:true,json:async()=>({success:true,...jobs.find(j=>j.jobId===parsed.pathname.split('/').pop()),logTail:[]})};
            const active=j=>['running','queued','held'].includes(j.status),view=parsed.searchParams.get('view');
            const offset=Number(parsed.searchParams.get('offset')||0),all=view==='overview'?jobs:jobs.filter(j=>view==='history'?!active(j):active(j));
            return {ok:true,json:async()=>({success:true,jobs:all.slice(offset,offset+20),counts:{active:jobs.filter(active).length,history:jobs.filter(j=>!active(j)).length},total:all.length,nextOffset:offset+20<all.length?offset+20:null})};
        },
        refreshResultsList:async()=>{refreshes++;if(failResults)throw Error('offline');},
        showStatus:(...x)=>calls.push(x),switchTab:tab=>calls.push(tab),updateCustomSelectDisplay:value=>calls.push(value),loadSelectedResult:async()=>calls.push('loaded')
    });
    vm.runInContext(fs.readFileSync(require.resolve('../public/jobs-panel.js'),'utf8')+'\nthis.panel=JobsPanel;',context);
    return {context,ids,requests,state,calls,get refreshes(){return refreshes;},set jobs(v){jobs=v;},set failResults(v){failResults=v;},init:()=>ready()};
}
const job=(status='running')=>({jobId:'example-job',configName:'motor.yaml',source:'MCP agent',created:'2026-10-24T00:00:00Z',status,progress:status==='completed'?100:20,resultId:'results/run',finished:status==='completed'?'2026-10-24T00:01:00Z':null});
test('MCP job completion refreshes results without reload or replacing the loaded result',async()=>{
    const f=fixture();f.jobs=[job()];await f.context.panel.refresh();assert.equal(f.ids.analysisJobs.children.length,1);assert.equal(f.refreshes,0);
    f.jobs=[job('completed')];await f.context.panel.refresh();
    assert.equal(f.ids.analysisJobs.children.length,0);assert.equal(f.ids.solverProgressContainer.style.display,'none');
    assert.equal(f.refreshes,1);assert.deepEqual(f.calls,[]);
    await f.context.panel.refresh();assert.equal(f.refreshes,1,'unchanged history must not repeatedly fetch results');
});
test('initial finished history stays hidden and a failed auto-refresh is retried',async()=>{
    const f=fixture();f.jobs=[job('completed')];f.failResults=true;await f.context.panel.refresh();
    assert.equal(f.ids.analysisJobs.children.length,0);assert.match(f.ids.jobsResultNotice.textContent,/Retrying/);
    f.failResults=false;await f.context.panel.refresh();assert.equal(f.refreshes,2);assert.match(f.ids.jobsResultNotice.textContent,/updated/);
});
test('opening job results updates the picker and reports missing fields without a stale load',async()=>{
    const f=fixture();f.ids.resultSelect=new Element();f.ids.resultSelect.options=[{value:'outputs/wa/run'}];
    await f.context.panel.openResult(job('completed'));
    assert.equal(f.ids.resultSelect.value,'outputs/wa/run');assert.deepEqual(f.calls,['outputs/wa/run','dashboard','loaded']);
    f.ids.resultSelect.options=[];await assert.rejects(f.context.panel.openResult(job()),/No field output/);assert.equal(f.calls.length,3);
});
