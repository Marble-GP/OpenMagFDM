/* Display-only flux estimation. Never modifies solver results. */
(function(root,factory){
    const api=factory();
    if(typeof module==='object'&&module.exports)module.exports=api;
    else root.FluxFilter=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
    'use strict';
    const finite=v=>typeof v==='number'&&Number.isFinite(v);
    const number=s=>typeof s==='string'&&s.trim()!==''&&Number.isFinite(Number(s))?Number(s):null;
    const median=a=>{const b=a.slice().sort((x,y)=>x-y),n=b.length;return n?(b[(n-1)>>1]+b[n>>1])/2:0;};
    const rms=a=>a.length?Math.sqrt(a.reduce((s,v)=>s+v*v,0)/a.length):null;
    function parseFlux(text){
        const lines=text.trim().split(/\r?\n/).filter(l=>l.trim()&&!l.startsWith('#'));
        const headers=(lines.shift()||'').split(',').map(s=>s.trim());
        if(headers[0]?.toLowerCase()!=='step'||headers.length<2)throw Error('Invalid flux CSV header.');
        const phiNames=headers.slice(1),steps=[],phiSeries=Object.fromEntries(phiNames.map(n=>[n,[]]));
        if(new Set(phiNames).size!==phiNames.length||phiNames.some(n=>!n))throw Error('Duplicate or empty flux column.');
        for(const line of lines){
            const cells=line.split(','),step=number(cells[0]);
            if(step===null)throw Error('Invalid flux step coordinate.');
            if(steps.length&&step<=steps[steps.length-1])throw Error('Flux steps must increase strictly.');
            steps.push(step);phiNames.forEach((n,k)=>phiSeries[n].push(number(cells[k+1])));
        }
        return {headers,phiNames,steps,phiSeries};
    }
    function parseStatus(text){
        const lines=text.trim().split(/\r?\n/).filter(l=>l.trim()&&!l.startsWith('#'));
        const h=(lines.shift()||'').split(',').map(s=>s.trim()),at=n=>h.indexOf(n);
        if(at('step')<0||at('strict_converged')<0)throw Error('Invalid flux quality metadata.');
        const result={};
        for(const line of lines){
            const c=line.split(','),step=number(c[at('step')]),ok=c[at('strict_converged')]?.trim();
            if(step===null||!['0','1'].includes(ok)||Object.hasOwn(result,step))throw Error('Invalid flux quality row.');
            result[step]={converged:ok==='1',residual:number(c[at('residual')])};
        }
        return result;
    }
    function options(value={}){
        const mode=value.mode||'off',window=Number(value.window??9),period=Number(value.period??0);
        if(!['off','local','periodic'].includes(mode))throw Error('Unknown flux filter mode.');
        if(!Number.isInteger(window)||window<5||window>51||window%2!==1)throw Error('Fit window must be an odd number from 5 to 51.');
        if(mode==='periodic'&&(!Number.isInteger(period)||period<window+1))throw Error('Full-state period must be an integer larger than the fit window.');
        return {mode,window,period};
    }
    // Weighted cubic least squares using reorthogonalized QR, in a local,
    // dimensionless coordinate. No normal equations or large global powers.
    function polynomial(points,target,weights,scale){
        const n=points.length,p=4,q=[],r=Array.from({length:p},()=>Array(p).fill(0));
        const y=points.map((v,i)=>Math.sqrt(weights[i])*v.y);
        for(let k=0;k<p;k++){
            const v=points.map((v,i)=>Math.sqrt(weights[i])*((v.x-target)/scale)**k);
            for(let pass=0;pass<2;pass++)for(let j=0;j<k;j++){
                const dot=v.reduce((s,a,i)=>s+a*q[j][i],0);r[j][k]+=dot;
                for(let i=0;i<n;i++)v[i]-=dot*q[j][i];
            }
            r[k][k]=Math.sqrt(v.reduce((s,a)=>s+a*a,0));
            if(r[k][k]<1e-10)return null;
            q.push(v.map(a=>a/r[k][k]));
        }
        const b=q.map(v=>v.reduce((s,a,i)=>s+a*y[i],0)),c=Array(p).fill(0);
        for(let k=p-1;k>=0;k--)c[k]=(b[k]-r[k].slice(k+1).reduce((s,a,j)=>s+a*c[k+1+j],0))/r[k][k];
        return c;
    }
    function fitAt(points,target,window,maxGap){
        let lo=0,hi=points.length;
        while(lo<hi){const m=(lo+hi)>>1;if(points[m].x<target)lo=m+1;else hi=m;}
        if(lo>0&&lo<points.length&&points[lo].x-points[lo-1].x>maxGap)return null;
        let l=lo-1,r=lo,left=true,right=true;const selected=[];
        while(selected.length<window&&(left||right)){
            if(l<0||(l+1<points.length&&points[l+1].x-points[l].x>maxGap))left=false;
            if(r>=points.length||(r>0&&points[r].x-points[r-1].x>maxGap))right=false;
            if(!left&&!right)break;
            if(left&&(!right||Math.abs(points[l].x-target)<=Math.abs(points[r].x-target)))selected.push(points[l--]);
            else selected.push(points[r++]);
        }
        if(selected.length<5)return null;
        const scale=Math.max(...selected.map(v=>Math.abs(v.x-target)));
        if(!(scale>0))return null;
        const base=selected.map(v=>v.w*(1-(Math.abs(v.x-target)/(scale*1.1))**3)**3);
        let weights=base,c=null;
        const floor=1e-12*Math.max(1e-12,...selected.map(v=>Math.abs(v.y)));
        for(let iteration=0;iteration<4;iteration++){
            c=polynomial(selected,target,weights,scale);if(!c)return null;
            const residuals=selected.map(v=>{const u=(v.x-target)/scale;return v.y-(c[0]+u*(c[1]+u*(c[2]+u*c[3])));});
            const centre=median(residuals),mad=1.4826*median(residuals.map(v=>Math.abs(v-centre)));
            if(mad<=floor)break;
            // Huber adaptation reduces isolated deviations without declaring
            // them solver errors or assigning a statistical confidence interval.
            weights=base.map((w,i)=>w*Math.max(.05,Math.min(1,1.5*mad/Math.max(floor,Math.abs(residuals[i]-centre)))));
        }
        return {value:c[0],derivative:-c[1]/scale,count:selected.length};
    }
    function series(steps,values,settings={},status=null){
        const o=options(settings),n=steps.length;
        if(n!==values.length||(o.mode!=='off'&&n>100000))throw Error('Invalid or oversized flux series (maximum 100,000 filtered points).');
        for(let i=0;i<n;i++)if(!finite(steps[i])||(i&&steps[i]<=steps[i-1]))throw Error('Flux steps must increase strictly.');
        const spacing=median(steps.slice(1).map((v,i)=>v-steps[i]));
        const x=steps.slice(1).map((v,i)=>(v+steps[i])/2);
        const raw=steps.slice(1).map((v,i)=>finite(values[i])&&finite(values[i+1])?-(values[i+1]-values[i])/(v-steps[i]):null);
        const result={x,raw,estimated:raw.map(()=>null),smoothed:values.map(()=>null),excluded:0,pooled:0,qualityKnown:status!==null,differenceRms:null,note:''};
        if(o.mode==='off')return result;
        if(n<5){result.note='At least five samples are required for an estimate.';return result;}
        const usable=steps.map((s,i)=>finite(values[i])&&(status===null||status[s]?.converged===true));
        result.excluded=usable.filter(v=>!v).length;
        const positiveResiduals=steps.filter((s,i)=>usable[i]&&finite(status?.[s]?.residual)&&status[s].residual>0).map(s=>status[s].residual);
        const reference=median(positiveResiduals);
        const weights=steps.map(s=>{const r=status?.[s]?.residual;return reference>0&&finite(r)&&r>=0?Math.max(.1,1/(1+(r/reference)**2)):1;});
        let points=steps.flatMap((s,i)=>usable[i]?[{x:s,y:values[i],w:weights[i]}]:[]);
        const phase=s=>((s-steps[0])%o.period+o.period)%o.period;
        if(o.mode==='periodic'){
            if(steps[n-1]-steps[0]+spacing<2*o.period){result.note='Need at least two complete full-state periods; no periodic estimate shown.';return result;}
            const groups=new Map();
            for(const p of points){const key=Number(phase(p.x).toFixed(9));if(!groups.has(key))groups.set(key,[]);groups.get(key).push(p);}
            const pooled=[],deviations=[];
            for(const [p,group]of groups){
                if(group.length<2)continue;
                const centre=median(group.map(v=>v.y)),mad=1.4826*median(group.map(v=>Math.abs(v.y-centre)));
                const floor=1e-12*Math.max(1e-12,Math.abs(centre));
                let sw=0,sy=0;
                for(const v of group){const w=v.w*(mad>floor?Math.min(1,1.5*mad/Math.max(floor,Math.abs(v.y-centre))):1);sw+=w;sy+=w*v.y;deviations.push(v.y-centre);}
                pooled.push({x:p,y:sy/sw,w:sw/group.length});result.pooled+=group.length;
            }
            const bounds=points.reduce((b,p)=>[Math.min(b[0],p.y),Math.max(b[1],p.y)],[Infinity,-Infinity]);
            const span=points.length?bounds[1]-bounds[0]:0;
            const disagreement=rms(deviations)||0;
            if(disagreement>Math.max(1e-12,span*.05)){
                result.note='Cycle disagreement exceeds 5% of flux range; repeating-state estimate withheld.';return result;
            }
            pooled.sort((a,b)=>a.x-b.x);
            if(pooled.length<=o.window){result.note='Not enough repeated phase positions for the selected window; no estimate shown.';return result;}
            points=[-1,0,1].flatMap(k=>pooled.map(p=>({...p,x:p.x+k*o.period})));
        }
        const evaluate=t=>fitAt(points,o.mode==='periodic'?phase(t):t,o.window,spacing*1.5);
        for(let i=0;i<n;i++)if(usable[i])result.smoothed[i]=evaluate(steps[i])?.value??null;
        for(let i=0;i<x.length;i++)if(usable[i]&&usable[i+1]&&steps[i+1]-steps[i]<=spacing*1.5)
            result.estimated[i]=evaluate(x[i])?.derivative??null;
        result.differenceRms=rms(result.estimated.flatMap((v,i)=>finite(v)&&finite(raw[i])?[v-raw[i]]:[]));
        result.note=result.estimated.some(finite)?'Estimated from flux; changes may include real harmonics.':'Insufficient contiguous converged samples; no estimate shown.';
        return result;
    }
    function analyze(data,settings,status){
        if(settings.mode!=='off'&&(data.phiNames.length>64||data.steps.length*data.phiNames.length>500000))throw Error('Filter limit: 64 series and 500,000 total samples.');
        return Object.fromEntries(data.phiNames.map(name=>[name,series(data.steps,data.phiSeries[name],settings,status)]));
    }
    return {parseFlux,parseStatus,options,series,analyze};
});
