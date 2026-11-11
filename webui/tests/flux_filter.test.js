'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const filter=require('../public/flux-filter');
const steps=n=>Array.from({length:n},(_,i)=>i);
const error=(values,truth)=>Math.sqrt(values.reduce((s,v,i)=>s+(v-truth[i])**2,0)/values.length);
test('flux CSV retains missing values and respects real step spacing',()=>{
    const data=filter.parseFlux('step,Phi_A,Phi_B\n0,0,\n2,4,NaN\n5,10,3\n');
    assert.deepEqual(data.phiSeries.Phi_B,[null,null,3]);
    const r=filter.series(data.steps,data.phiSeries.Phi_A);
    assert.deepEqual(r.x,[1,3.5]);assert.deepEqual(r.raw,[-2,-2]);
    assert.throws(()=>filter.parseFlux('step,Phi\n0,1\n0,2'),/increase/);
    assert.throws(()=>filter.parseFlux('step,Phi,Phi\n0,1,2'),/Duplicate/);
});
test('local fit reproduces cubic derivatives on irregular coordinates, including edges',()=>{
    const x=steps(60).map(i=>i+.08*Math.sin(i)),y=x.map(t=>.003*t**3-.1*t*t+2*t+1);
    const original=structuredClone({x,y});
    const r=filter.series(x,y,{mode:'local',window:9});
    r.estimated.forEach((v,i)=>assert.ok(Math.abs(v-(-.009*r.x[i]**2+.2*r.x[i]-2))<1e-10));
    assert.deepEqual({x,y},original);
});
test('local and repeating-state estimates reduce synthetic noise without changing raw data',()=>{
    const x=steps(288),w=2*Math.PI/72;
    const y=x.map(i=>Math.sin(w*i)+.01*Math.sin(1.723*i));
    const local=filter.series(x,y,{mode:'local',window:9}),periodic=filter.series(x,y,{mode:'periodic',window:9,period:72});
    const truth=local.x.map(t=>-w*Math.cos(w*t));
    assert.ok(error(local.estimated,truth)<error(local.raw,truth)*.2);
    assert.ok(error(periodic.estimated,truth)<error(periodic.raw,truth)*.02);
    assert.deepEqual(local.raw,periodic.raw);
    assert.equal(periodic.pooled,288);
    // Derivative remains centred in phase (no causal-filter lag).
    const quadrature=periodic.estimated.reduce((s,v,i)=>s+(v-truth[i])*Math.sin(w*periodic.x[i]),0);
    assert.ok(Math.abs(quadrature)<.001);
});
test('clean fifth harmonic remains within 5% amplitude at the default window',()=>{
    const x=steps(288),w=2*Math.PI/72,y=x.map(t=>Math.sin(w*t)+.15*Math.sin(5*w*t));
    const r=filter.series(x,y,{mode:'periodic',window:9,period:72});
    const amplitude=-2*r.estimated.reduce((s,v,i)=>s+v*Math.cos(5*w*r.x[i]),0)/r.x.length;
    assert.ok(Math.abs(amplitude/(.15*5*w)-1)<.05,`fifth-harmonic gain=${amplitude/(.15*5*w)}`);
});
test('quality metadata excludes non-convergence and missing status rows; gaps are not filled',()=>{
    const x=steps(40),y=x.map(i=>i<20?i:100+3*i),quality={};
    x.forEach(i=>quality[i]={converged:true,residual:1e-8});
    quality[19].converged=false;delete quality[20];
    const r=filter.series(x,y,{mode:'local',window:9},quality);
    assert.equal(r.excluded,2);
    for(const i of [18,19,20])assert.equal(r.estimated[i],null);
    for(const i of [5,10,15])assert.ok(Math.abs(r.estimated[i]+1)<1e-9);
    for(const i of [24,30,35])assert.ok(Math.abs(r.estimated[i]+3)<1e-9);
    assert.equal(r.smoothed[19],null);assert.equal(r.smoothed[20],null);
    assert.ok(Number.isFinite(r.raw[19]));
    const metadata=filter.parseStatus('step,nonlinear,strict_converged,state_reusable,iterations,residual\n0,0,1,1,1,\n1,1,0,0,200,0.2');
    assert.equal(metadata[0].residual,null);assert.equal(metadata[1].converged,false);
    assert.throws(()=>filter.parseStatus('step,other\n0,1'),/metadata/);
});
test('quality weights downweight an explicitly high-residual converged point',()=>{
    const x=steps(40),y=x.map(t=>t*.02);y[20]+=.1;
    const good=Object.fromEntries(x.map(t=>[t,{converged:true,residual:1e-9}]));
    const bad=structuredClone(good);bad[20].residual=.01;
    const a=filter.series(x,y,{mode:'local',window:9},good),b=filter.series(x,y,{mode:'local',window:9},bad);
    const truth=a.x.map(()=>-.02);
    assert.ok(error(b.estimated,truth)<error(a.estimated,truth));
});
test('period mismatch, insufficient cycles and sparse repeats do not fabricate an estimate',()=>{
    const x=steps(144),y=x.map(i=>Math.sin(i*2*Math.PI/72)+(i>=72?.8:0));
    const r=filter.series(x,y,{mode:'periodic',window:9,period:72});
    assert.match(r.note,/disagreement/);assert.ok(r.estimated.every(v=>v===null));
    const short=filter.series(x.slice(0,60),y.slice(0,60),{mode:'periodic',window:9,period:72});
    assert.match(short.note,/two complete/);
    assert.throws(()=>filter.options({mode:'local',window:8}),/odd/);
    assert.throws(()=>filter.options({mode:'periodic',period:5}),/period/);
    assert.throws(()=>filter.options({mode:'mystery'}),/Unknown/);
});
test('missing values stay null rather than becoming zero samples',()=>{
    const x=steps(20),y=x.map(i=>2*i);y[9]=null;y[10]=NaN;
    const r=filter.series(x,y,{mode:'local',window:9});
    assert.deepEqual(r.raw.slice(8,11),[null,null,null]);
    assert.deepEqual(r.estimated.slice(8,11),[null,null,null]);
    assert.equal(r.excluded,2);
});
