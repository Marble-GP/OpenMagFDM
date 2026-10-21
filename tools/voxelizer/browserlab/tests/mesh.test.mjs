import test from 'node:test';
import assert from 'node:assert/strict';
import { buildMesh } from '../dist/mesh-core.js';
import { summarize, parsePitches, csv } from '../dist/metrics.js';

const meta=(dims)=>({dimensions_cells:dims,origin_m:[-.01,0,.02],spacing_m:[.001,.002,.003]});
test('one voxel has 6 outward quads, correct anisotropic coordinates',()=>{
  const m=meta([1,1,1]),result=buildMesh(new Uint32Array([1]),m);
  assert.equal(result.faces,6);assert.equal(result.triangles,12);
  const g=result.groups[0],center=m.origin_m.map((x,i)=>x+m.spacing_m[i]/2);
  for(let at=0;at<g.positions.length;at+=9){
    const a=[...g.positions.slice(at,at+3)],b=[...g.positions.slice(at+3,at+6)],c=[...g.positions.slice(at+6,at+9)];
    const u=b.map((x,i)=>x-a[i]),v=c.map((x,i)=>x-a[i]);
    const normal=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
    assert.ok(normal.reduce((s,n,i)=>s+n*(a[i]-center[i]),0)>0);
    assert.ok(normal.reduce((s,n,i)=>s+n*g.normals[at+i],0)>0);
  }
});
test('shared faces culled only within identical material; sparse IDs preserved',()=>{
  assert.equal(buildMesh(new Uint32Array([17,17]),meta([2,1,1])).faces,10);
  const result=buildMesh(new Uint32Array([17,4294967295]),meta([2,1,1]));
  assert.equal(result.faces,12);
  assert.deepEqual(result.groups.map(g=>g.id),[17,4294967295]);
});
test('air is a real cell region including cavity surfaces; input immutable',()=>{
  const ids=new Uint32Array(27);ids[13]=1;
  const original=ids.slice();
  assert.equal(buildMesh(ids,meta([3,3,3])).faces,6);
  const result=buildMesh(ids,meta([3,3,3]),{includeAir:true});
  assert.equal(result.groups[0].id,0);assert.equal(result.groups[0].count,60);
  assert.equal(result.groups[1].count,6);assert.deepEqual(ids,original);
});
test('instance matrices contain x-fast positions and XYZ size; limits before allocation',()=>{
  const ids=new Uint32Array([1,0,2,3]),m=meta([2,2,1]);
  const result=buildMesh(ids,m,{mode:'instances'});
  assert.equal(result.instances,3);assert.equal(result.triangles,36);
  const matrix=result.groups[1].matrices;
  assert.ok(Math.abs(matrix[12]-(m.origin_m[0]+.5*.001))<1e-8);
  assert.ok(Math.abs(matrix[13]-1.5*.002)<1e-8);
  assert.ok(Math.abs(matrix[10]-.003)<1e-8);
  assert.throws(()=>buildMesh(ids,m,{mode:'instances',maxInstances:2}),/上限/);
  assert.throws(()=>buildMesh(ids,m,{maxFaces:1}),/上限/);
  assert.throws(()=>buildMesh(ids,meta([1,1,1])),/一致/);
});
test('benchmark summaries do not invent GPU data; pitch order and CSV escaping',()=>{
  const result=summarize([10,20,30],[1,2],[]);
  assert.equal(result.fps,50);assert.equal(result.frameMedianMs,20);
  assert.equal(result.frameP95Ms,29);assert.equal(result.gpuMedianMs,null);
  assert.deepEqual(parsePitches('2, 1 .5, .25'),[2,1,.5,.25]);
  assert.throws(()=>parsePitches('NaN,.5'));
  assert.throws(()=>parsePitches('0'));
  const output=csv([{name:'=SUM(A1)',note:'a,"b"\nc',fps:50}]);
  assert.ok(output.includes("'=SUM(A1)"));assert.ok(output.includes('a,""b""\nc'));
});
