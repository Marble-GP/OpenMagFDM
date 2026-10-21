import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';
import { summarize, parsePitches, csv } from './metrics.js';

async function startLab(){
const $=id=>document.getElementById(id), number=(n,d=1)=>n==null?'—':Number(n).toLocaleString('ja-JP',{maximumFractionDigits:d});
const state={token:null,model:null,grid:null,busy:false,stop:false,job:null,worker:null,workerReject:null,
  rows:[],sampler:null,cadBytes:0,meshBytes:0,contextLost:false};
const viewport=$('viewport'),scene=new THREE.Scene();
scene.background=new THREE.Color('#f8fbfd');
const camera=new THREE.PerspectiveCamera(40,1,.000001,10);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
renderer.setPixelRatio(1);renderer.setClearColor('#f8fbfd');renderer.setSize(viewport.clientWidth,viewport.clientHeight);
viewport.prepend(renderer.domElement);
renderer.domElement.setAttribute('aria-label','3Dモデル表示。マウスドラッグで回転、ホイールで拡大縮小');
const controls=new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true;controls.dampingFactor=.08;controls.autoRotate=true;controls.autoRotateSpeed=2;
scene.add(new THREE.HemisphereLight(0xffffff,0x697b87,2.5));
const light=new THREE.DirectionalLight(0xffffff,2.1);light.position.set(1,-2,4);scene.add(light);
const cadGroup=new THREE.Group(),voxelGroup=new THREE.Group(),outlineGroup=new THREE.Group();
scene.add(cadGroup,voxelGroup,outlineGroup);
const gl=renderer.getContext(),timer=gl.getExtension('EXT_disjoint_timer_query_webgl2');
const debugRenderer=gl.getExtension('WEBGL_debug_renderer_info');
const pendingQueries=[];
let frameNumber=0,previousFrame=null,liveIntervals=[],lastLive=0,rotateBeforeMeasurement=true,longTasksSupported=false,animationHandle;

function status(text){$('status').textContent=text;}
function updateButtons(){
  for(const element of document.querySelectorAll('aside input, aside select, aside button, .benchcontrols input, .benchcontrols button'))
    element.disabled=state.busy||state.contextLost;
  $('generate').disabled=state.busy||!state.model||state.contextLost;
  $('sweep').disabled=state.busy||!state.model||state.contextLost;
  $('measure').disabled=state.busy||!state.model||state.contextLost;
  $('stop').disabled=!state.busy;
  $('export').disabled=!state.rows.length;
}
async function operation(fn){
  if(state.busy)return;
  state.busy=true;state.stop=false;updateButtons();
  try{await fn();}catch(error){status(state.stop?'停止しました。':`エラー：${error.message}`);}
  finally{state.busy=false;state.job=null;state.worker=null;state.workerReject=null;updateButtons();}
}
async function api(path,body,raw=false){
  const response=await fetch(path,{method:'POST',headers:{'X-Lab-Token':state.token,
    'Content-Type':raw?'application/octet-stream':'application/json'},
    body:raw?body:JSON.stringify(body??{})});
  const result=await response.json();
  if(!response.ok)throw Error(result.error||response.statusText);
  return result;
}
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function waitJob(id){
  state.job=id;
  while(true){
    const response=await fetch(`/api/jobs/${id}`),job=await response.json();
    if(!response.ok)throw Error(job.error);
    if(job.state==='complete'){state.job=null;if(state.stop)throw Error('停止');return job.result;}
    if(job.state!=='running'){state.job=null;throw Error(job.error||'停止');}
    status(job.stage);await sleep(250);
  }
}
function disposeGroup(group){
  const geometries=new Set(),materials=new Set();
  group.traverse(object=>{
    if(object.geometry)geometries.add(object.geometry);
    if(object.material)(Array.isArray(object.material)?object.material:[object.material]).forEach(m=>materials.add(m));
    if(object.isInstancedMesh)object.dispose();
  });
  geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());group.clear();
  renderer.renderLists.dispose();
}
function color(id){
  if(id===0)return 0x8fcfdf;
  const colors=[0x4d7b91,0x9eaab7,0xe68a51,0x87b4ad,0xaaa1cc,0xd1b369];
  return colors[(id-1)%colors.length];
}
function material(id){
  const m=new THREE.MeshStandardMaterial({color:color(id),roughness:.82,metalness:.05,
    side:THREE.DoubleSide,forceSinglePass:true});
  if(!id){m.polygonOffset=true;m.polygonOffsetFactor=1;m.polygonOffsetUnits=1;}
  return m;
}
function appearance(){
  $('solidValue').textContent=Number($('solidAlpha').value).toFixed(2);
  $('airValue').textContent=Number($('airAlpha').value).toFixed(2);
  const display=$('display').value;
  cadGroup.visible=display!=='voxel'||!state.grid;
  voxelGroup.visible=display!=='cad'&&!!state.grid;
  for(const group of [cadGroup,voxelGroup])group.traverse(object=>{
    if(!object.isMesh)return;
    const alpha=Number($(object.userData.id===0?'airAlpha':'solidAlpha').value);
    object.visible=alpha>0;
    object.material.opacity=alpha;object.material.transparent=alpha<1;object.material.depthWrite=alpha>=1;
    object.material.wireframe=$('wire').checked;object.material.needsUpdate=true;
  });
  $('viewName').textContent=!state.model?'3D WORKSPACE':
    !state.grid?'STEP / CAD PREVIEW':display==='cad'?'STEP / CAD PREVIEW':
      display==='both'?'STEP + VOXELS':`VOXELS / ${state.grid.pitchMm} mm`;
}
function currentBounds(){
  if(state.grid){
    const m=state.grid.manifest;
    return new THREE.Box3(new THREE.Vector3(...m.origin_m),
      new THREE.Vector3(...m.origin_m.map((v,a)=>v+m.spacing_m[a]*m.dimensions_cells[a])));
  }
  if(state.model)return new THREE.Box3(new THREE.Vector3(...state.model.bounds_m.min),new THREE.Vector3(...state.model.bounds_m.max));
  return new THREE.Box3(new THREE.Vector3(-1,-1,-1),new THREE.Vector3(1,1,1));
}
function fit(){
  const box=currentBounds(),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
  const radius=Math.max(size.x,size.y,size.z)*1.6/Math.min(1,camera.aspect);
  controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(.85,-1.15,1.1).multiplyScalar(radius));
  camera.up.set(0,0,1);camera.near=radius/10000;camera.far=radius*100;
  camera.updateProjectionMatrix();controls.minDistance=radius*.04;controls.maxDistance=radius*15;
  controls.update();controls.saveState();
  disposeGroup(outlineGroup);
  const helper=new THREE.Box3Helper(box,0xa6c5d0);outlineGroup.add(helper);
}
function resize(){
  const width=viewport.clientWidth,height=viewport.clientHeight;
  if(renderer.domElement.clientWidth!==width||renderer.domElement.clientHeight!==height)invalidate('画面サイズ変更');
  camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height);
}
new ResizeObserver(resize).observe(viewport);
function settings(){
  return {representation:$('mode').value,display:$('display').value,includeAir:$('air').checked,
    effectiveDisplay:state.grid?$('display').value:'cad',
    solidOpacity:Number($('solidAlpha').value),airOpacity:Number($('airAlpha').value),wireframe:$('wire').checked,
    pixelRatio:renderer.getPixelRatio(),canvasWidth:renderer.domElement.width,canvasHeight:renderer.domElement.height,
    cadTriangles:state.model?.cadTriangles??0,modelSha256:state.model?.sha256??null,threeVersion:THREE.REVISION,
    webglRenderer:debugRenderer?gl.getParameter(debugRenderer.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),
    cameraPosition:camera.position.toArray(),cameraTarget:controls.target.toArray(),
    userAgent:navigator.userAgent};
}
async function loadModel(sample,file){
  status('STEPを送信中…');
  if(file&&file.size>50*1024*1024)throw Error('STEPは50 MiB以内にしてください');
  const started=await (sample?api('/api/sample'):api('/api/models?name='+encodeURIComponent(file.name),file,true));
  const metadata=await waitJob(started.jobId);
  status('CAD表示用メッシュを読み込んでいます…');
  const response=await fetch(metadata.previewUrl);
  if(!response.ok)throw Error('CADプレビューを取得できません');
  const document=await response.json();
  if(state.stop)throw Error('停止');
  disposeGroup(cadGroup);disposeGroup(voxelGroup);disposeGroup(outlineGroup);
  state.grid=null;state.meshBytes=0;state.cadBytes=0;state.model=metadata;
  for(const body of document.solids){
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(body.positions,3));
    geometry.computeVertexNormals();
    state.cadBytes+=geometry.attributes.position.array.byteLength+geometry.attributes.normal.array.byteLength;
    const mesh=new THREE.Mesh(geometry,material(body.material_id));mesh.userData.id=body.material_id;cadGroup.add(mesh);
  }
  $('empty').hidden=true;$('downloads').hidden=true;
  $('modelInfo').textContent=`${metadata.name} / ${metadata.solids}部品 / CAD ${number(metadata.cadTriangles,0)}三角形`;
  appearance();fit();status(`STEP読込完了：${number(metadata.importMs,0)} ms。粒度を指定して生成できます。`);
}
async function buildGrid(result){
  status('材料IDを転送し、Web Workerで描画メッシュを生成中…');
  disposeGroup(voxelGroup);state.meshBytes=0;state.grid=null;$('downloads').hidden=true;appearance();
  const start=performance.now(),response=await fetch(result.gridUrl);
  if(!response.ok)throw Error('格子データを取得できません');
  const buffer=await response.arrayBuffer(),transferMs=performance.now()-start;
  if(buffer.byteLength!==result.manifest.cell_count*4)throw Error('格子データ長が一致しません');
  if(state.stop)throw Error('停止');
  const mode=$('mode').value,includeAir=$('air').checked;
  const output=await new Promise((resolve,reject)=>{
    const worker=new Worker('/mesh-worker.js',{type:'module'});
    state.worker=worker;state.workerReject=reject;
    worker.onmessage=({data})=>{
      worker.terminate();state.worker=null;state.workerReject=null;
      data.error?reject(Error(data.error)):resolve(data);
    };
    worker.onerror=event=>{worker.terminate();state.worker=null;state.workerReject=null;reject(Error(event.message));};
    worker.postMessage({buffer,manifest:result.manifest,options:{mode,includeAir}},[buffer]);
  });
  if(state.stop)throw Error('停止');
  const meshStart=performance.now();
  for(const group of output.groups){
    let mesh;
    if(mode==='surface'){
      const geometry=new THREE.BufferGeometry();
      geometry.setAttribute('position',new THREE.BufferAttribute(group.positions,3));
      geometry.setAttribute('normal',new THREE.BufferAttribute(group.normals,3));
      geometry.computeBoundingSphere();
      mesh=new THREE.Mesh(geometry,material(group.id));
    }else{
      mesh=new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1),material(group.id),group.count);
      mesh.instanceMatrix.array.set(group.matrices);mesh.instanceMatrix.needsUpdate=true;mesh.computeBoundingSphere();
    }
    mesh.userData.id=group.id;voxelGroup.add(mesh);
  }
  state.meshBytes=output.attributeBytes;
  state.grid={...result,meshMode:mode,meshIncludesAir:includeAir,
    transferMs,workerMs:output.workerMs,meshSetupMs:performance.now()-meshStart,
    geometryFaces:output.faces,instances:output.instances,geometryTriangles:output.triangles};
  appearance();fit();
  const first=performance.now();renderer.render(scene,camera);
  state.grid.firstRenderCpuMs=performance.now()-first;
  $('downloadVti').href=result.vtiUrl;$('downloadManifest').href=result.manifestUrl;
  $('downloadVti').setAttribute('download',`voxels-${result.pitchMm}mm.vti`);
  $('downloadManifest').setAttribute('download',`conditions-${result.pitchMm}mm.json`);
  $('downloads').hidden=false;
  status(`${result.pitchMm} mm：${number(result.manifest.cell_count,0)}セル / 描画用 ${number(output.triangles,0)}三角形`);
}
async function generate(pitch=Number($('pitch').value)){
  if(!state.model)throw Error('STEPを読み込んでください');
  const margin=Number($('margin').value);
  const job=await api('/api/generate',{modelId:state.model.modelId,pitchMm:pitch,marginMm:margin});
  const result=await waitJob(job.jobId);
  await buildGrid(result);
}

function clearQueries(){
  for(const pending of pendingQueries)gl.deleteQuery(pending.query);
  pendingQueries.length=0;
}
function pollQueries(){
  if(!timer)return;
  if(gl.getParameter(timer.GPU_DISJOINT_EXT)){
    for(const p of pendingQueries){p.sampler.gpuInvalid=true;p.sampler.gpu.length=0;}
    if(state.sampler){state.sampler.gpuInvalid=true;state.sampler.gpu.length=0;}
    clearQueries();return;
  }
  for(let i=pendingQueries.length-1;i>=0;i--){
    const p=pendingQueries[i];
    if(gl.getQueryParameter(p.query,gl.QUERY_RESULT_AVAILABLE)){
      const ns=gl.getQueryParameter(p.query,gl.QUERY_RESULT);
      if(!p.sampler.gpuInvalid)p.sampler.gpu.push(ns/1e6);
      gl.deleteQuery(p.query);pendingQueries.splice(i,1);
    }
  }
}
function invalidate(reason){
  if(state.sampler){state.sampler.invalid=reason;state.sampler.finish();}
}
document.addEventListener('visibilitychange',()=>{if(document.hidden)invalidate('タブが非表示');});
renderer.domElement.addEventListener('webglcontextlost',event=>{
  event.preventDefault();state.contextLost=true;invalidate('WebGLコンテキスト喪失');clearQueries();
  status('WebGLが停止しました。ページを再読込し、粗い粒度から試してください。');updateButtons();
});
try{
  new PerformanceObserver(list=>{
    const s=state.sampler;if(!s)return;
    for(const entry of list.getEntries())if(entry.startTime>=s.start+1000&&entry.startTime<s.end){
      s.longTasks++;s.longTaskMs+=entry.duration;
    }
  }).observe({type:'longtask',buffered:false});
  longTasksSupported=true;
}catch{/* Browser does not expose long tasks. */}

function animation(now){
  animationHandle=requestAnimationFrame(animation);
  if(state.contextLost)return;
  pollQueries();
  const delta=previousFrame==null?0:now-previousFrame;previousFrame=now;
  controls.autoRotate=state.sampler?true:$('rotate').checked;
  controls.update(state.sampler?delta/1000:Math.min(delta/1000,.1));
  const sampler=state.sampler,measuring=sampler&&now>=sampler.start+1000&&now<sampler.end&&!sampler.invalid;
  let query=null;
  if(measuring&&timer&&frameNumber%4===0&&pendingQueries.length<8&&!sampler.gpuInvalid){
    query=gl.createQuery();gl.beginQuery(timer.TIME_ELAPSED_EXT,query);
  }
  const cpuStart=performance.now();renderer.render(scene,camera);const cpu=performance.now()-cpuStart;
  if(query){gl.endQuery(timer.TIME_ELAPSED_EXT);pendingQueries.push({query,sampler});}
  // Include the interval crossing the deadline, so a long final frame is not hidden.
  if(sampler&&now>=sampler.start+1000&&!sampler.invalid){
    if(sampler.previous!=null)sampler.intervals.push(now-sampler.previous);
    sampler.previous=now;if(measuring)sampler.cpu.push(cpu);
  }
  if(sampler&&now>=sampler.end)sampler.finish();
  if(delta>0&&delta<1000)liveIntervals.push(delta);
  if(now-lastLive>600){
    const fps=liveIntervals.length?1000*liveIntervals.length/liveIntervals.reduce((a,b)=>a+b,0):null;
    $('liveFps').textContent=number(fps,0);
    $('liveCounts').textContent=`${number(renderer.info.render.triangles,0)} triangles / ${renderer.info.render.calls} draw calls`;
    liveIntervals=[];lastLive=now;
  }
  frameNumber++;
}
animationHandle=requestAnimationFrame(animation);
async function measure(){
  if(document.hidden)throw Error('計測するタブを前面にしてください');
  fit();controls.enabled=false;rotateBeforeMeasurement=$('rotate').checked;
  const condition=settings(),grid=state.grid;
  clearQueries();
  const sample={start:performance.now(),previous:null,intervals:[],cpu:[],gpu:[],invalid:null,
    gpuInvalid:false,longTasks:0,longTaskMs:0};
  sample.end=sample.start+4000;
  status('計測中：1秒ウォームアップ → 3秒の自動回転。タブと画面サイズを維持してください。');
  await new Promise(resolve=>{
    let finished=false;
    sample.finish=()=>{
      if(finished)return;finished=true;state.sampler=null;resolve();
    };
    state.sampler=sample;
  });
  // Give asynchronous GPU queries a bounded time to return; never gl.finish().
  await sleep(150);pollQueries();clearQueries();controls.enabled=true;$('rotate').checked=rotateBeforeMeasurement;
  const summaries=summarize(sample.intervals,sample.cpu,sample.gpuInvalid?[]:sample.gpu);
  const row={timestamp:new Date().toISOString(),...condition,
    pitchMm:grid?.pitchMm??null,dimensions:grid?.manifest.dimensions_cells??null,
    requestedMarginMm:grid?.requestedMarginMm??null,paddingCells:grid?.manifest.padding_cells??null,
    originM:grid?.manifest.origin_m??null,spacingM:grid?.manifest.spacing_m??null,
    totalCells:grid?.manifest.cell_count??null,occupiedCells:grid?.manifest.occupied_cells??null,
    airCells:grid?.manifest.air_cells??null,nativeMs:grid?.nativeMs??null,
    transferMs:grid?.transferMs??null,workerMs:grid?.workerMs??null,meshSetupMs:grid?.meshSetupMs??null,
    firstRenderCpuMs:grid?.firstRenderCpuMs??null,triangles:renderer.info.render.triangles,
    drawCalls:renderer.info.render.calls,voxelAttributeBytes:state.meshBytes,
    residentAttributeBytesEstimate:state.meshBytes+state.cadBytes,
    jsHeapBytes:performance.memory?.usedJSHeapSize??null,
    longTasks:longTasksSupported?sample.longTasks:null,longTaskMs:longTasksSupported?sample.longTaskMs:null,gpuTimerAvailable:!!timer,
    measuredIntervalMs:sample.intervals.reduce((sum,ms)=>sum+ms,0),
    gpuDisjoint:sample.gpuInvalid,warmupMs:1000,targetMeasureMs:3000,...summaries,
    status:sample.invalid||'valid'};
  if(sample.intervals.length<2&&!sample.invalid)row.status='計測フレーム不足';
  addRow(row);status(row.status==='valid'?`計測完了：${number(row.fps)} FPS / p95 ${number(row.frameP95Ms)} ms`:`計測無効：${row.status}`);
  return row;
}
function addRow(row){
  state.rows.push(row);$('noResults')?.remove();
  const tr=document.createElement('tr'),values=[
    number(row.pitchMm,3),`${row.effectiveDisplay==='cad'?'STEP':row.representation==='surface'?'表面':'全セル'} / 空気${row.includeAir?'あり':'なし'}`,
    `${number(row.totalCells,0)} / ${number(row.occupiedCells,0)}`,
    `${number(row.triangles,0)} / ${number(row.drawCalls,0)}`,number(row.nativeMs),number(row.workerMs),
    number(row.fps),number(row.frameP95Ms),`${number(row.cpuMedianMs,2)} / ${number(row.gpuMedianMs,2)}`,
    row.status==='valid'?'有効':row.status];
  values.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===9)td.className=row.status==='valid'?'good':'bad';tr.append(td);});
  tr.title=`${row.representation} / air=${row.includeAir} / opacity=${row.solidOpacity} / DPR=${row.pixelRatio}`;
  $('results').append(tr);$('export').disabled=false;
}

$('sample').onclick=()=>operation(()=>loadModel(true));
$('file').onchange=()=>{const file=$('file').files[0];if(file)operation(()=>loadModel(false,file));};
$('generate').onclick=()=>operation(()=>generate());
$('measure').onclick=()=>operation(()=>measure());
$('sweep').onclick=()=>operation(async()=>{
  const pitches=parsePitches($('pitches').value);
  if($('display').value==='cad'){$('display').value='voxel';appearance();}
  for(const pitch of pitches){
    if(state.stop)break;
    try{
      $('pitch').value=pitch;await generate(pitch);
      const row=await measure();if(row.status!=='valid')break;
    }catch(error){
      addRow({timestamp:new Date().toISOString(),...settings(),pitchMm:pitch,status:state.stop?'ユーザー停止':error.message});
      throw error;
    }
  }
});
$('reverse').onclick=()=>{try{$('pitches').value=parsePitches($('pitches').value).reverse().join(', ');}catch(e){status(e.message);}};
for(const id of ['mode','air'])$(id).onchange=()=>operation(async()=>{
  if(state.grid)await buildGrid(state.grid);
});
for(const id of ['solidAlpha','airAlpha','wire','display'])$(id).oninput=appearance;
$('dpr').onchange=()=>{renderer.setPixelRatio(Number($('dpr').value));resize();};
$('reset').onclick=fit;
$('stop').onclick=async()=>{
  state.stop=true;
  if(state.job)await api(`/api/jobs/${state.job}/cancel`).catch(()=>{});
  if(state.worker){state.worker.terminate();state.worker=null;state.workerReject?.(Error('停止'));state.workerReject=null;}
  invalidate('ユーザー停止');
  status('停止処理中…');
};
$('export').onclick=()=>{
  const url=URL.createObjectURL(new Blob([csv(state.rows)],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download=`voxel-benchmark-${Date.now()}.csv`;a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
window.addEventListener('pagehide',()=>{
  cancelAnimationFrame(animationHandle);
  state.worker?.terminate();clearQueries();disposeGroup(cadGroup);disposeGroup(voxelGroup);disposeGroup(outlineGroup);
  controls.dispose();renderer.dispose();
});
window.addEventListener('pageshow',event=>{if(event.persisted)location.reload();});
try{
  const response=await fetch('/api/status');if(!response.ok)throw Error('ローカルサーバーへ接続できません');
  const info=await response.json();state.token=info.token;
  $('connection').textContent='LOCAL · READY';status('STEPを選択するか、サンプルを読み込んでください。');updateButtons();
}catch(error){$('connection').textContent='接続エラー';status(error.message);}
}
startLab().catch(error=>{
  document.getElementById('connection').textContent='初期化エラー';
  document.getElementById('status').textContent=`3D初期化に失敗しました：${error.message}。WebGL 2が使えるブラウザーで開いてください。`;
  for(const element of document.querySelectorAll('button, input, select'))element.disabled=true;
});
