'use strict';
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process'),{once}=require('node:events');
const {createFixture}=require('./console_fixture.cjs');
const {Jimp}=require('jimp');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function until(fn){const end=Date.now()+20000;while(Date.now()<end){const result=await fn();if(result)return result;await delay(50);}throw Error('Manual smoke timed out');}
async function main(){
 const index=process.argv.indexOf('--solver');assert.ok(index>=0&&process.argv[index+1],'Usage: node webui/tests/rest_manual_smoke.cjs --solver <absolute-native-solver>');
 const solver=path.resolve(process.argv[index+1]);process.chdir(path.resolve(__dirname,'../..'));
 const fixture=await createFixture({solver});
 const {base,root,env,port,executable,args}=fixture;
 const child=spawn(executable,[...args,'--console'],{cwd:base,env,windowsHide:true,stdio:['pipe','pipe','pipe']});let log='';
 child.stdout.on('data',b=>log+=b);child.stderr.on('data',b=>log+=b);
 const request=(url,options={})=>fetch(`http://127.0.0.1:${port}${url}`,{...options,signal:AbortSignal.timeout(5000)});
 const json=(method,url,body)=>request(url,{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 try{
  await until(()=>log.includes('Server ready:')&&log.includes('OpenMagFDM>'));
  assert.equal((await (await request('/api/auth/me')).json()).mode,'local');
  const manual=(await fs.readFile('docs/REST_API.md','utf8')).replace(/\r\n/g,'\n');
  const config=JSON.parse(manual.match(/```json\n(\{\n  "userId": "demo",\n  "file": "model.yaml",[\s\S]*?)\n```/)[1]);
  let response=await request('/api/config?userId=demo&file=model.yaml');assert.equal(response.status,200);assert.equal(response.headers.get('x-file-revision'),'');
  response=await json('POST','/api/config',config);assert.equal(response.status,200);const saved=await response.json();assert.match(saved.revision,/^[a-f0-9]{64}$/);
  assert.equal((await json('POST','/api/config',config)).status,409);
  response=await request('/api/config?userId=demo&file=model.yaml');assert.equal(response.headers.get('x-file-revision'),saved.revision);
  const form=new FormData();form.set('userId','demo');form.set('image',new Blob([await new Jimp({width:32,height:32,color:0xffffffff}).getBuffer('image/png')]),'model.png');
  assert.equal((await request('/api/upload-image',{method:'POST',body:form})).status,200);
  const input={userId:'demo',configFile:'model.yaml',imageFile:'model.png',idempotencyKey:'demo-run-001'};
  response=await json('POST','/api/jobs',input);assert.equal(response.status,202);const started=await response.json();
  response=await json('POST','/api/jobs',input);assert.equal(response.status,200);assert.equal((await response.json()).jobId,started.jobId);
  const job=await until(async()=>{const j=await(await request(`/api/jobs/${started.jobId}?userId=demo`)).json();if(j.status==='failed')throw Error(j.message);return j.status==='completed'&&j;});
  assert.equal(job.completed,true);assert.equal(job.converged,true);
  const result=job.resultPath,folder=job.resultId.split('/')[1];assert.ok(result);
  response=await request(`/api/load-field?result=${encodeURIComponent(result)}&file=Az%2Fstep_0001`);assert.equal(response.headers.get('content-type'),'image/tiff');await response.arrayBuffer();
  const desc=`/api/user-outputs/${folder}/description`;
  assert.equal((await(await request(desc+'?userId=demo')).json()).revision,null);
  assert.equal((await json('PUT',desc,{userId:'demo',description:'REST manual smoke',expectedRevision:null})).status,200);
  assert.equal((await(await request(desc+'?userId=demo')).json()).description,'REST manual smoke');
  response=await json('POST','/api/export',{userId:'demo',entries:['configs/model.yaml','images/model.png']});assert.equal(response.status,200);
  const archive=await response.arrayBuffer(),upload=new FormData();upload.set('userId','demo');upload.set('backup',new Blob([archive]),'manual.zip');
  response=await request('/api/import',{method:'POST',body:upload});assert.equal(response.status,200);const preview=await response.json();assert.equal(preview.files.length,2);
  response=await json('POST','/api/import/commit',{userId:'demo',transferId:preview.transferId,files:preview.files.map(f=>({id:f.id,revision:f.expectedRevision}))});assert.equal(response.status,200);
  const meta=await(await request('/api/workspace/file?userId=demo&id='+encodeURIComponent(job.resultId))).json();assert.ok(meta.file.revision);
  response=await json('POST','/api/file-operations',{userId:'demo',files:[{id:meta.file.id,revision:meta.file.revision}]});assert.equal(response.status,200);const operation=(await response.json()).operation;
  await until(async()=>{const o=(await(await request(`/api/file-operations/${operation.id}?userId=demo`)).json()).operation;if(o.status==='interrupted')throw Error(JSON.stringify(o));if(o.status!=='complete')return false;assert.equal(o.results[0].success,true,JSON.stringify(o));return o;});
  assert.equal((await(await request('/api/workspace/file?userId=demo&id='+encodeURIComponent(job.resultId))).json()).file,null);
  const files=(await(await request('/api/user-files?userId=demo')).json()).files;
  assert.equal(files.some(file=>file.category==='results'&&file.name===folder),false,'Result listing must follow the filesystem, not retained job history');
  assert.equal((await(await request(`/api/jobs/${started.jobId}?userId=demo`)).json()).status,'completed');
  console.log('PASS: documented JSON, revisions, upload, native job/idempotency, TIFF, description, ZIP preview/commit, completed-result deletion and filesystem-based result listing.');
 }finally{
  if(child.exitCode===null&&child.signalCode===null){const exit=once(child,'exit');child.stdin.write('exit --force\n');await exit;}
  assert.ok(path.basename(root).startsWith('omfdm-console-'));await fs.rm(root,{recursive:true,force:true,maxRetries:20,retryDelay:100});
 }
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
