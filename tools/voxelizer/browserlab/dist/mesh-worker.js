import { buildMesh } from './mesh-core.js';
self.onmessage = ({data}) => {
  try {
    const start=performance.now();
    const result=buildMesh(new Uint32Array(data.buffer),data.manifest,data.options);
    const transfer=result.groups.flatMap(g=>g.matrices?[g.matrices.buffer]:[g.positions.buffer,g.normals.buffer]);
    self.postMessage({...result,workerMs:performance.now()-start},transfer);
  } catch(error) { self.postMessage({error:error.message}); }
};
