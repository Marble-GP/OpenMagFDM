// Pure worker-side meshing. Cell labels are categorical UInt32, not interpolated colors.
const corners = [
  [[0,0,0],[0,0,1],[0,1,1],[0,1,0]],
  [[1,0,0],[1,1,0],[1,1,1],[1,0,1]],
  [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],
  [[0,1,0],[0,1,1],[1,1,1],[1,1,0]],
  [[0,0,0],[0,1,0],[1,1,0],[1,0,0]],
  [[0,0,1],[1,0,1],[1,1,1],[0,1,1]],
];
const normals = [[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]];
const triangles = [0,1,2,0,2,3];

function validate(ids, meta) {
  const dims = meta.dimensions_cells;
  if (!Array.isArray(dims) || dims.length !== 3 ||
      dims.some(n => !Number.isSafeInteger(n) || n < 1) ||
      dims.reduce((a,b) => a*b, 1) !== ids.length) throw Error('格子寸法とデータ長が一致しません');
  for (const name of ['origin_m', 'spacing_m']) {
    if (meta[name]?.length !== 3 || meta[name].some(x => !Number.isFinite(x))) throw Error('不正な格子座標です');
  }
  if (meta.spacing_m.some(x => x <= 0)) throw Error('格子間隔が正ではありません');
}

function eachFace(ids, dims, includeAir, visit) {
  const [nx,ny,nz] = dims, plane = nx*ny;
  for (let z=0; z<nz; z++) for (let y=0; y<ny; y++) for (let x=0; x<nx; x++) {
    const n = x+nx*y+plane*z, id = ids[n];
    if (!id && !includeAir) continue;
    // Keep interfaces between different labels, even inside the assembly.
    const neighbor = [x ? n-1 : -1, x+1<nx ? n+1 : -1,
      y ? n-nx : -1, y+1<ny ? n+nx : -1, z ? n-plane : -1, z+1<nz ? n+plane : -1];
    for (let face=0; face<6; face++) {
      if (neighbor[face] < 0 || ids[neighbor[face]] !== id) visit(id,x,y,z,face);
    }
  }
}

export function buildMesh(ids, meta, {mode='surface', includeAir=false, maxFaces=1500000, maxInstances=750000}={}) {
  validate(ids, meta);
  const counts = new Map();
  let faces = 0, instances = 0;
  if (mode === 'surface') {
    eachFace(ids,meta.dimensions_cells,includeAir,id => {
      if (++faces > maxFaces) throw Error(`表示面数が上限 ${maxFaces.toLocaleString()} を超えました。粒度を粗くしてください`);
      counts.set(id,(counts.get(id)||0)+1);
    });
  } else if (mode === 'instances') {
    for (const id of ids) if (id || includeAir) {
      if (++instances > maxInstances) throw Error(`全セル描画の上限 ${maxInstances.toLocaleString()} 個を超えました。表面抽出へ切り替えてください`);
      counts.set(id,(counts.get(id)||0)+1);
    }
  } else throw Error('不明な描画方式です');
  const groups = [...counts].sort((a,b)=>a[0]-b[0]).map(([id,count]) =>
    mode === 'surface' ? {id,count,positions:new Float32Array(count*18),normals:new Float32Array(count*18)} :
      {id,count,matrices:new Float32Array(count*16)});
  const map = new Map(groups.map(g=>[g.id,g]));
  const cursor = new Map();
  if (mode === 'surface') {
    eachFace(ids,meta.dimensions_cells,includeAir,(id,x,y,z,face)=>{
      const group=map.get(id), offset=(cursor.get(id)||0)*18, cell=[x,y,z];
      for (let v=0;v<6;v++) for (let axis=0;axis<3;axis++) {
        group.positions[offset+v*3+axis]=meta.origin_m[axis]+
          (cell[axis]+corners[face][triangles[v]][axis])*meta.spacing_m[axis];
        group.normals[offset+v*3+axis]=normals[face][axis];
      }
      cursor.set(id,(cursor.get(id)||0)+1);
    });
  } else {
    const [nx,ny,nz]=meta.dimensions_cells;
    for(let z=0;z<nz;z++) for(let y=0;y<ny;y++) for(let x=0;x<nx;x++){
      const id=ids[x+nx*(y+ny*z)];
      if(!id && !includeAir)continue;
      const a=map.get(id).matrices, offset=(cursor.get(id)||0)*16;
      a[offset]=meta.spacing_m[0];a[offset+5]=meta.spacing_m[1];a[offset+10]=meta.spacing_m[2];a[offset+15]=1;
      a[offset+12]=meta.origin_m[0]+(x+.5)*meta.spacing_m[0];
      a[offset+13]=meta.origin_m[1]+(y+.5)*meta.spacing_m[1];
      a[offset+14]=meta.origin_m[2]+(z+.5)*meta.spacing_m[2];
      cursor.set(id,(cursor.get(id)||0)+1);
    }
  }
  return {groups,faces,instances,triangles:mode==='surface'?faces*2:instances*12,
    attributeBytes:groups.reduce((sum,g)=>sum+(g.matrices?.byteLength||0)+(g.positions?.byteLength||0)+(g.normals?.byteLength||0),0)};
}
