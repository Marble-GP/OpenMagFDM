export function percentile(values, fraction) {
  if (!values.length) return null;
  const sorted=[...values].sort((a,b)=>a-b), at=(sorted.length-1)*fraction;
  const lo=Math.floor(at),hi=Math.ceil(at);
  return sorted[lo]+(sorted[hi]-sorted[lo])*(at-lo);
}
export function summarize(intervals, cpu, gpu) {
  return {
    frames:intervals.length,
    fps:intervals.length ? 1000*intervals.length/intervals.reduce((a,b)=>a+b,0) : null,
    frameMedianMs:percentile(intervals,.5),frameP95Ms:percentile(intervals,.95),
    cpuMedianMs:percentile(cpu,.5),gpuMedianMs:percentile(gpu,.5),gpuSamples:gpu.length,
  };
}
export function parsePitches(text) {
  const values=text.split(/[,\s]+/).filter(Boolean).map(Number);
  if(!values.length || values.length>12 || values.some(n=>!Number.isFinite(n)||n<.001||n>1000))
    throw Error('粒度リストは0.001〜1000 mmの数値を12個以内で指定してください');
  return values;
}
export function csv(rows) {
  if(!rows.length)return '';
  const fields=[...new Set(rows.flatMap(r=>Object.keys(r)))];
  const cell=value=>{
    let text=value==null?'':typeof value==='object'?JSON.stringify(value):String(value);
    if(/^[=+\-@]/.test(text))text="'"+text; // spreadsheet formula injection
    return '"'+text.replaceAll('"','""')+'"';
  };
  return '\ufeff'+fields.map(cell).join(',')+'\r\n'+rows.map(r=>fields.map(f=>cell(r[f])).join(',')).join('\r\n');
}
