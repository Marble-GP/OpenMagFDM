/* Preserve the existing cell-threshold contour definition in a cancellable worker. */
function contourPixels(data, levels) {
    const rows=data.length,cols=data[0]?.length||0;
    if(rows<2||cols<2||rows*cols>2000000||!Number.isInteger(levels)||levels<2||levels>500)throw Error('Contour worker size limit exceeded');
    let min=Infinity,max=-Infinity;
    for(const row of data)for(const value of row)if(Number.isFinite(value)){min=Math.min(min,value);max=Math.max(max,value);}
    const pixels=new Uint8ClampedArray(rows*cols*4);
    for(let i=3;i<pixels.length;i+=4)pixels[i]=255;
    if(!Number.isFinite(min))return {pixels,rows,cols};
    for(let y=0;y<rows-1;y++)for(let x=0;x<cols-1;x++){
        const values=[data[y][x],data[y][x+1],data[y+1][x],data[y+1][x+1]];
        if(!values.every(Number.isFinite))continue;
        const lo=Math.min(...values),hi=Math.max(...values);
        for(let level=0;level<levels;level++){
            const value=min+(max-min)*level/(levels-1);
            if(value>=lo&&value<=hi){const at=(y*cols+x)*4;pixels[at]=pixels[at+1]=pixels[at+2]=255;break;}
        }
    }
    return {pixels,rows,cols};
}
if(typeof module!=='undefined')module.exports={contourPixels};
if(typeof self!=='undefined'&&typeof document==='undefined')self.onmessage=event=>{
    try {const result=contourPixels(event.data.data,event.data.levels);self.postMessage(result,[result.pixels.buffer]);}
    catch(error){self.postMessage({error:error.message});}
};
