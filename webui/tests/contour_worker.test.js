const test=require('node:test');
const assert=require('node:assert/strict');
const {contourPixels}=require('../public/contour-worker');
test('worker contours match the legacy threshold-cell definition for finite values',()=>{
    const data=[[0,1,2,4],[1,3,-2,8],[2,5,7,10]],levels=10;
    const actual=contourPixels(data,levels),expected=new Set();
    for(let k=0;k<levels;k++){
        const level=-2+12*k/(levels-1);
        for(let y=0;y<2;y++)for(let x=0;x<3;x++){
            const v=[data[y][x],data[y][x+1],data[y+1][x],data[y+1][x+1]];
            if(level>=Math.min(...v)&&level<=Math.max(...v))expected.add(`${x},${y}`);
        }
    }
    for(let y=0;y<3;y++)for(let x=0;x<4;x++)assert.equal(actual.pixels[(y*4+x)*4],expected.has(`${x},${y}`)?255:0);
});
test('constant fields, missing cells, and resource guards are deterministic',()=>{
    assert.equal(contourPixels([[1,1],[1,1]],10).pixels[0],255);
    assert.equal(contourPixels([[1,null],[2,3]],10).pixels[0],0);
    assert.throws(()=>contourPixels([[1]],10),/limit/);
    assert.throws(()=>contourPixels([[1,2],[3,4]],501),/limit/);
});
