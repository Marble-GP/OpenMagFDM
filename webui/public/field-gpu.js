/* Opt-in WebGL2 scalar viewer. One context, bounded textures, original doubles retained. */
const FieldGPU = (() => {
    const views=new Map(), budget=64*1024*1024;
    let canvas,gl,program,frame=0,unavailable=false;
    const vertex=`#version 300 es
    in vec2 position;out vec2 uv;
    void main(){uv=(position+1.0)*0.5;gl_Position=vec4(position,0.,1.);}`;
    const fragment=`#version 300 es
    precision highp float;precision highp int;
    in vec2 uv;out vec4 color;uniform sampler2D field;
    uniform ivec2 dims;uniform vec4 domain;uniform vec4 sourceDomain;uniform vec2 limits;
    uniform vec3 polar;uniform int transformed;uniform int horizontal;
    uniform int fullModel;uniform int repeats;uniform int periodic;uniform int anti;uniform int harmonic;uniform int imageMode;
    float sampleAt(int t,int r){return texelFetch(field,horizontal==1?ivec2(r,t):ivec2(t,r),0).r;}
    vec3 viridis(float t){
      vec3 c[10]=vec3[10](vec3(68,1,84),vec3(72,40,120),vec3(62,73,137),vec3(49,104,142),vec3(38,130,142),vec3(31,158,137),vec3(53,183,121),vec3(110,206,88),vec3(181,222,43),vec3(253,231,37));
      float p=clamp(t,0.,1.)*9.;int i=min(8,int(floor(p)));return mix(c[i],c[i+1],p-float(i))/255.;
    }
    void main(){
      float value;
      if(transformed==1){
        vec2 xy=mix(domain.xz,domain.yw,uv);float r=length(xy);float theta=atan(xy.y,xy.x);if(theta<0.)theta+=6.28318530718;
        if(r<polar.x||r>polar.y)discard;
        if(fullModel==1)theta=mod(theta,6.28318530718/float(repeats));else if(theta>polar.z)discard;
        int nr=horizontal==1?dims.x:dims.y,nt=horizontal==1?dims.y:dims.x;
        float ri=clamp((r-polar.x)/(polar.y-polar.x)*float(nr-1),0.,float(nr-1));
        float ti=theta/polar.z*float(nt-1);
        if(imageMode==1){ivec2 cell=horizontal==1?ivec2(int(floor(ri)),int(floor(ti))):ivec2(int(floor(ti)),int(floor(ri)));color=texelFetch(field,clamp(cell,ivec2(0),dims-1),0);if(color.a==0.)discard;return;}
        ri=min(ri,float(nr)-1.001);
        if(periodic==1)ti=mod(ti,float(nt));else ti=clamp(ti,0.,float(nt)-1.001);
        int t0=int(floor(ti)),t1=periodic==1?(t0+1)%nt:min(t0+1,nt-1),r0=int(floor(ri)),r1=min(r0+1,nr-1);
        vec4 v=vec4(sampleAt(t0,r0),sampleAt(t0,r1),sampleAt(t1,r0),sampleAt(t1,r1));
        if(anti==1&&t1<t0)v.zw=-v.zw;
        if(any(isnan(v))||any(isinf(v)))discard;
        vec2 f=vec2(fract(ri),fract(ti));vec4 w=vec4((1.-f.y)*(1.-f.x),(1.-f.y)*f.x,f.y*(1.-f.x),f.y*f.x);
        if(harmonic==1){v=max(abs(v),vec4(1e-10));value=(anti==1&&t1<t0?-1.:1.)/dot(w,1./v);}else value=dot(w,v);
      }else{
        vec2 xy=mix(domain.xz,domain.yw,uv);vec2 p=(xy-sourceDomain.xz)/(sourceDomain.yw-sourceDomain.xz);if(any(lessThan(p,vec2(0.)))||any(greaterThan(p,vec2(1.))))discard;ivec2 cell=ivec2(floor(p*vec2(dims)));if(imageMode==1){color=texelFetch(field,clamp(cell,ivec2(0),dims-1),0);if(color.a==0.)discard;return;}value=texelFetch(field,clamp(cell,ivec2(0),dims-1),0).r;
        if(isnan(value)||isinf(value))discard;
      }
      color=vec4(viridis((value-limits.x)/max(limits.y-limits.x,1e-30)),1.);
    }`;
    function init(){
        if(unavailable)return false;if(gl)return true;
        canvas=document.createElement('canvas');canvas.style.cssText='position:fixed;inset:0;pointer-events:none;z-index:2';
        gl=canvas.getContext('webgl2',{alpha:true,antialias:false,preserveDrawingBuffer:false});
        if(!gl){unavailable=true;return false;}
        const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
        try{
            const vs=shader(gl.VERTEX_SHADER,vertex),fs=shader(gl.FRAGMENT_SHADER,fragment);
            program=gl.createProgram();gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);
            gl.deleteShader(vs);gl.deleteShader(fs);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
            gl.useProgram(program);const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
            const loc=gl.getAttribLocation(program,'position');gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
            document.body.append(canvas);
            canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();unavailable=true;for(const container of [...views.keys()])remove(container);gl=null;canvas.remove();document.getElementById('fieldRenderer').value='legacy';refreshAllPlots();});
            window.addEventListener('resize',schedule);document.addEventListener('scroll',schedule,true);
            return true;
        }catch(error){console.warn('GPU renderer unavailable',error);unavailable=true;gl=null;canvas.remove();return false;}
    }
    const uniform=(name)=>gl.getUniformLocation(program,name);
    function bounds(view){
        const outer=view.container.getBoundingClientRect();let left=outer.left+55,top=outer.top+26,width=outer.width-95,height=outer.height-65;
        if(view.polarView.cartesianTransform&&view.polarView.aspectLocked!==false){const side=Math.min(width,height);left+=(width-side)/2;top+=(height-side)/2;width=height=side;}
        return {left,top,width,height};
    }
    function schedule(){if(!frame)frame=requestAnimationFrame(draw);}
    function draw(){
        frame=0;if(!gl)return;
        const dpr=Math.min(devicePixelRatio||1,2),w=Math.floor(innerWidth*dpr),h=Math.floor(innerHeight*dpr);
        if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;canvas.style.width=innerWidth+'px';canvas.style.height=innerHeight+'px';}
        gl.disable(gl.SCISSOR_TEST);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.enable(gl.SCISSOR_TEST);gl.useProgram(program);
        for(const [container,view] of views){
            if(!container.isConnected){remove(container);continue;}if(!container.getClientRects().length)continue;
            const b=bounds(view);if(b.width<=0||b.height<=0)continue;
            const x=Math.round(b.left*dpr),y=Math.round(h-(b.top+b.height)*dpr),vw=Math.round(b.width*dpr),vh=Math.round(b.height*dpr);
            let left=Math.max(0,b.left),right=Math.min(innerWidth,b.left+b.width),top=Math.max(0,b.top),bottom=Math.min(innerHeight,b.top+b.height);
            for(let parent=container.parentElement;parent;parent=parent.parentElement){
                const style=getComputedStyle(parent),rect=parent.getBoundingClientRect();
                if(style.overflowX!=='visible'){left=Math.max(left,rect.left);right=Math.min(right,rect.right);}
                if(style.overflowY!=='visible'){top=Math.max(top,rect.top);bottom=Math.min(bottom,rect.bottom);}
            }
            gl.viewport(x,y,vw,vh);gl.scissor(Math.round(left*dpr),Math.round(h-bottom*dpr),Math.max(0,Math.round((right-left)*dpr)),Math.max(0,Math.round((bottom-top)*dpr)));
            gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,view.texture);gl.uniform1i(uniform('field'),0);gl.uniform2i(uniform('dims'),view.cols,view.rows);
            gl.uniform4fv(uniform('sourceDomain'),view.initial);gl.uniform4fv(uniform('domain'),view.domain);gl.uniform2fv(uniform('limits'),view.limits);gl.uniform3fv(uniform('polar'),view.polar);
            for(const [key,value] of Object.entries(view.flags))gl.uniform1i(uniform(key),value);
            gl.drawArrays(gl.TRIANGLES,0,6);
        }
    }
    function remove(container){const view=views.get(container);if(view){if(gl)gl.deleteTexture(view.texture);view.observer?.disconnect();container.onwheel=null;container.onpointermove=null;container.onpointerdown=null;container.ondblclick=null;views.delete(container);schedule();}}
    function render(container,data,title,physical,harmonic,context){
        const choice=document.getElementById('fieldRenderer')?.value;
        // Automatic stays on the validated legacy path until browser acceptance passes.
        if(choice!=='gpu') {delete container.dataset.gpuFallback;return false;}
        assertRenderContextCurrent(context,container);
        const fallback=reason=>{container.dataset.gpuFallback=reason;return false;};
        if(container._plotlyPending)return fallback('Waiting for an earlier Plotly commit');
        const c=context?.analysisConditions||AppState.analysisConditions||{},p=context?.polarView||snapshotPolarView(c);
        const config=AppState.plotConfigs[container.id.replace('container-','')]||{};
        const rows=data.length,cols=data[0]?.length||0;
        if(rows<2||cols<2||(config.colorscale&&config.colorscale!=='Viridis')||config.xRange&&config.xRange!=='auto'||config.yRange&&config.yRange!=='auto')return fallback('Unsupported dimensions, color scale or fixed axes');
        if(!init())return fallback('WebGL2 unavailable');
        if(cols>gl.getParameter(gl.MAX_TEXTURE_SIZE)||rows>gl.getParameter(gl.MAX_TEXTURE_SIZE))return fallback('Texture dimensions exceed device limit');
        const bytes=rows*cols*4,resident=rows*cols*28+rows*64,used=[...views.values()].reduce((sum,v)=>sum+v.resident,0)-(views.get(container)?.resident||0);
        if(resident+used>budget)return fallback('GPU memory budget');
        let min=Infinity,max=-Infinity;const buffer=new Float32Array(rows*cols);
        for(let r=0;r<rows;r++)for(let col=0;col<cols;col++) {const value=data[r][col];buffer[r*cols+col]=value===null?NaN:value;if(Number.isFinite(value)){min=Math.min(min,value);max=Math.max(max,value);}}
        if(!Number.isFinite(min)||!Number.isFinite(max)||Math.abs(min)>3e38||Math.abs(max)>3e38)return fallback('No finite Float32 display range');
        const ri=(c.polar?.r_start||c.r_i||0)*1000,ro=(c.polar?.r_end||c.r_o||1)*1000,angle=c.polar?.theta_range||c.theta_range||0;
        if(p.cartesianTransform&&(!(ro>ri)||angle<=0))return fallback('Invalid polar domain');
        const horizontal=c.polar?.r_orientation!=='vertical';
        const bc=c.boundary_conditions||{},periodic=bc.theta_min?.type==='periodic'&&bc.theta_max?.type==='periodic';
        // Non-transformed polar displays retain Plotly's r/theta axes.
        if(p.isPolar&&!p.cartesianTransform)return fallback('r-theta axes use Plotly');
        const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D,0,gl.R32F,cols,rows,0,gl.RED,gl.FLOAT,buffer);
        if(gl.getError()!==gl.NO_ERROR){gl.deleteTexture(texture);return fallback('Texture upload failed');}
        assertRenderContextCurrent(context,container);
        const key=JSON.stringify([context?.resultPath,p,physical,rows,cols]);
        const previous=views.get(container);const domain=(p.cartesianTransform?[-ro,ro,-ro,ro]:[0,(cols-1)*(physical?(c.dx||.001)*1000:1),0,(rows-1)*(physical?(c.dy||.001)*1000:1)]);
        const view={container,data,cols,rows,texture,bytes,resident,domain:previous?.key===key?previous.domain.slice():domain.slice(),initial:domain.slice(),key,polar:[ri,ro,angle],polarView:p,limits:config.zRange&&config.zRange!=='auto'?config.zRange:[min,max===min?min+1:max],flags:{imageMode:0,transformed:+p.cartesianTransform,horizontal:+horizontal,fullModel:+p.fullModel,repeats:p.fullModelMultiplier,periodic:+periodic,anti:+(periodic&&(bc.theta_min.value<0||bc.theta_max.value<0)),harmonic:+harmonic}};
        remove(container);purgePlotlyTree(container);container.replaceChildren();container.style.position='relative';
        const heading=document.createElement('div');heading.textContent=title+' · GPU (experimental)';heading.style.cssText='position:absolute;top:3px;left:4px';container.append(heading);
        const legend=document.createElement('div');legend.style.cssText='position:absolute;right:0;top:28px;bottom:40px;width:32px;background:linear-gradient(#fde725,#35b779,#31688e,#440154);font-size:10px';legend.textContent=`${view.limits[1].toPrecision(4)} / ${view.limits[0].toPrecision(4)}`;container.append(legend);
        const readout=document.createElement('div');readout.style.cssText='position:absolute;bottom:0;left:3px;font-size:11px';readout.textContent='Wheel: zoom · drag: pan · double click: reset';container.append(readout);
        let drag=null;
        container.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,domain:view.domain.slice()};container.setPointerCapture(e.pointerId);};
        container.onpointermove=e=>{const b=bounds(view);if(drag&&e.buttons){const dx=(e.clientX-drag.x)/b.width*(drag.domain[1]-drag.domain[0]),dy=(e.clientY-drag.y)/b.height*(drag.domain[3]-drag.domain[2]);view.domain=[drag.domain[0]-dx,drag.domain[1]-dx,drag.domain[2]+dy,drag.domain[3]+dy];schedule();}else drag=null;
            const x=view.domain[0]+(e.clientX-b.left)/b.width*(view.domain[1]-view.domain[0]),y=view.domain[3]-(e.clientY-b.top)/b.height*(view.domain[3]-view.domain[2]);
            readout.textContent=`X ${x.toPrecision(5)} Y ${y.toPrecision(5)}${physical?' mm':''} · original data retained for numerical export`;
        };
        container.onwheel=e=>{e.preventDefault();const scale=Math.exp(Math.max(-1,Math.min(1,e.deltaY*.001)));for(const offset of [0,2]){const mid=(view.domain[offset]+view.domain[offset+1])/2;view.domain[offset]=mid+(view.domain[offset]-mid)*scale;view.domain[offset+1]=mid+(view.domain[offset+1]-mid)*scale;}schedule();};
        container.ondblclick=()=>{view.domain=view.initial.slice();schedule();};
        container.dataset.fieldBackend='gpu';delete container.dataset.gpuFallback;delete container.dataset.pendingError;
        view.observer=new ResizeObserver(schedule);view.observer.observe(container);views.set(container,view);schedule();
        if(typeof ResultsWorkspace!=='undefined')ResultsWorkspace.plotStatus(context,'ready',[{z:data}]);
        return true;
    }
    function renderImage(container,data,image,title,context) {
        if(image.width!==data[0]?.length||image.height!==data.length){container.dataset.gpuFallback='Overlay dimensions differ from field';return false;}
        if(!render(container,data,title,true,false,context))return false;
        const view=views.get(container);
        const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,gl.RGBA,gl.UNSIGNED_BYTE,image);
        if(gl.getError()!==gl.NO_ERROR){gl.deleteTexture(texture);remove(container);container.dataset.gpuFallback='Overlay upload failed';return false;}
        gl.deleteTexture(view.texture);view.texture=texture;view.flags.imageMode=1;view.cols=image.width;view.rows=image.height;
        const legend=container.children[1];if(legend)legend.remove();schedule();return true;
    }
    return {render,renderImage,remove,has:container=>views.has(container),get bytes(){return [...views.values()].reduce((sum,v)=>sum+v.bytes,0);},get residentBytes(){return [...views.values()].reduce((sum,v)=>sum+v.resident,0);}};
})();
