'use strict';
const fs = require('node:fs/promises');
const crypto = require('node:crypto');
const Jimp = require('jimp');
const { imageSize } = require('image-size');
const { fail, locked } = require('./workspace');
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_PIXELS = 16 * 1024 * 1024;
function inspectImage(buffer) {
    if (!Buffer.isBuffer(buffer) || buffer.length > MAX_IMAGE_BYTES) fail('LIMIT_EXCEEDED', 'Image limit is 10 MiB');
    // Restrict magic bytes BEFORE invoking Jimp/file-type (including its legacy ASF parser).
    const png = buffer.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]));
    const jpg = buffer[0] === 255 && buffer[1] === 216 && buffer[2] === 255;
    const bmp = buffer[0] === 66 && buffer[1] === 77;
    if (!png && !jpg && !bmp) fail('INVALID_IMAGE', 'Only PNG, JPEG and BMP images are supported');
    const dimensions = imageSize(buffer);
    if (!dimensions.width || !dimensions.height || dimensions.width * dimensions.height > MAX_PIXELS) fail('LIMIT_EXCEEDED', 'Image exceeds the 16-megapixel limit');
    return dimensions;
}
class Images {
    constructor(backend) {
        this.backend = backend; this.previews = new Map();
        this.timer = setInterval(() => { void this.cleanup(); }, 60000); this.timer.unref();
    }
    async cleanup(all = false) {
        for (const [key,p] of this.previews) if (all || p.expires < Date.now()) {
            try {
                await locked(p.ws.lockKey, async () => {
                    await p.ws.checkRevision(p.id,p.revision);
                    await fs.unlink(await p.ws.resolve(p.id));
                });
            } catch {} // Never remove a preview that someone edited or replaced.
            this.previews.delete(key);
        }
    }
    async close() { clearInterval(this.timer); await this.cleanup(true); }
    async load(ws, id) {
        if (!id.startsWith('images/')) fail('INVALID_INPUT', 'Expected an image ID');
        const p = await ws.resolve(id); const stat = await fs.stat(p);
        if (stat.size > MAX_IMAGE_BYTES) fail('LIMIT_EXCEEDED', 'Image limit is 10 MiB');
        const buffer = await fs.readFile(p); inspectImage(buffer);
        return Jimp.read(buffer);
    }
    async info(ws, id) {
        return locked('mcp-image-operation', async () => {
            const image = await this.load(ws,id); const counts = new Map();
            const data = image.bitmap.data;
            for (let i=0;i<data.length;i+=4) if (data[i+3]) {
                const key=(data[i]<<16)|(data[i+1]<<8)|data[i+2]; counts.set(key,(counts.get(key)||0)+1);
                if (counts.size > 1000000) fail('LIMIT_EXCEEDED','Image has too many unique colors');
            }
            const colors = [...counts].sort((a,b)=>b[1]-a[1]).slice(0,256).map(([key,count])=>({rgb:[key>>16,(key>>8)&255,key&255],count}));
            return {width:image.bitmap.width,height:image.bitmap.height,uniqueColors:counts.size,colors,truncated:counts.size>256};
        });
    }
    async process(ws, input) {
        return locked('mcp-image-operation', async () => {
            ws.guard('images');
            await this.cleanup();
            if (!input.outputId && input.operation !== 'detect_polar' && this.previews.size >= 16) fail('LIMIT_EXCEEDED','Too many live previews; wait for expiry or save to an explicit output');
            const image = await locked(ws.lockKey,async()=>{await ws.checkRevision(input.id,input.revision);return this.load(ws,input.id);});
            if (input.operation === 'detect_polar') return this.backend.detectPolarGeometry(image, {shape:input.shape || 'auto'}, null);
            let result, metadata = {};
            if (input.operation === 'warp') {
                const p = input.warp;
                if (!p || p.nr*p.ntheta > MAX_PIXELS || p.r_end_px <= p.r_start_px || p.theta_end <= p.theta_start) fail('INVALID_INPUT','Invalid polar grid or radii');
                result = this.backend.warpPolar(image,p);
                metadata.polar_domain = {r_start:p.r_start_px/p.r_end_px*p.r_outer_physical,r_end:p.r_outer_physical,r_orientation:p.r_orientation,theta_range:p.theta_end-p.theta_start};
            } else {
                const processed = this.backend.quantizeImage(image, input.filter || {});
                result = processed.image; const {image:_, ...rest} = processed; metadata = rest;
            }
            const id = input.outputId || `images/preview_${crypto.randomUUID()}.png`;
            if (!/^images\/[^/]+\.png$/.test(id)) fail('INVALID_INPUT','Output must be a PNG file in images');
            const buffer = await result.getBufferAsync(Jimp.MIME_PNG);
            inspectImage(buffer);
            const file = await ws.write(id,buffer,input.expectedRevision ?? null,{permission:'images'});
            const expires = Date.now()+10*60000;
            if (!input.outputId) this.previews.set(`${ws.user}:${id}`,{ws,id,revision:file.revision,expires});
            return {...file,...metadata,preview:!input.outputId,...(!input.outputId?{expires:new Date(expires).toISOString()}:{})};
        });
    }
}
module.exports = {Images, inspectImage, MAX_IMAGE_BYTES};
