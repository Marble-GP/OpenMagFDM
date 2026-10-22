'use strict';
const defaults = Object.freeze({configs:20, images:20, libraries:null});
function value(text) {
    if(text === 'unlimited') return null;
    if(!/^(0|[1-9][0-9]*)$/.test(String(text)) || !Number.isSafeInteger(Number(text))) throw Error('Limit must be a nonnegative integer or unlimited');
    return Number(text);
}
function effective(settings, workspace) {
    const overrides=settings.fileLimits?.workspaces;
    const result={...defaults,...settings.fileLimits?.defaults,...(overrides && Object.hasOwn(overrides,workspace) ? overrides[workspace] : {})};
    for(const [key,count] of Object.entries(result)) if(!Object.hasOwn(defaults,key) || (count!==null && (!Number.isSafeInteger(count)||count<0))) throw Error('Invalid file limit settings');
    return result;
}
module.exports={defaults,value,effective};
