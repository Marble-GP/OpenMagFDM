'use strict';
importScripts('flux-filter.js');
self.onmessage=event=>{
    try {self.postMessage({result:FluxFilter.analyze(event.data.data,event.data.settings,event.data.status)});}
    catch(error){self.postMessage({error:error.message});}
};
