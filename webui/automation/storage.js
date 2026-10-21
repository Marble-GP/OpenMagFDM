'use strict';
const path = require('node:path');
const fs = require('node:fs');
const configurations = new Map();
function readSettings(base) {
    const filename = path.join(require('./access').privateDirectory(base), 'server-settings.json');
    try { return JSON.parse(fs.readFileSync(filename, 'utf8')); }
    catch(error) { if (error.code === 'ENOENT') return {version:1,mode:'local'}; throw error; }
}
function configure(base, settings = readSettings(base)) { configurations.set(path.resolve(base), settings); return settings; }
function root(base, category) {
    const settings = configurations.get(path.resolve(base)) || {};
    return category === 'outputs' && settings.resultsRoot ? path.resolve(settings.resultsRoot) : path.join(base, category);
}
module.exports = {configure, root, readSettings};
