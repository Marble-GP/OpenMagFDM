'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {tokenize,complete}=require('../automation/server-console');
test('console paths preserve Windows separators and quoted spaces',()=>{
    assert.deepEqual(tokenize('storage set-results-root "D:\\Analysis Results"'),['storage','set-results-root','D:\\Analysis Results']);
    assert.deepEqual(tokenize("users create 'example' --admin"),['users','create','example','--admin']);
    assert.throws(()=>tokenize('storage show "unfinished'),/Unclosed quote/);
    assert.ok(complete('limits s')[0].includes('limits set-default'));
});
