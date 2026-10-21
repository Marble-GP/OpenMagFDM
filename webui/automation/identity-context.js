'use strict';
const {AsyncLocalStorage} = require('node:async_hooks');
const context = new AsyncLocalStorage();
function authorizeUser(user) {
    const identity = context.getStore();
    if (identity && identity.role !== 'admin' && !identity.workspaces.includes(user)) {
        const error = new Error('Workspace access denied'); error.statusCode=403; error.code='FORBIDDEN'; throw error;
    }
    return user;
}
module.exports = {context, authorizeUser};
