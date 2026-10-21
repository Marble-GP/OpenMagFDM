'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const {promisify} = require('node:util');
const {privateDirectory, atomicJSON, localRequest} = require('./access');
const {fail, userKey, noLinks, segments} = require('./workspace');
const {context} = require('./identity-context');
const scrypt = promisify(crypto.scrypt);
const digest = value => crypto.createHash('sha256').update(value).digest('hex');
async function passwordHash(password) {
    if (typeof password !== 'string' || password.length < 12 || Buffer.byteLength(password)>1024) fail('INVALID_PASSWORD','Use a password of at least 12 characters (at most 1024 bytes)');
    const salt=crypto.randomBytes(16).toString('hex');
    const key=await scrypt(password,salt,64,{N:32768,r:8,p:1,maxmem:64*1024*1024});
    return {salt,key:key.toString('hex')};
}
async function verify(password, record) {
    if (typeof password !== 'string' || Buffer.byteLength(password)>1024) return false;
    const key=await scrypt(password,record.salt,64,{N:32768,r:8,p:1,maxmem:64*1024*1024});
    return crypto.timingSafeEqual(key,Buffer.from(record.key,'hex'));
}
class Accounts {
    constructor(base,settings,{directory=privateDirectory(base)}={}) {
        this.base=base; this.settings=settings; this.filename=path.join(directory,'accounts.json');
        this.sessions=new Map(); this.attempts=new Map(); this.inflight=0;
        this.sessionFile=path.join(directory,'web-sessions.json');
        this.ready=this.loadSessions(); this.lastSaved=0;
    }
    async loadSessions() {
        try {await noLinks(this.sessionFile);this.sessions=new Map(JSON.parse(await fs.readFile(this.sessionFile,'utf8')));}
        catch(error){if(error.code!=='ENOENT')throw error;}
    }
    async saveSessions() {
        await require('./workspace').locked(this.sessionFile,()=>atomicJSON(this.sessionFile,[...this.sessions]));
        this.lastSaved=Date.now();
    }
    async read() { try { await noLinks(this.filename); return JSON.parse(await fs.readFile(this.filename,'utf8')); } catch(e) { if(e.code==='ENOENT') return {version:1,users:[],owners:{}}; throw e; } }
    async workspaceEnabled(user) {
        if(this.settings.mode!=='shared')return true;
        const data=await this.read(); return data.users.some(u=>u.name===data.owners[user]&&!u.disabled);
    }
    async identity(req) {
        await this.ready;
        const token=(req.headers.cookie||'').split(';').map(s=>s.trim()).find(s=>s.startsWith('omfdm_session='))?.slice(14);
        const session=token&&this.sessions.get(digest(token));
        if(!session)return null;
        const now=Date.now();
        const data=await this.read(); const account=data.users.find(u=>u.name===session.name);
        if(now>session.expires||now-session.last>session.idle||!account||account.disabled||account.revision!==session.revision) { this.sessions.delete(digest(token));return null; }
        session.last=now;
        if(now-this.lastSaved>60000)await this.saveSessions();
        return {name:account.name,role:account.role,workspaces:Object.keys(data.owners).filter(w=>data.owners[w]===account.name||account.role==='admin'),csrf:session.csrf,tokenHash:digest(token)};
    }
    install(app) {
        const route=fn=>async(req,res)=>{res.set('Cache-Control','no-store');try{await fn(req,res);}catch(e){res.status(e.statusCode||500).json({success:false,error:e.statusCode?e.message:'Account operation failed'});}};
        const sameOrigin=req=>{
            const expected=`${req.socket.encrypted?'https':'http'}://${req.headers.host}`;
            if(req.headers.origin!==expected || (req.headers['sec-fetch-site']&&!['same-origin','none'].includes(req.headers['sec-fetch-site'])))fail('ORIGIN','Same-origin request required',403);
        };
        app.post('/api/auth/login',route(async(req,res)=>{
            if(this.settings.mode!=='shared')fail('LOCAL_MODE','Login is not enabled',400);
            sameOrigin(req);
            const ip=req.socket.remoteAddress, now=Date.now();
            for(const [key,value] of this.attempts)if(now-value.at>15*60*1000)this.attempts.delete(key);
            const attempt=this.attempts.get(ip)||{at:now,count:0};
            if(attempt.count>=10||this.inflight>=4||this.attempts.size>10000)fail('RATE_LIMIT','Try again later',429);
            attempt.count++;this.attempts.set(ip,attempt);this.inflight++;
            try {
                const data=await this.read();const account=data.users.find(u=>u.name===req.body.name);
                const dummy={salt:'00000000000000000000000000000000',key:'00'.repeat(64)};
                const ok=await verify(req.body.password,account?.password||dummy);
                if(!ok||!account||account.disabled)fail('LOGIN_FAILED','Invalid login',401);
                const old=await this.identity(req);if(old)this.sessions.delete(old.tokenHash);
                for(const [key,value] of this.sessions)if(now>value.expires||now-value.last>value.idle)this.sessions.delete(key);
                if(this.sessions.size>=10000)fail('SESSION_LIMIT','Session capacity reached',503);
                const token=crypto.randomBytes(32).toString('base64url'), csrf=crypto.randomBytes(32).toString('base64url');
                const ttl=req.body.remember===true?30*86400000:86400000;
                this.sessions.set(digest(token),{name:account.name,revision:account.revision,csrf,last:now,idle:req.body.remember===true?ttl:8*3600000,expires:now+ttl});
                res.setHeader('Set-Cookie',`omfdm_session=${token}; Path=/; HttpOnly; Secure; SameSite=Strict${req.body.remember===true?`; Max-Age=${ttl/1000}`:''}`);
                await this.saveSessions();this.attempts.delete(ip);res.json({success:true});
            } finally {this.inflight--;}
        }));
        app.get('/api/auth/me',route(async(req,res)=>{
            if(this.settings.mode!=='shared'){localRequest(req);return res.json({success:true,mode:'local'});}
            const identity=await this.identity(req);if(!identity)fail('LOGIN_REQUIRED','Login required',401);
            const {tokenHash,...safe}=identity;res.json({success:true,mode:'shared',...safe});
        }));
        app.use(async(req,res,next)=>{
            try {
                if(this.settings.mode!=='shared') { localRequest(req); return next(); }
                if(['/login.html','/login.js','/favicon.ico'].includes(req.path)||req.path.startsWith('/icon/'))return next();
                const identity=await this.identity(req);if(!identity){if(req.path==='/'||req.path==='/index.html')return res.redirect('/login.html');fail('LOGIN_REQUIRED','Login required',401);}
                if(!['GET','HEAD','OPTIONS'].includes(req.method)) {sameOrigin(req);if(req.headers['x-csrf-token']!==identity.csrf)fail('CSRF','Session token required',403);}
                // Every legacy file parameter is a logical relative ID, never an OS path.
                for(const source of [req.query,req.body])if(source) {
                    if(source.userId) {const user=userKey(source.userId);if(identity.role!=='admin'&&!identity.workspaces.includes(user))fail('FORBIDDEN','Workspace access denied',403);}
                    for(const key of ['result','file'])if(source[key])segments(source[key]);
                }
                const decoded=decodeURIComponent(req.path);if(decoded.split('/').some(p=>p==='..'||p==='.'||p.includes('\\')))fail('PATH','Invalid URL',400);
                if(['/api/get-material-image','/api/solve','/api/solve-stream','/api/stop-solver'].includes(req.path))fail('DEPRECATED','Use workspace-scoped job and image endpoints',410);
                if((req.path==='/api/health'||req.path==='/api/solver/info'||req.path.startsWith('/api/mcp/'))&&identity.role!=='admin')fail('FORBIDDEN','Administrator required',403);
                req.identity=identity;
                context.run(identity,next);
            }catch(e){res.status(e.statusCode||500).json({success:false,error:e.statusCode?e.message:'Access denied'});}
        });
        app.post('/api/auth/logout',route(async(req,res)=>{
            if(req.identity){this.sessions.delete(req.identity.tokenHash);await this.saveSessions();}
            res.setHeader('Set-Cookie','omfdm_session=; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=0');res.json({success:true});
        }));
        app.post('/api/auth/password',route(async(req,res)=>{
            if(!req.identity)fail('LOGIN_REQUIRED','Login required',401);
            await require('./workspace').locked(this.filename,async()=>{
                const data=await this.read(),account=data.users.find(u=>u.name===req.identity.name);
                if(!await verify(req.body.currentPassword,account.password))fail('LOGIN_FAILED','Invalid password',401);
                account.password=await passwordHash(req.body.password);account.revision=crypto.randomUUID();await atomicJSON(this.filename,data);
            });
            res.json({success:true});
        }));
    }
}
module.exports={Accounts,passwordHash,verify};
