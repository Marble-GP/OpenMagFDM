# REST API manual

The API shares the WebUI server: `http://localhost:3000` locally, or the configured
HTTPS address in shared mode. Baseline: v1.7.2. See [CLI](CLI.md) and [Operations](OPERATIONS.md).

## 1. Conventions

- Send JSON mutations with `Content-Type: application/json`; uploads use multipart.
- `userId` is a workspace ID, not authorization. Shared access is checked against the login session. Explicitly send the workspace, even where `default` is accepted.
- Logical file IDs are `configs/<name>`, `images/<name>`, `libraries/<name>` and `results/<folder>`.
- A result query uses the returned `outputs/<workspace>/<folder>` identifier, never an OS path. `file` is relative to that result, e.g. `Az/step_0001`. URL-encode parameters.
- Revisions are opaque 64-character SHA-256 hexadecimal values. Retrieve before updating/deleting.
- Responses may be JSON, YAML text, images, TIFF or ZIP. Some legacy getters return HTTP 200 with `success:false` on error: inspect status, body and Content-Type.
- MCP Bearer tokens are MCP-specific. Ordinary shared REST calls use Web sessions. Local calls must pass loopback, Host and Origin checks. If sending Origin, match the destination.

## 2. Authentication and CSRF

Local mode requires no Web login; `/api/auth/me` returns `{success:true,mode:"local"}`.
MCP administration pairing is separate. In shared mode:

1. POST `{name,password,remember?}` to `/api/auth/login`, with a matching Origin.
2. Retain the HttpOnly/Secure/SameSite `omfdm_session` Cookie.
3. GET `/api/auth/me` for `name,role,workspaces,csrf`.
4. Mutations send Cookie, matching `Origin` and `X-CSRF-Token: <csrf>`.

| Method / path | Request | Effect |
|---|---|---|
| `POST /api/auth/login` | JSON `name,password,remember?:boolean` | Shared login; `{success:true}` and session Cookie |
| `GET /api/auth/me` | Cookie in shared mode | Identity/workspaces/CSRF; mode only locally |
| `POST /api/auth/logout` | Cookie, Origin, CSRF | Invalidate this session |
| `POST /api/auth/password` | JSON `currentPassword,password`, Cookie, Origin, CSRF | Change password; invalidate sessions |

Passwords require 12+ characters. Login throttling returns 429. Normal sessions
expire after eight idle hours or 24 total; explicit remember allows up to 30 days.
Use a client that trusts the TLS certificate. Regular users access assigned
workspaces; administrators access all. Health, solver info and MCP administration
require an administrator in shared mode.

```bash
# login.json: {"name":"alice","password":"<your password>","remember":false}
curl --fail-with-body -c cookies.txt \
  -H 'Origin: https://server.example:3000' -H 'Content-Type: application/json' \
  --data-binary @login.json https://server.example:3000/api/auth/login
curl --fail-with-body -b cookies.txt https://server.example:3000/api/auth/me
```

Use `curl.exe` in Windows PowerShell. Keep login/Cookie files private. Subsequent
shared mutations also need Origin and `X-CSRF-Token`.

## 3. Minimal analysis workflow

These examples use local mode; add session/Origin/CSRF for shared mode.

```bash
curl --fail-with-body -F 'userId=demo' -F 'image=@model.png' \
  http://localhost:3000/api/upload-image
curl --fail-with-body -D config-headers.txt -o model.yaml \
  'http://localhost:3000/api/config?userId=demo&file=model.yaml'
curl --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @config-save.json http://localhost:3000/api/config
curl --fail-with-body -H 'Content-Type: application/json' \
  --data '{"userId":"demo","configFile":"model.yaml","imageFile":"model.png","idempotencyKey":"demo-run-001"}' \
  http://localhost:3000/api/jobs
curl --fail-with-body 'http://localhost:3000/api/jobs/JOB_ID?userId=demo'
```

Example `config-save.json` for a new white-only material image:

```json
{
  "userId": "demo",
  "file": "model.yaml",
  "content": "coordinate_system: cartesian\nmesh: {dx: 0.001, dy: 0.001}\nmaterials:\n  air: {rgb: [255, 255, 255], mu_r: 1, jz: 0}\nboundary_conditions:\n  left: {type: dirichlet, value: 0}\n  right: {type: dirichlet, value: 0}\n  top: {type: dirichlet, value: 0}\n  bottom: {type: dirichlet, value: 0}\n",
  "expectedRevision": null
}
```

Define every image material color. Updating an existing file requires its
retrieved revision. Use a fresh idempotency key per new analysis, reuse on retry.

## 4. Configuration, images and material libraries

| Method / path | Input | Response |
|---|---|---|
| `GET /api/config/list` | query `userId` | `{success,files:[name]}` |
| `GET /api/config` | query `userId,file?` (default `sample_config.yaml`) | YAML, `X-File-Revision` |
| `POST /api/config` | JSON `userId,file,content,expectedRevision` | `{success,revision}` |
| `POST /api/config/inspect` | JSON `content`, up to 1 MiB | `{version:1,valid,diagnostics,summary,scope}` |
| `POST /api/validate-config` | JSON `config` (YAML string) | Legacy `{valid,errors}` |
| `POST /api/upload-image` | multipart `userId,image` | `{success,filename,path,originalName}` |
| `GET /api/images` | query `userId` | `{success,images:[name]}` |
| `GET /uploads/:user/:filename` | Workspace/image in URL | Authorized image bytes |
| `GET /api/material-libraries` | query `userId` | `{success,libraries:[{filename,size,modified}]}` |
| `POST /api/material-libraries` | multipart `userId,library` | New YAML `filename,revision` |
| `GET /api/material-libraries/:filename` | query `userId` | YAML, `X-File-Revision` |
| `PUT /api/material-libraries/:filename` | JSON `userId,content,expectedRevision` | `{success,filename,revision}` |
| `GET /api/user-files` | query `userId` | `{success,files}` across categories |
| `GET /api/workspace/file` | query `userId,id` | `{success,file}`; absent target is `file:null` |

A missing config GET returns server-default YAML with an empty revision; use
workspace metadata to check existence. Creation uses `expectedRevision:null`;
replacement uses the current revision. Missing revision never grants unconditional
overwrite. Config/material YAML saving is limited to 100 KiB. Images are PNG,
JPEG or BMP, up to 10 MiB, with additional decoded-pixel guards. CLI quotas apply.
Inspection positions use zero-based `row,column`. Editor validity does not
establish physical validity, evaluated expressions, file resolution or convergence.

## 5. Persistent jobs

| Method / path | Input | Response |
|---|---|---|
| `POST /api/jobs` | JSON `userId,configFile,imageFile,idempotencyKey`, optional `materialLibraryFile` | 202 initially, 200 on retry; `jobId,status,resultId,...` |
| `GET /api/jobs` | query `userId` | `{success,jobs}`, newest first, including history |
| `GET /api/jobs/:jobId` | query `userId` | State, progress, completion/convergence/warning, `logTail` |
| `DELETE /api/jobs/:jobId` | query `userId` | 202; state after cancellation request |
| `POST /api/jobs/resume` | JSON `userId` | `{success,jobs}` after releasing held jobs |

Keys must be nonempty and at most 128 characters. The same workspace/key/input
names return the original job; reusing the key for different names returns 409.
Use saved workspace filenames. YAML file references use logical IDs such as
`configs/...`. Inputs, effective configuration, materials and references are
frozen at acceptance. This server stores its durable job ledger in local private
JSON; it is not a result-directory index. Result listings scan actual storage.

States: `preparing,queued,held,running,cancelling,completed,failed,cancelled,interrupted`.
Completion and convergence differ: check `converged,warning,exitCode,message`
and logs. `resultId` is a logical file ID; `resultPath` is for retrieval APIs and
may be null before preparation. Disconnect does not stop jobs. Poll after
reconnection/cancellation until final state. Restart interrupts running jobs
and holds queued jobs; calculations do not automatically resume midway.

## 6. Results, fields and images

Inputs below are query parameters unless specified in the URL.

| Method / path | Input | Response / behavior |
|---|---|---|
| `GET /api/results` | `userId` | `{success,results}` with `name,path,steps` |
| `GET /api/user-outputs` | `userId` | `{success,outputs}` including sizes |
| `GET /api/detect-steps` | `result` | `{success,steps,hasTiff,hasCsv}`; deduplicates CSV/TIFF steps |
| `GET /api/load-field` | `result,file,prefer?` | Raw TIFF or decoded CSV JSON |
| `GET /api/load-csv-raw` | `result,file` | CSV text, preserving time-series headers |
| `GET /api/load-conditions` | `result` | Conditions JSON |
| `GET /api/get-conditions` | `result` | Alias for conditions |
| `GET /api/get-log` | `result` | `log.txt` text |
| `GET /api/get-flux-linkage` | `result` | `{success,steps,series,step_column}` |
| `GET /api/get-original-input` | `result` | Accepted initial image, or 404 if absent |
| `GET /api/get-step-input-image` | `result,step?` (default 0) | Step `InputImg`, or 404 |
| `GET /api/get-boundary-image` | `result,step?` (default 0) | `BoundaryImg` PNG |
| `GET /api/get-coarsening-mask` | `result,step?` (missing/0 means 1) | Legacy mask, or 404 |
| `GET /api/results/:resultFolder/field-at-point` | `userId,x,y,step?` | Point values; Cartesian x/y [m], polar x=r [m], y=theta [rad]; default step 0 |

`resultFolder` is a folder name. Handle missing optional output and legacy data.

```bash
curl --get --data-urlencode 'result=outputs/demo/output_example' \
  --data-urlencode 'file=Az/step_0001' \
  --output field.tiff http://localhost:3000/api/load-field
```

`load-field` accepts an omitted extension. Default preference is TIFF then CSV;
`prefer=csv` reverses it. TIFF returns `image/tiff` and `X-Field-Format: tiff`.
CSV returns `{success:true,data:[[...]],format:"csv",precision:"double"}` with
vertically reversed rows and null missing values. Raw TIFF has no such row
conversion. Decode according to Content-Type, not the requested extension.
Az field lines do not require an image. An initial image is not a substitute
for moved/rotated step material geometry.

## 7. Descriptions, rename and deletion

| Method / path | JSON input (GET uses query) | Response |
|---|---|---|
| `GET /api/user-outputs/:folderName/description` | `userId` | `{success,description,revision}`; absent description is empty/null |
| `PUT /api/user-outputs/:folderName/description` | `userId,description,expectedRevision` | Saved metadata; description up to 32 KiB |
| `PUT /api/user-outputs/:folderName/rename` | `userId,newName,revision` | `{success,newName,revision}` |
| `POST /api/file-operations` | `userId,files:[{id,revision}]` | 200 `{success,operation}`; accepts asynchronous deletion |
| `GET /api/file-operations` | `userId` | Latest 100 `operations` |
| `GET /api/file-operations/:id` | `userId` | Current `operation` |

Retrieve each target's current revision first. One operation accepts 1..100
targets; concurrent workspace deletion returns 409. Current-source WebUI splits
larger selections into operations automatically.

```json
{"userId":"demo","files":[{"id":"results/output_example","revision":"<current 64-character SHA-256>"}]}
```

Operation fields include `id,status,total,current,results,created,finished?`.
States are `running,complete,interrupted`. **`complete` does not mean deletion
succeeded**: count each result's `success` and inspect `error,code`. Reconnect
using the operation ID. Refresh state/revisions before retrying interrupted
targets. A timeout is not proof of failure. Active/queued results are protected.
Current source keeps failed selections and displays target-specific errors after
list refresh; large-result inspection/deletion no longer uses the transfer-entry cap.
Progress remains in server memory; current-source journals are written at
acceptance and completion only. After a server restart, an interrupted operation
may have removed targets without a persisted outcome: refresh actual files.

Legacy synchronous DELETE paths: `/api/config/:filename`, `/api/images/:filename`,
`/api/material-libraries/:filename`, `/api/user-outputs/:folderName`. Send
`userId,revision` in the body, or query `userId` and the `If-Match` header.
`DELETE /api/user-files` takes `userId,category,name,revision`;
`DELETE /api/user-outputs/bulk` takes `userId,files:[{name,revision}]`.
Use asynchronous operations for large results.

## 8. ZIP backups

| Method / path | Input | Response |
|---|---|---|
| `GET /api/backup-manifest` | query `userId` | `categories,totalSize` |
| `GET /api/export` | query `userId,include?` | ZIP; default include `configs,images,libraries` |
| `POST /api/export` | JSON `userId,entries:[logical ID]` | ZIP of selected 1..10000 files |
| `POST /api/import` | multipart `userId,backup` | Preview `transferId,files:[{id,size,expectedRevision}],...` |
| `POST /api/import/commit` | JSON `userId,transferId,files:[{id,revision}]` | `success,written,skipped` |
| `DELETE /api/import/:transferId` | query `userId` | Release preview data |

`include` is comma-separated; request `results` explicitly. Initial import only
inspects/stages data. Commit all preview targets using the preview's
`expectedRevision` as `revision` (null for new targets). This adapter does not
commit subsets. Conflicts return 409; partial failures may include `applied`
IDs. ZIP upload limit is 1 GiB, with additional expanded-size/resource guards.
Legacy JSON backups are limited to 16 MiB. Links, encryption and foreign paths
are rejected. Backup transfer-entry limits still apply to large directories.

## 9. Image preprocessing

Saved-image operations share JSON `userId,filename`.

| Method / path | Additional input | Effect |
|---|---|---|
| `POST /api/materials/detect` | multipart `image`; optional query `rareThreshold,blendTolerance,minColorDist,blendBaseMinRatio` | Colors, anti-alias candidates and material YAML |
| `POST /api/preprocess-polar/detect` | optional `hint,color_groups` | Estimate center, radii and periodicity |
| `POST /api/preprocess-polar/warp` | `center_x,center_y,r_start_px,r_end_px,theta_start,theta_end,nr,ntheta`; optional `r_orientation,output_filename,r_outer_physical,preview` | Polar image and domain |
| `POST /api/preprocess-polar/cleanup-preview` | Common input | Remove this image's warp preview |
| `POST /api/preprocess-filter/quick-stats` | Common input | `unique_colors,image_width,image_height,looks_noisy` |
| `POST /api/preprocess-filter/auto-tune` | optional `nTargets,maxEvals,subsampleSize,initial,rareThreshold` | Search quantization settings |
| `POST /api/preprocess-filter/quantize` | optional `rareThreshold,nTargets,minTargetDist,despeckleRadius,bilateralSigmaSpatial,bilateralSigmaColor,minIslandSize,preview,output_filename` | Quantized image |

Warp angles use radians; image center/radii use pixels; `r_outer_physical` uses
metres. `nr,ntheta` are integers >=2 with product <=16*1024*1024.
Orientation is `horizontal` (default) or `vertical`. Outputs include `filename,path`.
`preview:true` uses a fixed per-input preview name. Output images share workspace
quotas. Review inferred geometry/materials before analysis.

## 10. Diagnostics, MCP administration and retired APIs

| Method / path | Purpose |
|---|---|
| `GET /api/health` | Solver availability and storage writability; 200 `ok`, 503 `degraded` |
| `GET /api/solver/info` | Solver version/path/features and Node/OS |
| `GET /api/get-transient-config` | Server-default motion settings; use conditions for result-specific settings |
| `GET /api/automation/activity?userId=...` | Local job/operation activity |
| `POST /api/automation/jobs/:id/cancel` | Legacy local cancellation; prefer common jobs API |

MCP management adds loopback checks/browser pairing to ordinary REST auth.
POST `{code,remember?}` to `/api/mcp/pair` for an `omfdm_admin` Cookie and `csrf`.
Management mutations use `X-OMFDM-CSRF`, separate from Web-session CSRF.

| Method / path | Input / effect |
|---|---|
| `GET /api/mcp/status` | Listener/connections and administration CSRF |
| `POST /api/mcp/forget-browser` | Remove trusted-browser state |
| `GET /api/mcp/solver` | Saved solver path/detection |
| `PUT /api/mcp/solver` | `{path}`; absolute, or empty for detection; restart required |
| `PUT /api/mcp/settings` | `{enabled:boolean,port:1024..65535}` |
| `POST /api/mcp/connections` | `{name,user,permissions}`; raw token shown once |
| `PATCH /api/mcp/connections/:id` | Update permissions/revocation |
| `POST /api/mcp/connections/:id/test` | Server-side connection check |
| `GET /api/mcp/setup/:id` | HTTP URL and stdio arguments, without raw token |

See [MCP](MCP_INTEGRATION.md) for the shared HTTPS limitation. MCP JSON-RPC tools
use the separate `/mcp` listener. Old POST `/api/solve`, `/api/solve-stream`,
`/api/stop-solver` and `/data` return 410. `/api/load-csv` is deprecated CSV-only
compatibility; prefer `load-field`. `/api/get-material-image` returns 410 in
shared mode; use workspace-scoped image routes.

## 11. Errors and retries

| HTTP | Typical meaning | Action |
|---|---|---|
| 400 | Invalid input/path/format | Correct from response details |
| 401 | Login absent/expired | Log in; retrieve workspace and CSRF again |
| 403 | Ownership, Origin or CSRF | Check assignment, Cookie and headers |
| 404 | Target/output absent | Distinguish optional, legacy and deleted output |
| 409 | Revision/idempotency/quota/busy conflict | Refresh or inspect existing operation |
| 410 | Retired route | Use common jobs/scoped routes |
| 413 | Resource/size limit | Check endpoint-specific limits |
| 428 | Required revision absent | Retrieve metadata first |
| 429 | Attempt/resource throttling | Wait before retry |
| 500 / 503 | Server/readiness/capacity issue | Inspect logs/status; query accepted operations |

Errors vary by adapter: `{success:false,error,code?}`, `{error,code}`, or text.
After disconnection, retry jobs with the same key and query deletions by operation ID.
