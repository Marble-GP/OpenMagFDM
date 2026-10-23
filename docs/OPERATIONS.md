# Server operations

The published baseline is v1.7.2. Whole-workspace storage commands, active-only
console job lists and deletion fixes in current source are **unreleased**.
See [CLI](CLI.md), [REST API](REST_API.md) and [MCP](MCP_INTEGRATION.md).

## Installation and startup

Use the Windows installer or install the solver and WebUI archives together.
The packaged WebUI embeds Node; source installations use Node.js 24.22.1.
WebUI-only archives do not include the C++ solver. Retain its runtime libraries.
The Windows solver needs AVX2. The Linux solver targets x86_64, the macOS solver
arm64. The current macOS WebUI is Intel x64 and needs Rosetta 2 on Apple Silicon.

A terminal startup opens `OpenMagFDM>`. Try `help`, `menu` or `configure`.
Default address: `http://localhost:3000`. Use `--no-console` for a service.
Solver lookup prioritizes `SOLVER_PATH`, the saved Help & Integrations setting,
then standard layouts. Restart after changing a saved path. Check `/api/health`
and `/api/solver/info` for diagnostics.

## Local mode and shared HTTPS

Local mode is loopback-only HTTP without WebUI login. MCP administration uses
separate browser pairing. Shared mode targets LAN/VPN use, requires HTTPS and
requires login even from localhost. Create accounts and assign existing workspaces:

```text
users create admin --admin
users create alice
workspaces list
workspaces assign user_example alice
stop
network shared --cert "C:\certs\server.pem" --key "C:\certs\server-key.pem" --host 0.0.0.0
start
```

Ordinary sessions expire after eight idle hours or 24 hours total. Explicitly
choosing Remember this device allows up to 30 days. Logout, password changes
and account disabling invalidate sessions. Regular users access assigned
workspaces; administrators access all. Unassigned workspaces are not exposed
to regular users. Disabling retains data and running analyses while blocking
new operations and queued starts. Supply TLS certificates and client trust.

MCP remains local. Its shared HTTPS management UI has limitations: create
profiles and save client settings while operating locally before switching to
shared mode. Ownership and enabled-account checks still apply.

## Data and metadata

| Content | Location |
|---|---|
| Configuration YAML | `<workspaceRoot>/configs/<workspace>/` |
| Input images | `<workspaceRoot>/uploads/<workspace>/` |
| Material libraries | `<workspaceRoot>/user-libs/<workspace>/` |
| Results | `<workspaceRoot>/outputs/<workspace>/`, unless separately overridden |
| Accounts, ownership, sessions, settings, job ledger and MCP profiles | OS-private directory keyed by application installation path |

The workspace root defaults to the application directory. Published v1.7.2 can
change results storage only; current source can configure/migrate all categories.
`storage show` reports resolved paths. Private server state is stored under
`%LOCALAPPDATA%/OpenMagFDM/automation/<installation-path-hash>/` on Windows, or
`~/.local/share/OpenMagFDM/automation/<installation-path-hash>/` on Linux/macOS.
Changing the application path changes that lookup. Changing workspace storage
does not change account or MCP identity. Data ZIPs exclude credentials/sessions.

The server **does store JSON locally**. Its `jobs.json` is a durable job ledger,
not a filesystem index or browser cache. Queue recovery, idempotency and
recorded completion/convergence use that ledger. Workspace/result lists scan
actual directories. External moves/deletions can leave finished records pointing
to absent results; history does not prove a result still exists. The current
ledger retains up to 10,000 records and refuses new submissions when full.

New runs freeze the accepted image, effective YAML, material/reference files,
hashes and solver version. Failed input saving prevents launch. Original edits
do not change queued inputs. Optional step-level `InputImg` differs from the
initial image. `conditions.json` holds analysis/rendering conditions;
`run-inputs.json` records provenance once per result, not once per field file.
Legacy results remain readable; missing inputs are reported.

## Jobs and shutdown

Reload, tab close, logout and MCP disconnect do not cancel jobs. Reconnect to
restore controls. Default concurrency is two overall and one per workspace.
Retry submissions with the same idempotency key. On server restart, running
jobs become `interrupted`, queued jobs become `held`. Use `jobs resume <workspace>`;
interrupted calculations do not continue automatically. Normal stop/exit
refuses active work; `--force` terminates it. EOF, OS shutdown and supervisor
loss stop the server and solvers. Current source normally shows unfinished jobs;
`jobs list --all` includes history.

## Deletion failures

Operation status `complete` means targets were attempted, not all deleted.
Inspect each result's `success`, `code` and `error`. Current source keeps
failure details after list refresh and retains failed selections. `EACCES`
indicates permissions; `EPERM` can mean permissions or a file held open;
`EBUSY` means in use. `BUSY` protects active/queued results; `REVISION_CONFLICT`
means a target changed after inspection. Check permissions, close external
viewers and refresh metadata before retrying. Query the operation ID after a
timeout. Large-result metadata/deletion is now independent of the 10,000-entry
transfer budget; backup limits remain in effect.

Deletion progress is kept in server memory. Current source writes the private
operation journal only at acceptance and completion. After a server restart,
an unfinished operation is marked interrupted: some targets may already be
removed, so refresh actual files before retrying. It does not replay deletion.

## Upgrades and relocation

1. Inspect jobs; finish or cancel active work.
2. Export required data and separately back up OS-private administration state.
3. Use `exit`, then replace binaries, public assets and manuals.
4. Preserve input/result directories and private state.
5. Start; check `status`, `storage show`, `limits show`, old results and login/MCP.
6. Resume only the intended held jobs.

The v1.7.1 to v1.7.2 upgrade needs no 2D YAML/result conversion. Use the resident
console for live administration. To change settings without moving data, use
`stop`, `storage set-workspace-root <absolute-path>`, then `start`.
For verified copying use `storage migrate-workspaces <absolute-empty-path>`.
It includes separately stored results, then clears that override. Originals
remain. These workspace commands are unreleased. Published v1.7.2 provides
`set-results-root` and `migrate-results` for results only. Migration rejects
overlap/nonempty targets. Verification failure retains settings and source data;
partial destination copies remain for inspection.

## Environment and resources

Set environment variables before startup; the console passes them to its server.

| Variable | Default / meaning |
|---|---|
| `PORT` | 3000; WebUI/REST port |
| `SOLVER_PATH` | Absolute solver override |
| `MAX_SOLVER_PROCESSES` | 2; solver process cap |
| `MAX_CONCURRENT_JOBS` | 2; job concurrency cap |
| `MAX_CONCURRENT_JOBS_PER_USER` | 1; per-workspace cap |
| `MAX_SOLVER_START_OPERATIONS` | `MAX_SOLVER_PROCESSES`; preparation cap |
| `MAX_SOLVER_START_OPERATIONS_PER_USER` | 1; per-workspace preparation cap |
| `MAX_JOB_RECORDS` | 200; temporary in-memory backend history cap |
| `MAX_JOB_LOG_LINES` / `MAX_JOB_LOG_CHARS` | 500 / 262144; bounded job logs |
| `JOB_TTL_MS` | 86400000; in-memory finished-history retention |
| `OMP_NUM_THREADS` | Solver OpenMP threads |

The durable ledger and temporary backend history are different stores.
`JOB_TTL_MS` does not expire durable jobs. Inputs/configurations are not deleted
solely by age. Quotas use `configure`/`limits`. Measure peak memory before
increasing concurrency. Historical investigations are in [Archive](archive/README.md).
