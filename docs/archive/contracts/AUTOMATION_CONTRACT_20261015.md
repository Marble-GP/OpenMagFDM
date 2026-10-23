# Workspace and embedded MCP contracts

The text-inspection contract below remains v1. As of 2026-10-15 an embedded MCP
implementation is under acceptance testing; see [MCP setup and status](../../MCP_INTEGRATION.md).
The GUI user ID separates data directories but is not authentication. Do not expose
the legacy WebUI to untrusted networks or treat it as an authenticated multi-user API.

## Configuration inspection

`POST /api/config/inspect`, JSON body:

```json
{"content":"coordinate_system: polar\npolar_domain: {r_start: 0, r_end: 0.1}"}
```

Accepts YAML text up to 1 MiB, no filesystem paths, no solver invocation and no
saving. Returns `{version:1, valid:boolean, diagnostics:[], summary, scope}`.
Diagnostics have zero-based `row`, `column`, severity `type` and text. `valid`
means only that editor-level checks found no errors. It is **not** certification
of physical correctness, resolvable expressions, image compatibility or external
material files. Summary values are authored literals/formulas, not evaluated
effective solver parameters. Missing parameters are labelled defaults, not
invented numerical values. Unknown keys are warnings.

The same `workspace-core.js` / `yaml-intellisense.js` code is used in the GUI;
the legacy `/api/validate-config` endpoint is retained unchanged for compatibility.

## MCP capability mapping (implemented, acceptance testing incomplete)

| Capability | Shared implementation | MCP gate |
|---|---|---|
| Inspect YAML | workspace-core.js | bounded text, editor-level scope |
| Files and backups | automation/workspace.js, transfers.js | fixed user, real-path checks, size limits |
| Save config/materials | Workspace.write | expected revision, edit + replace permissions |
| Start analysis | shared submitAnalysisJob + AnalysisJobs | input snapshot, run permission, durable idempotency |
| Inspect progress | AnalysisJobs.get | owner-only record, bounded logs |
| Cancel owned job | common process lifecycle | cancel permission; no cancellation on disconnect |
| Delete data | Workspace.remove | exact revision, delete permission, active-output protection, permanent deletion |

The official SDK supplies Streamable HTTP and a stdio bridge to that same server.
Connection credentials and fixed workspace bindings are outside tool arguments.
The GUI management API requires a paired session, CSRF checks and loopback / Host /
Origin validation. Turning MCP off or revoking a token never grants an alternate
REST route to an MCP client. Existing REST is not blindly proxied.

GUI file mutations now require `revision` (deletion/rename) or `expectedRevision`
(text replacement). Obtain metadata using `GET /api/workspace/file?userId=...&id=...`.
Missing preconditions return 428; changed revisions return 409. Legacy callers must
be updated instead of silently bypassing compare-and-swap. Backup import now uses
`POST /api/import` for inspection and `/api/import/commit` with inspected revisions.
Inspection does not write published data; partial commit failures report `applied`.

MCP output schemas and stable tool errors are defined in `automation/tools.js`.
No merge, tag or release is implied by this contract. Packaged / native-OS / real
solver / long-session tests outstanding are listed in the integration document.
