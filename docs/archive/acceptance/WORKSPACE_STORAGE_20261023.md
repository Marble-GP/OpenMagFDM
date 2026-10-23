# Workspace storage and file-management follow-up — 2026-10-23

Status: verified source changes after v1.7.2; no replacement release/tag published.
Production changes: `fa89bb1`. English manuals: `5eb63fa`.

## Verified behavior

- Directory metadata/deletion supports a result containing 10,001 field files;
  the separate 10,000-entry transfer guard remains intact. Link rejection,
  revisions, workspace isolation and active-result protection remain covered.
- Failed deletions retain target/error/code and failed selections after refresh.
  Selections over 100 items are split into durable operations.
- Deletion progress stays in memory. A regression asserts exactly two journal
  writes for a normal operation: acceptance and completion. Restarted operations
  are interrupted, retain their target IDs and explicitly report uncertain outcomes.
- Whole-workspace settings/migration cover inputs, configurations, material
  libraries and results, including an existing separate results root. Copies
  are hash-verified; sources and private account identity are retained.
- A running server rejects storage changes. stop/migrate/start is exercised
  through the resident console, followed by real legacy HTTP reads/writes.
- Console status/jobs default to unfinished work; `jobs list --all` includes
  history. Workspace/result discovery reads directories. The server's local
  durable JSON ledger remains; history is not evidence of result-file existence.
- Public UI labels and current manuals use English. Historical records and
  measurement JSON retain their original contents.

## Checks

- Local Node regressions: **113 passed**.
- Source and Windows packaged console HTTP/lifecycle checks, including a native
  running analysis, forced stop and no orphan server after EOF/supervisor loss.
- Source PTY: completion/history, menus/forms, private password input, cancel,
  resize, asynchronous log recovery and shutdown.
- Packaged HTTP/stdio MCP and image API regression checks.
- Release staging regression and current/index documentation link/anchor checks.
- `node webui/tests/rest_manual_smoke.cjs --solver <native-solver>`: runs the
  documented JSON example using the published Windows v1.7.2 solver; verifies
  idempotency, field output, description, ZIP preview/commit, actual completed-result
  deletion, and a filesystem-based result listing despite retained job history.
- [Three-OS packaged validation](https://github.com/Marble-GP/OpenMagFDM/actions/runs/39948925422):
  Windows, Linux and macOS all passed Node tests, audit, source/package terminal
  tests, image operations and packaged MCP HTTP/stdio on `5eb63fa`.

## Limits

The deletion failure reported from another environment was not reproduced there;
the underlying permissions/file-lock/error code remains unknown. Successful
local deletion does not establish its cause. Full Chrome/Edge click-through
verification was not run in this follow-up; the deletion controller was tested
with the actual application functions in the regression fixture. No C++ solver
or 3D code changed, and no new release artifacts were published.
