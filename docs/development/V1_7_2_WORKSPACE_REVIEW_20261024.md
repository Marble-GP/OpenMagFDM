# Workspace / job-management review follow-up

Date: 2026-10-24, Asia/Tokyo. Baseline: main at 36165f8; published v1.7.2
binaries and the running evaluation installation are unchanged. These are
unreleased source changes. No 3D, GPU or numerical-solver changes are included.

## User reports and source behavior

| Report | Follow-up |
|---|---|
| Workspace storage cannot be selected from the resident CLI menu | Add guided Storage actions for all workspace data or results only; verified copy/migration retains originals and can stop/restart an idle server. Existing typed storage commands remain. |
| Console menu prints all completed jobs | Main already defaults status/jobs to unfinished work. Explicit history is available through jobs list --all and now a separate menu item. |
| Workspace activity scrolls through all completed jobs | Replace job cards with counts and a Jobs link. Keep recent file audit events collapsed and bounded. |
| Solver Control expands into a long job list | Add a dedicated Jobs page with active/history filters, 20-row pages, selected-job details, collapsible log, cancellation/resume and result actions. Solver Control shows only active progress and a manager link. |
| Jobs should belong to the user/workspace without permanent UI clutter | Preserve the existing workspace/owner authorization. Display workspace, source and input configuration. Hide finished history by default; keep its durable records for recovery/idempotency/convergence. This does not delete the ledger or results. |
| No result-count/storage limits in the menu | Add results (folder count) and result-mib (stored bytes in MiB) to default/per-workspace limits, CLI help and guided configure. Both default unlimited. |
| Japanese result-description buttons | The existing unreleased English UI changes are retained; the application-owned public-source English regression passes. The published installation is still older. |
| Completed MCP jobs require a page refresh to discover results | Poll a bounded job overview, detect terminal-state changes and refresh the result picker automatically. Keep the loaded plots and selection; retry discovery failures. |

Result quotas are admission controls across common WebUI/REST/MCP submissions,
queued starts, result writes and backup imports. Queued snapshots consume a
folder slot. Storage usage is measured from actual files without a quota JSON
index. Running native solvers may grow beyond the configured byte budget;
they are not automatically cancelled. Existing results are not deleted.
The CLI and REST manuals state this limitation explicitly.

## Deletion investigation

The other-environment deletion failure remains unidentified: its operating
system error code and exact timeline are unavailable. Do not claim its root
cause was reproduced.

Read-only source inspection found a concrete broader-than-needed protection:
AnalysisJobs.busy previously blocked every result in a workspace whenever any
solver was running there. Narrow it to the active job's actual result subtree.
Keep the broad fallback only for an unscoped legacy solver process.

A real Streamable HTTP MCP client test now reads a completed result, exports
it to a temporary archive, remains connected, and deletes that result through
the GUI HTTP adapter while another job in the same workspace is marked running.
Deletion succeeds; the active job's subtree remains protected; the archived
transfer can still be released and the MCP connection remains usable.

An idle connected MCP client does not retain result file descriptors.
Transient reads/exports serialize with file operations; a genuinely open
external handle, OS permissions, antivirus or filesystem failure remains a
possible environment-specific cause. Directory deletion now allows three
short filesystem retries; partial failures preserve their codes/messages.

## Verification and remaining limit

- YAML and result pickers now share a bounded scrolling dropdown (maximum
  300 CSS pixels, further limited by available viewport space). YAML no longer
  relies on the OS-native select popup, whose height cannot be reliably styled.
  Long names use ellipsis with full-name tooltips; keyboard selection scrolls
  the focused item into view. Menus open upward when needed and close when the
  workspace scrolls or the window resizes. YAML load cancellation/failure
  restores the selected filename without discarding editor changes.
- Node regression suite: **124 tests passed**, including workspace isolation,
  result quotas, idempotent quota retries, paged job history, automatic completed
  result discovery/retry, English public UI, connected-MCP deletion, revision
  conflicts, large-directory deletion, 100-entry file pickers, constrained
  viewport placement and cancelled/failed YAML selection.
- Real Windows ConPTY resident console: default startup, Tab/history, guided
  result count/storage settings, verified whole-workspace migration with spaces
  in the destination, restart, masked passwords, cancellation and shutdown passed.
- Isolated resident-console HTTP/native-solver smoke: live quotas, invalid
  command recovery, migration, running-analysis stop guard, force cancellation,
  restart and no orphan process after EOF/parent termination passed.
- Browser regression script updated for the Jobs screen and automatic result
  discovery, but **not run** in this session. Browser-use inventory was empty
  and IAB was unavailable. The computer-use skill found Edge, but activation
  failed; its one refreshed observation retry stopped with: "could not determine
  the current browser URL on Windows with enough confidence to enforce policy."
  All UI automation stopped. Chrome/Edge visual/interaction acceptance remains
  pending; do not present VM DOM tests as real-browser acceptance.

Only isolated fixtures were started. The temporary UI fixture was stopped.
No deployed server, account or user result was changed by this source work.
Motor trial inputs/results and local acceptance artifacts stay outside Git
worktrees under qa; they are not staged or committed. No release/tag was made.
