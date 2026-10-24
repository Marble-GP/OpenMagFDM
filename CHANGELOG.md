# Changelog

All notable user-facing changes are summarized here. Detailed technical
investigations remain under `docs/archive/research/`.

## 1.7.3 — 2026-10-24

- Fix metadata/deletion of result directories exceeding 10,000 entries without
  relaxing backup transfer limits; avoid repeated ancestor/tree inspections.
- Retain deletion errors after refresh, report per-target filesystem failure
  reasons, preserve failed selections and batch selections of more than 100 items.
- Keep deletion progress in memory; write its durable journal only at acceptance
  and completion, with explicit uncertainty after a server restart.
- Add `storage set-workspace-root` and verified `storage migrate-workspaces` for
  configurations, images, material libraries and results, preserving original data.
- Show unfinished jobs in console status/list by default; expose finished
  history explicitly with `jobs list --all`. Retain the server's durable JSON
  ledger for queue recovery, idempotency and recorded convergence.
- Use English for application-owned UI, README and current manuals. Add a
  regression check for Japanese text in first-party public UI source.
- Add a paged Jobs screen, compact workspace activity, and automatic result
  discovery after job completion.
- Add per-workspace result-count and storage admission limits; report upload,
  import and deletion progress in the file manager.
- Preserve input snapshots and enforce quota checks during job preparation;
  improve concurrent import and deletion recovery.

## 1.7.2 — 2026-10-23

- Open a resident `OpenMagFDM>` console on terminal startup, with completion,
  session history, Inquirer menus/settings forms and private password input.
- Apply account, workspace and file-limit changes to the running server; add
  job inspection/cancellation and stop/settings/start without closing the console.
- Guard active analyses during normal shutdown; close the child server on EOF
  or supervisor loss. Preserve headless startup and MCP stdio behavior.
- Add source/packaged PTY acceptance tests alongside HTTP, lifecycle and MCP checks.
- Refresh the README and add CLI, REST API and operations manuals. Archive
  historical plans, implementation records, numerical investigations and measurements.
- See `docs/CLI.md` and `docs/REST_API.md` for current usage;
  UI selection and earlier validation records are under `docs/archive/`.

## 1.7.1 — 2026-10-23

- Unify WebUI/MCP jobs with durable queues, reconnectable controls, idempotency and frozen inputs.
- Add result descriptions, aspect locking, image-independent field lines and tracked asynchronous deletion.
- Retain plots during step preparation and expose preload/cache diagnostics.
- Add CLI storage migration and account/workspace management with explicit HTTPS sharing.
- Add CLI help and configurable per-workspace config/image/material-library quotas without deleting existing files.
- Pin packaged Node to security release 24.22.1 via SEA; update vulnerable image/upload dependencies.
- Consolidate documentation under `docs/`, with release history and archived development plans.
- Keep existing 2D YAML semantics; exclude experimental GPU and 3D features from release packages.
- See `docs/releases/RELEASE_NOTES_v1.7.1.md` for API migration and remaining release acceptance checks.

## 1.7.0 — 2026-10-19

- Refresh the WebUI workspace, result loading guidance and YAML assistance.
- Add scoped local MCP access over HTTP and stdio, typed tools, revision checks,
  idempotent analyses, warning-aware results and connection management.
- Add optional 30-day browser trust and GUI-persisted solver location settings;
  preserve MCP startup settings and automatically detect standard solver layouts.
- Include solver optimization research and numerical regression coverage without
  silently enabling every experimental option.
- Validate native MCP packages on all three release platforms before publication.

See `docs/releases/RELEASE_NOTES_v1.7.0.md` for setup, verification and known limitations.

## 1.6.2 — 2026-08-07

- Fixed an infinite-growth feedback loop in the Run & Preview Input Image
  viewer. The ruler grid and zoom stage now have bounded tracks and reset
  min-content sizing, so loading a large image no longer expands its preview
  area indefinitely.
- Kept image zoom, pan, pixel rulers and bounded image fitting intact.
- Synchronized solver, WebUI package, CMake, vcpkg and release-contract
  version metadata to 1.6.2.

See `docs/releases/RELEASE_NOTES_v1.6.2.md` for the release details.

## 1.6.1 — release candidate

- Fixed the v1.6.0 default banded-DD OpenMP performance regression.
- Added polar full-disc domains (`r_start: 0`, inner Dirichlet required).
- Added metre-authored slide bounds while retaining integer-pixel compatibility.
- Unified static/transient default force export on Distributed Amperian.
- Added Cartesian/Linear templates, image properties, improved previews and
  zoom/pixel rulers.
- Added selective ZIP backup/import and lazy large-list rendering.
- Unified version metadata, export defaults, sample configuration and YAML
  specification; added release-contract CI smoke coverage.
- Polar slide templates now map the detected air-gap radius from source-image
  pixels to physical metres independently of output `nr`, default to the inner
  side, and emit concise step time/iteration summaries when `verbose: false`.
- `conditions.json` is prepared before solver initialization and full-disc
  polar metadata (`r_start: 0`) is accepted by the WebUI; new users receive
  `general_materials.yaml` automatically.
- Fixed the polar permanent-magnet curl so every theta neighbour is projected
  in its own local basis and periodic/anti-periodic seams use wrapped central
  differences. This removes mesh-independent fictitious magnet currents that
  appeared as non-physical high harmonics in transient flux-linkage and EMF
  waveforms.
- Safeguarded opt-in Newton-Krylov Anderson acceleration with true residual and
  energy checks, beta backtracking, rollback/restart, and automatic per-solve
  disable after repeated rejection. Nonconverged runs now restore the best
  evaluated state, prevent poor states from seeding the next transient step,
  and annotate flux-linkage rows in `flux_linkage_status.csv`.
- Clarified solver completion reporting: linear-system messages identify an
  inner solve rather than the whole analysis, while exit code `2` explicitly
  means the analysis completed with retained diagnostic results and a nonlinear
  convergence warning (not an execution failure or a converged result).
- Flux-linkage material definitions now accept either `material_a` or
  `material_b` alone for antiperiodic half-period models, and Dashboard
  flux/Back-EMF timelines reload growing CSV files instead of retaining an
  earlier point count.
- Bounded long-running memory use across the application: decoded Dashboard
  fields now use a 256 MiB byte-budgeted LRU cache, field decoding is
  serialized under an estimated 384 MiB field-data working budget, stale
  consumer leases and plots are released when views change, solver children
  and start preparations are bounded through shutdown, slow legacy responses
  retain bounded log payloads, and completed WebUI jobs/logs expire instead of
  accumulating indefinitely.
- Reduced transient output memory and Windows process churn by sharing one
  immutable field snapshot between asynchronous CSV/TIFF writes, validating
  `export.async_queue_depth` to `1..16`, resetting per-run histories, and
  creating output directories without spawning command shells.

See `docs/releases/RELEASE_NOTES_v1.6.1.md` for the full release candidate notes.

## 1.6.0 — 2026-07-06

- Added polar radial-band Domain Decomposition accuracy mode.
- Integrated nonlinear-solver speedups and chunk-parallel transient sweeps.
- Consolidated AMGCL native multigrid and documented negative research paths.

## 1.5.0 — 2026-06-10

- Added CAD screenshot uniform-colour and polar preprocessing workflows.
- Added polar warp/template insertion, material libraries and dashboard timeline.
- Added global variables, multi-slide/rectangle motion and flux-linkage variants.

## 1.4.0 — 2026-05-28

- Switched default field output to compressed float TIFF and asynchronous I/O.
- Added browser-side TIFF decoding and CSV compatibility paths.

## 1.3.0 and earlier

- Added OpenMP, permanent-magnet models, material libraries, force methods,
  nonlinear solvers, polar coordinates and the original WebUI/API foundation.
- Historical notes are retained in `docs/archive/`, `docs/releases/`, and GitHub Releases.
