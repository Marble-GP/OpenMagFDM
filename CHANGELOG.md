# Changelog

All notable user-facing changes are summarized here. Detailed technical
investigations remain under `docs/research/`.

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

See `docs/RELEASE_NOTES_v1.6.1.md` for the full release candidate notes.

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
- Historical notes are retained in `.doc/` and GitHub Releases.
