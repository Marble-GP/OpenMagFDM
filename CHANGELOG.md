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
