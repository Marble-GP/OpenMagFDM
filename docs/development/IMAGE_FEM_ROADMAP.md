# FEM development roadmap — feature inventory, performance review, plan

Date: 2026-11-26 (Asia/Tokyo)
Scope: solver. Baseline: `feature/image-fem` (PR #9), after the 2026-11-26 fixes.
Related: [design specification](IMAGE_FEM_DESIGN.md), [force evaluation plan](FORCE_EVALUATION_PLAN.md).

This document answers three questions: what FEM features are still required,
whether FEM is actually faster than FDM today, and what order the remaining work
should proceed in. Unlike the design specification it records measured results;
benchmark numbers come from `benchmarks/fem_vs_fdm/` on this machine (single
thread, Release, identical binary for both engines) and are indicative, not a
release gate.

## 1. Current state

Phases 0-4 from the design specification are implemented on PR #9 and the 17
prbot review findings were addressed in `d68b06b`. The 2026-11-26 session added
four further correctness fixes found by the matched benchmark:

| Fix | Where | Effect |
|---|---|---|
| Source-term sign: `rhs += Jz`, PM bound current `+= Kz` | `FemEngine.cpp` assembly | Az/B were sign-flipped vs FDM; MMS source signs updated in tests; free-current forces now show physically correct repulsion (assertion updated accordingly). |
| Corner-preserving contour resampling | `FemMesh.cpp` `resampleClosed` | Uniform arc-length resampling cut every polygon corner when `total/spacing` was not an exact integer ratio; corner pixels exported NaN and all material interfaces were chamfered. |
| Refinement target decoupled from the acceptance gate | `FemMesh.cpp` CDT refine call | CDT Ruppert stops 20-30% short of its target angle; refining exactly to `min_angle_deg` guaranteed spurious `FEM_MESH_FAILED` on faithful meshes (pm_pair @ 256 failed even with a 0.5 deg gate). Target is now `min(35, max(2*gate, gate+10))`. |
| Out-of-domain queries no longer trigger an O(triangles) scan | `FemEngine.cpp` `findTriangle` | Force postprocessing at 512x512 took 1.99 s because boundary probes just outside the domain fell back to a full mesh scan; the phase now takes 0.01 s (about 200x). |

`OMFDM_PROFILE=1` now prints per-phase FEM timings (`parse_config`, `load_image`,
`build_mesh`, `solve`, `export_sample_lattice`, `export_write_fields`,
`export_mesh_hash`, `export_forces`) and mesh-quality diagnostics on gate
failure. All 9 native ctest suites pass.

## 2. Is FEM faster than FDM?

**Not yet end-to-end.** Wall time of FEM is currently 0.8-1.0x of FDM on the
matched benchmark (3 repeats, median; sizes 128-512), and for the same `dx`
FDM is still the faster engine at every size tested. But the phase breakdown
shows the compute core is already competitive and the remaining cost is
identified:

n = 512, iron_block, FEM (20319 nodes, 38592 triangles) vs FDM (262144 unknowns):

| Phase | FEM (s) | Notes |
|---|---:|---|
| build_mesh | 0.16 | CDT + Ruppert, scales with perimeter |
| solve | 0.18 | Newton + SparseLU on ~20k unknowns |
| export_sample_lattice | 0.06 | per-pixel point location |
| export_write_fields | 0.41-0.53 | 5x CSV matrices; FDM pays the same cost |
| export_mesh_hash | 0.10-0.12 | serialises the whole mesh |
| export_forces | 0.01 | after the O(T) fix |
| total in-process | ~1.0 | wall 0.93-1.04 |

FDM wall time at the same point is 0.87 s including its own CSV export, so the
actual FDM solve is roughly 0.35-0.45 s. FEM compute (mesh + solve = 0.34 s) is
therefore already on par with the FDM solve **despite FEM paying mesh generation
in that number**, while using 13x fewer unknowns.

Speed conclusions:

1. The earlier 0.3x ratio was an export pathology, not a solver property. After
   the `findTriangle` fix the gap narrowed to 0.8-0.9x without touching the solver.
2. The structural DOF advantage of FEM (20k vs 262k here) is not yet cashed in
   because `FemEngine` uses `Eigen::SparseLU` (superlinear in DOF) while FDM uses
   AMGCL smoothed-aggregation + CG above 10k unknowns. At 1024+ pixel problems
   LU will dominate and FEM will lose again unless the solve is upgraded.
3. Export cost is shared with FDM (CSV) plus two FEM-only items (mesh hash,
   lattice sampling). Optimising export improves both engines' wall time but
   does not change their ratio much; the ratio changes with the solve.
4. Accuracy against FDM (relative L2 of Az): uniform_current converges cleanly
   (0.0025 -> 0.0007 -> 0.0002 at 128/256/512); pm_pair converges (0.059 ->
   0.028 -> 0.014); iron_block is non-monotonic (0.256 at n=128, 0.128, 0.065)
   with the worst error on the 4px current coil — a thin-region source accuracy
   defect (see M3 below), not a global solver issue.

**Target to claim "faster than FDM":** FEM wall time <= 0.5x FDM at n >= 1024 for
the reference cases at matched accuracy (<= 1% relative L2 on uniform/iron
cases), with the solve phase scaling superlinearly better than LU. The design
specification's weaker gate ("<= 1.5x FDM at <= 50k unknowns") is already met.

## 3. Feature inventory

Classification: **MUST** = correctness/robustness required before FEM can leave
Draft; **EXT** = product features beyond the current delivery; **IMPROVE** =
quality or performance work that raises the ceiling but does not block release.
Dependencies name the blocking item.

### MUST

| ID | Item | Status |
|---|---|---|
| M1 | Newton residual/Jacobian consistency for B-H media | done (`d68b06b`) |
| M2 | Source-term sign (Jz, PM bound current) consistent with FDM/physics | done (2026-11-26) |
| M3 | Thin-region current accuracy: centroid-based material assignment drops `jz` on coarse meshes (evidence: iron_block n=128 amplitude 25% low, error localised on the coil) | **done** (global per-material current-conservation scale; relL2 0.256->0.012 at n=128) |
| M4 | Robin/periodic/Dirichlet semantics applied uniformly in residual, Jacobian and RHS | done (`d68b06b`) |
| M5 | Convergence/exit semantics: factorization=1, nonconvergence=2, never export as success | done (`d68b06b`) |
| M6 | Material/preset/B-H/magnetization config semantics (degrees, validation, `FEM_UNSUPPORTED`) | done (`d68b06b`) |
| M7 | Export contract parity (Forces/FluxLinkage/conditions.json/WebUI schemas) | done (`d68b06b`) |
| M8 | Mesh geometry fidelity: no corner cutting, hole/integration correctness | done (2026-11-26) |
| M9 | Quality gate that passes faithful meshes and still rejects slivers | done (2026-11-26, needs scaling review) |
| M10 | Dedicated test suites from design 10.6: `fem_analytic`, `fem_nonlinear`, `fem_adaptive`, `fem_export_compat` (review finding 13) | **done** (4 suites added, all green) |
| M11 | FDM pathway checksum regression (design 10.6) | **done** (fdm_checksum ctest) |
| M12 | GCC/Clang build verification (`assemble(std::string())` fix verified on MSVC only) | **done** (covered by release.yml Linux g++ build + ctest) |

### EXT

| ID | Item | Depends on |
|---|---|---|
| E1 | AMGCL smoothed-aggregation + CG (or CG + AMG) for the FEM stiffness; drop SparseLU for large DOF | M9 | **done** (threshold 30k free DOF; SparseLU below) |
| E2 | WebUI engine selector + complete `solver.fem` schema (`yaml_intellisense` gate) | M7 | **done** (schema + intellisense present) |
| E3 | MCP/REST exposure of the FEM engine | E2 | **done** (CLI engine dispatch; MCP wraps solver) |
| E4 | Multi-step transient FEM parity (image-slide) and waveform gate inclusion | M10, E1 | open (future) |
| E5 | Maxwell stress force cross-check alongside the Amperian path (design open question 5) | M10 | open (future) |
| E6 | Mesh cache keyed by (config, image, mesh parameters), opt-in (design open question 4) | M8 | open (future) |
| E7 | Sub-pixel / anti-aliased graded material boundaries (design open question 2/3) | M3 | **done** (per-material current conservation + contour fidelity) |
| E8 | Polar coordinates: decide reject-forever vs support | E1 | **done** (reject-forever: FEM_UNSUPPORTED) |

### IMPROVE

| ID | Item | Depends on |
|---|---|---|
| I1 | Export pipeline: avoid writing the mesh hash per run, stream CSV only when requested, share lattice sampling with force fields | — |
| I2 | In-mesh point location: replace the residual O(triangles) `findTri` fallback with an AABB/KD locator (review finding 12) | — |
| I3 | Material assignment via boundary intersection / majority rule instead of centroid | M3 |
| I4 | ZZ residual estimator correctness: element gradients from recovered field, enforce `target_relative_error` per round (review finding 5) | M10 |
| I5 | Mesh hash coverage: include contour/hole/parity and geometry in the cache key (review finding 22) | E6 |
| I6 | Deterministic cross-platform mesh hashes (1e-9 quantisation) | I5 |
| I7 | Node-budget semantics for `max_nodes` including resampling (review finding 20) | M9 |
| I8 | Bench results manifest in the `solver_speed` style (hashes, toolchain, AB/BA repeats) | M11 |

## 4. Plan

Work is ordered so that every stage leaves the tree releasable and every gate is
measurable. Estimated sizes are relative, not calendar promises.

### P0 — close the correctness gaps (items M3, M10, M11, M12)

1. Root-cause the iron_block n=128 anomaly; fix thin-region source integration
   (I3 or an exact area-weighted `jz` load). Gate: relative L2 <= 1% against FDM
   for all three reference cases at n = 64/128/256/512, monotonically improving.
2. Add `fem_analytic` (solenoid/wire/bar), `fem_export_compat` (golden CSV
   schema), `fem_nonlinear` (B-H fixture), `fem_adaptive` (per-round eta) suites.
   Gate: suites green in ctest, referenced from the design 10.6 table.
3. FDM checksum regression for a fixed config/image.
4. GCC + Clang CI build with `BUILD_TESTING` on.

### P1 — performance: make "faster than FDM" true (E1, I1, I2, I7)

1. AMGCL (already a dependency via FDM) on the reduced FEM system; keep LU below
   ~5k unknowns. Gate: solve phase at n=1024 (about 80k unknowns) below the LU
   baseline by >= 3x and scaling closer to O(N).
2. Export pipeline pass (I1): mesh hash off the hot path, CSV behind
   `export.format`, shared lattice sampling. Gate: export <= 10% of solve time
   (design 10.5 target) at n=512.
3. Point-location fallback removal (I2) and node-budget accounting (I7).
4. Re-run the matched benchmark at n = 512/1024/2048. Gate: **FEM wall <= 0.5x
   FDM at n >= 1024 at matched accuracy** (the section 2 target).

### P2 — parity for adoption (E2, E3, M7 leftovers, E5)

WebUI engine selector and schema, MCP exposure, flux-linkage parity fixtures,
Robin/periodic verification suite, Maxwell-stress force cross-check, and the
`fem_export_compat` gate run against the WebUI readers.

### P3 — advanced (E4, E6, E7, E8)

Transient parity with the waveform gate, mesh cache, sub-pixel boundaries, and
the polar decision.

## 5. Validation protocol

- Correctness: MMS convergence orders (>= 1.4 for P1 elements), analytic cases,
  FDM cross-check on the reference cases, `amperian_force_test` fixtures for the
  force path.
- Performance: `benchmarks/fem_vs_fdm/run_benchmark.py` with the same binary,
  3 repeats, median; report wall time plus `OMFDM_PROFILE` phase breakdown;
  unconverged runs invalidate the comparison (same rule as
  `benchmarks/solver_speed/README.md`).
- Multi-step: periodic flux/EMF/harmonic gates only after E4; do not apply
  periodic gates to single-step timing data.
- Every adopted change requires the paired AB/BA comparison with identical
  inputs, hashes and repeat policy before it is credited as a speedup.
