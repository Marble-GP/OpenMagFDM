# v1.6 — Domain Decomposition / Optimized Schwarz for the polar nonlinear solver

> **v1.6.1 status update (2026-07-11):** This is a historical investigation
> report. The later radial-band C++ DD implementation shipped in v1.6.0, and
> v1.6.1 fixes its default serial-band OpenMP performance regression. The
> production contract is now documented in `README.md` and
> `docs/CONFIGURATION.md`; statements below that the C++ build is deferred
> describe the earlier checkpoint, not the current implementation.

**Status: investigation COMPLETE and POSITIVE (science) / DEFERRED (the large C++ build).**
Branch `feature/adaptive-mesh-v1.6`. Date 2026-06-19. Author S.Watanabe + Claude Opus 4.8.
Working PoC + all results: `../bench_results/superlinear_poc/dd_poc/` (gitignored).

---

## 1. Goal & background

Speed up the polar nonlinear magnetostatics solve (IEEJ-D IPMSM, 1.34M DOF = nr 450 × ntheta 2976,
nonlinear iron `pure_iron_model`, mu_r 2.5↔9970). Prior negative results: algebraic/FAS coarsening
(Phase BK), true-Jacobian + line search (Phase BL: iter count is intrinsic to the stiff B-H knee), and
**adaptive monolithic mesh coarsening (v1.6 Stages 1-2): REFUTED** — the true-init diagnostic showed the
fine solution is NOT a fixed point of the coarse system (residual 40, flux 1/6), because mu(|curl Az|)
cannot be coarsened on a NON-UNIFORM (mixed-resolution) mesh: the skip-spanning curl gives a wrong B
which, through mu_r 2.5..9970, breaks the operator.

**The DD idea (user's):** split the domain into REGULAR sub-patches, each solved on its OWN UNIFORM grid
(so mu is self-consistent — the failure mode is structurally avoided), coupled by iterated interface
(optimized Schwarz / Robin) transmission. Coarsen the smooth/uniform patches; keep the gap + saturated
iron fine.

---

## 2. What was built (committed C++, all backward-compatible; default-off / md5-identical regressions)

| commit | feature |
|---|---|
| `7214fba` | per-theta boundary `value_profile`/`gamma_profile` (radial inner/outer) + **symmetric conservative Robin** assembly (replaces the old non-symmetric node-replacing Robin that broke AMGCL-CG) |
| `8449a46` | **warm-start** `nonlinear_solver.initial_az_path` (load initial Az from raw f64 file, skip linear init guess) |
| `608c787` | theta-edge per-r profiles (`theta_min`/`theta_max`) + `polar_domain.theta_offset` (absolute angle of local theta=0, for theta-sector sources) |
| `3c2132a` | magnetization-curl `theta_offset` fix (`computeMagnetizationCurlPolar` used local theta → wrong Jz_mag in a theta-sector) |
| `b5bc27a` | symmetric **theta-Robin** transmission |

Key C++ locations: `MagneticFieldAnalyzer.cpp` buildMatrixPolar (radial Robin in the radial if-chain
~11220; theta Robin in the theta section ~11290; profiles at the Dirichlet rows + neighbour-elimination);
setupPolarSystem profile loading ~531-577; computeMagnetizationCurlPolar ~2102; warm-start in
`MagneticFieldAnalyzer_nonlinear_newton.cpp` ~171.

### The symmetric Robin (reusable, generally correct)
Old polar Robin `(alpha+beta/dr)Az_B-(beta/dr)Az_{B-1}=gamma` was non-symmetric + badly scaled (AMGCL-CG
diverged → SparseLU/5000-iter, ~70 s/solve) AND dropped the boundary node's source/theta terms. New form:
the boundary node keeps its full FV balance; only the boundary FACE flux is replaced by
`c*(gamma-alpha*Az)/beta` (radial c=r*nu/dr; theta c_th=1/(r*mu*dtheta)), a symmetric `-c*alpha/beta`
diagonal + `-c*gamma/beta` RHS → SPD M-matrix, AMGCL-CG works. The python transmission gamma stays
`alpha*u + beta*du/dn` (plain derivative; the c factor cancels).

---

## 3. Results (all in dd_poc/, see S1/S2/S3 *_result.md)

1. **DD converges to the monolithic, linear AND nonlinear.** 2-domain radial: Robin 3 outer iters
   (Dirichlet 21). Nonlinear DD flux within **0.2%** of monolithic (fixed-point check: each subdomain
   reproduces the monolithic given the true trace).
2. **Image patch analysis (the key correction).** Omega_fine≈80% was a radial-split artifact. The image
   is 92% uniform-material interior; splitting by field: linear materials 34% + smooth iron 55% = **~89%
   cheap**, only **~8-11% irreducible-fine** (saturated tooth-tips/yoke-roots + boundary skeleton).
   Maps: patch_map.png, field_map.png, patch_debug.png.
3. **Patch-coarsening of nonlinear iron WORKS (the crux).** Coarsening the smooth yoke on its OWN uniform
   grid: **0.9% @ cf2, 1.6% @ cf4** (graceful, Stage-0-like) — exactly where adaptive monolithic
   coarsening gave 1/6. Confirms the structural claim.
4. **Coarse space gives scalability.** 2-level (theta-coarse Galerkin) bounds outer iters to **~10 for
   2→32 patches**; one-level never converges (grows with #patches). Textbook, confirmed on the real
   variable-mu operator.

### Gotchas found (important for any resumption)
- **Vertical flip:** the solver maps grid row j ↔ image row (nrows-1-j). Radial DD (crops cols) is
  immune; THETA DD (crops rows) MUST crop image rows `[NTH-1-g for g in reversed(grid_rows)]`.
- **theta_offset** must be applied in BOTH the magnetization array-pattern loop AND
  computeMagnetizationCurlPolar (Mx/My→Mr/Mtheta conversion).
- **theta-Robin optimal p ~ 40× smaller than radial** (c_th scale); even tuned, scalar-p theta Robin is
  rho~0.85 on the full-radial interface → needs the coarse space for many sectors.

---

## 4. Aggregate assessment & decision

All components validated → realistic aggregate: ~10 bounded outer iters × (re-solve fine ~11% warm
~2-3s + cheap coarse patches + coarse solve) ≈ **~2-3× vs monolithic 77 s**. MODEST, and **below Stage 0
uniform multi-fidelity (5-7× @ 3%)** which already ships and is far simpler. DD's genuine edge is
**ACCURACY** (gap + saturated teeth kept fully resolved — correct exactly where uniform downsampling is
wrong). The wall win REQUIRES a C++ in-solver DD (Python per-patch process overhead ~0.3 s × ~100
patches × ~10 iters would dominate).

**DECISION (2026-06-19): CONCLUDE here.** Bank the 6 committed solver improvements + this de-risking
record. Defer the large C++ in-solver patch-DD build. Stage 0 uniform multi-fidelity remains the
practical screening tool; pursue the C++ DD only if ACCURACY (not raw speed) becomes the priority — e.g.
a fast gap/saturation-accurate solve for the final-ranking stage of the shape-optimization workflow,
where Stage 0's 3% flux error is too coarse.

## 5. To resume (everything is de-risked; only the BUILD remains)
Build a C++ in-solver patch-DD in ONE process: (a) auto patch decomposition from the material+|grad B|
maps (field_map logic); (b) per-patch assembly reusing buildMatrixPolar + the symmetric radial/theta
Robin; (c) coarsen smooth/linear patches on own uniform grids; (d) mortar transmission across coarse/fine
resolution jumps; (e) 2-level coarse-space correction each Schwarz sweep; (f) warm-started patch solves.
PoC scripts to port/validate against: dd_poc/{s2_schwarz,s3_patch_dd,s3_coarsen_poc,coarse_space_poc}.py.
