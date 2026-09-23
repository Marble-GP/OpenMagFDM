# Solver speed / convergence experiment (2026-09)

Branch: `feature/solver-speed-convergence`, baseline: `25859a7` (v1.6.2).
Scope: fixed-grid equations, profiling/residual/CRS engineering first, then a
verified tangent with three bounded preconditioner candidates. No material-law
changes, new AMR/DD, merge, tag, or release are authorized by this experiment.

## Acceptance protocol

- Primary: 1,339,200-DOF IEEJ-D, 12-step rotation sweep, 12 Windows threads,
  tolerance 1e-3, max 300 outer iterations, chunks=1, identical export settings.
- Alternate baseline/trial three times; compare medians only if all steps
  converge. Check 4 threads, cold start and 35H300 secondarily.
- Phase 1: identical equations, normalized residual difference <=1e-11,
  target >=15% end-to-end reduction with no peak-memory increase.
- Phase 2: smooth-region Jv error <=1e-5 over a step-size sweep, one-sided
  tests at interpolation/clamp kinks; >=15% further speed improvement and
  no fewer converged steps. Otherwise retain only as a documented experiment.
- Only passing candidates: 248 points / 12 pixels per step, full revolution.
  Flux normalized RMS <=0.1%, periodic-difference EMF <=0.5%, harmonic
  amplitude difference <=0.1% of reference fundamental. An unconverged
  reference cannot pass waveform validation.
- Tangent peak memory <=2x optimized secant and <=8 GiB. Check retained
  memory across iterations/steps, not just one snapshot.

## Reproduction

Keep detached baseline and experimental source/build directories separate.
Build both with the same compiler, dependencies and Release flags. The benchmark
runner is `benchmarks/solver_speed/run_comparison.py` (Python + psutil). Supply
absolute paths with `--baseline`, `--trial`, `--baseline-repo`, `--trial-repo`,
`--config`, `--image`, `--output` (new directory), `--threads 12 --repeats 3`.
YAML `omp.threads` must match `--threads`; it overrides the environment.
Do not run builds or other solvers concurrently with timings.

The manifest records input and executable hashes, source commit/diff, DLL hashes,
CMake cache, machine and configuration. Process time includes initialization,
exports, async drain and process exit. Windows memory includes OS high-water
counters; Unix sampling is a lower bound, not a certified peak. An exit code 2
is a convergence warning and is never treated as a successful speed measurement.

## Prior experiments and differences

| Prior work | This experiment |
|---|---|
| `65e160a`: stale AMG increased inner CG from 2–3 to 73–183 | Reuse structure/storage only, rebuild numerical AMG every outer iteration |
| `b0b5ed9`: frozen-mu coarse residual/correction was negative | Refresh each candidate B/H/mu before evaluating the full nonlinear residual |
| `0133285`: FD true Newton + energy converged, but was expensive | Verify derivatives first; build preconditioning once per outer iteration |
| `16be6ba`: tangent + secant AMG / BiCGStab frequently hit 100 inner iterations | FGMRES, true inner residual gate, bounded A/B/C candidates and fallback |
| `f41dc86`: safeguarded Anderson | Preserve safeguards; include QR, candidate checks and rollback in timings |

The old blanket assertion of an intrinsic B-H convergence floor was superseded
by the July true-Newton experiments. No floor is inferred from truncated runs.
The cell-centered energy and face-based residual are not assumed to have an
exact discrete gradient relationship.

## Status

Implementation/verification in progress. No optimization has passed the full
adoption protocol yet; unmeasured items are not passes.

- Profiling commit `b93cdaf`: Windows Release build, polar curl and release-contract
  smoke passed. The deliberately limited 8-iteration IEEJ-D diagnostic exits 2
  and records the split timing correctly; it is not a convergence benchmark.
- Shared row/CRS implementation: 108 boundary/orientation/grid cases with three
  coefficient/source refreshes each (324 comparisons) against frozen `25859a7`
  triplet assemblers. Maximum normalized direct-residual error 1.84761e-17.
  Includes additive Neumann pin, axis Dirichlet, Robin, boundary profiles,
  periodic/antiperiodic seams and same-DOF/different-grid cache invalidation.
  Windows Release CTest 2/2 and release-contract smoke passed.
- `OMFDM_ASSEMBLED_RESIDUAL=1` is a diagnostic-only reference switch; ordinary
  line search and AA always recompute B/H/mu, then apply the shared row operator
  without assembling a trial matrix. The existing coarsened path is unchanged.
- The common static core is now compiled once and linked into solver, DD bench
  and numerical tests. SparseLU/legacy APIs convert explicitly; full-grid NK
  retains row-major CRS allocation and feeds the AMGCL Eigen adapter directly.
- Full-sweep timing/quality, tangent, Linux and macOS gates remain outstanding.
