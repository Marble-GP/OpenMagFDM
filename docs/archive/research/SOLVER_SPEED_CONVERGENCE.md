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
runner is `benchmarks/solver_speed/run_comparison.py` (install the pinned
`benchmarks/solver_speed/requirements.txt` in a dedicated Python environment). Supply
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

The follow-up workspace branch and its preserved tangent baseline are documented
in [TANGENT_WORKSPACE_OPTIMIZATION.md](TANGENT_WORKSPACE_OPTIMIZATION.md).

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
- Phase 2 now differentiates the **actual shared residual** using local sparse
  forward differentiation. Boundary elimination and profile RHS derivatives
  are included. PCHIP value derivatives and the piecewise-linear inverse B-H
  derivative match their respective evaluation paths. Arbitrary `mu_r` formula
  derivatives use a five-point scalar numerical derivative (not symbolic AD);
  material values, interpolation and extrapolation are unchanged.
- `jacobian: tangent` selects FGMRES(20), at most 100 inner iterations, EW
  tolerance, with one fresh preconditioner per outer iteration. Candidates:
  `OMFDM_TANGENT_PRECONDITIONER=secant-amg` (A, default), `tangent-amg` (B,
  SPAI(0)), `tangent-ilut` (C, p=2, tau=1e-2). These are benchmark environment
  switches, not new general YAML settings. Nonfinite, unmet-tolerance and
  iteration-limit directions are rejected; one checked secant fallback is
  attempted, then the existing safe state is retained if that also fails.
- Removed the old approximate tangent assembly and the historical
  `OMFDM_PROBE_WNEWTON` drive that rebuilt AMG inside Krylov applications.
  Setting that obsolete switch now reports an explicit error. Historical
  implementations and measurements remain available in Git; no claim is made
  that their energy merit was an exact potential of the residual.
- Windows/MSVC and Linux/GCC Release CTest: **3/3 passed** on 2026-09-23.
  Windows release-contract smoke passed. The Linux build used native Linux
  TIFF explicitly because this WSL environment otherwise found Windows conda's
  `tiff.lib`; this was an environment/link selection issue. macOS is untested.
- Tangent regression: **192 directions**, best finite-difference step selected
  from 1e-4 through 1e-8, maximum relative Jv error **2.48439e-7** on Windows.
  Cartesian/polar, both polar storage orientations, periodic/antiperiodic,
  Dirichlet/Robin/radial Neumann profiles, saturation, material interfaces and
  an actual Newton update are covered. Scalar material derivative maximum
  relative error: **2.76283e-10**; table knots, clamp and extrapolation have
  separate branch/one-sided checks. These are small-case correctness tests,
  not evidence of large-case convergence or speed.
- The discrete-energy diagnostic gives dW/dt = -907.017 versus the putative
  scaled residual product -5595.68 on its fixture. They are **not identical**;
  energy remains an optional merit function, not an asserted exact gradient.
- AA regressions exercise acceptance in three cases and rejection/rollback
  in three cases, direct versus assembled residuals, repeated solves and the
  outer iteration limit. FGMRES A/B/C small solves and nonfinite-RHS rejection
  pass. Chunk regression observes two actual OpenMP threads in each of two
  chunks for a four-thread allocation and checks restoration of parent state.
- Waveform comparison tooling rejects unfinished/nonfinite/misaligned inputs,
  compares periodic centered-difference EMF, and does not shift or rescale
  waveforms. Its synthetic regression passes; no 248-point physical waveform
  has yet passed the adoption gate.

## Measurements to date (not an adoption decision)

The phase-1 binary is retained separately in `build_solver_speed`, associated
with `63e8c68`; phase-2 source/binaries use `build_solver_tangent`. Do not rebuild
the phase-1 directory during the alternating comparison. The manifest pins its
executable and DLL hashes even when the working source has advanced.

| Run | Commit | Strictly converged steps | Wall time | Peak working set | Peak private commit |
|---|---|---:|---:|---:|---:|
| Baseline 1, 12 threads | `25859a7` | 12/12 | 1213.776 s | 1,077,956,608 B | 1,073,070,080 B |
| Phase-1 trial 1, 12 threads | `63e8c68` | 12/12 | 710.117 s | 1,045,348,352 B | 1,041,260,544 B |

First-pair review: 41.495% wall-time reduction (1.709x speed), 3.025% lower peak
working set. Outer iterations increased from 1254 to 1263, so the main benefit
is cheaper iterations, not faster nonlinear convergence. Approximate profile
totals: assembly 141.4 -> 18.2 s, line search 473.6 -> 135.5 s. Timer scopes
differ slightly between revisions. Trial linear work is 441.1 s, including
248.9 s AMG setup and 148.4 s Krylov work. Reusing stale AMG is not justified.

Partial 12-point flux normalized RMS differences (A/B/C) are 0.010206%,
0.001623%, 0.001860%. Identical centered differences on the interior ten points
give EMF differences 0.026698%, 0.032182%, 0.033498%. No wrapping or harmonic
gate was applied to this partial revolution. These are not full-waveform passes.
Time-sampled private-memory medians in the later run quarters are approximately
554 MiB baseline and 743 MiB trial: workspace retention increased even though
the peak decreased. Trial quarters 2-4 remain roughly level, not a long-run
leak certification. Only one pair has finished and runs were separated in time;
medians/reproducibility remain pending.

Raw evidence is in the sibling directory
`bench_results/solver_speed_20260923_phase1/1_baseline/measurement.json` with the
manifest, process logs, step statuses and sampled memory history. Baseline outer
iterations are 108, 100, 107, 103, 115, 101, 98, 112, 105, 100, 99, 106; final
residuals are all below 1e-3. One partial trial was interrupted between sessions
and is retained for diagnosis, excluded from timing comparisons. Use `--resume`
to verify hashes and retain completed measurements, and `--stop-after-runs 1`
to pause only after the next full run. Source identity in the original manifest
is preserved on resume.

## Large-grid tangent screening

The next bounded experiment is `run_tangent_screen.py`: one cold-start step on
the same 1,339,200-DOF model, optimized secant then candidates A/B/C, sequentially.
Each uses the same phase-2 executable, max=300, tolerance=1e-3, chunks=1 and
12 threads. A 600-second diagnostic guard and 8-GiB hard guard bound each child
process. This is a feasibility diagnostic, not the 12-step adoption benchmark;
termination is recorded explicitly and cannot count as convergence or speedup.
No general solver defaults, material laws or preconditioner parameters change.

### Completed cold-start screen (2026-09-23)

Source `1e7fea6` (runner committed as `b1ba180`); raw results are in
`../bench_results/tangent_screen_20260923/diagnostic_summary.json`.

| Path | Wall seconds | Outer iterations | True final residual | Tangent accepted / attempted | Secant fallbacks | Peak private bytes |
|---|---:|---:|---:|---:|---:|---:|
| Optimized secant | 56.697 | 108 | 9.59006e-4 | n/a | 0 | 938,811,392 |
| A: secant AMG | 124.057 | 75 | 5.40273e-4 | 72 / 74 | 2 | 1,880,981,504 |
| B: tangent AMG | 115.988 | 80 | 9.05506e-4 | 77 / 79 | 2 | 2,175,987,712 |
| C: tangent ILUT | 600.116 (guard) | 29 completed updates | no final state | 23 / 29 | 6 | 2,496,786,432 |

A and B converge but are over twice as slow as secant on this cold start. B
reduces tangent Krylov iterations to 978 versus A's 2125, but that is insufficient
to pay for tangent assembly/setup. C was interrupted while the 30th outer
iteration was running; it did not hit the outer limit and has no final
convergence result. All three exceed the conservative twice-secant private
peak criterion (A only slightly). None qualifies for speed adoption from this
screen. This does not establish their 12-step warm-sweep performance.

The small-step ratchet is already documented in Phase BL sections 4.5-4.6.
That historical ray-minimizing search was NEGATIVE. It is not reintroduced.
The next finite comparison is the already-planned energy merit with B, using
the newly verified residual tangent and one preconditioner construction per
outer iteration. Energy is only a merit function: final true residual remains
the convergence gate, not energy decrease.

### Storage correction and bounded follow-up (2026-09-25)

Inspection found that FGMRES owned an unnecessary second copy of the entire
tangent operator with widened indices. Replace that Krylov-only copy with the
bundled AMGCL `zero_copy_direct` adapter borrowing the compressed Eigen CRS.
The const tangent stays alive/unmodified until FGMRES returns; the fresh AMG or
ILUT preconditioner continues to own its numerical storage. No hierarchy reuse,
material or solver-tolerance change is involved. Regression checks compare
copied/borrowed products, coefficient refresh, ownership and input immutability.

Preserved the original Windows executable/dependencies under
`build_solver_tangent_original_1e7fea6/Release` before rebuilding. The original
manifest hashes still identify that binary; do not resume the old manifest
against the newly rebuilt executable at its previous path.

Follow-up is bounded to residual-merit A/B to measure the storage change, then
energy-merit B against a fresh default-secant cold start. C's expensive ILUT
Krylov work is not plausibly fixed by removing a matrix copy, so C is not rerun
in this follow-up. These are diagnostics, not an automatic production change.

### Follow-up results (2026-09-25, `abe9ca9`)

| Group / path | Wall seconds | Outer iterations | True residual | Peak private bytes | Within 2x same-group secant peak |
|---|---:|---:|---:|---:|---|
| Borrowed-CRS control secant | 61.542 | 108 | 9.59006e-4 | 937,865,216 | yes |
| Borrowed-CRS A, residual merit | 148.571 | 75 | 5.40273e-4 | 1,592,508,416 | yes |
| Borrowed-CRS B, residual merit | 140.264 | 80 | 9.05506e-4 | 1,885,573,120 | no (slightly over) |
| Energy-group control secant | 70.088 | 108 | 9.59006e-4 | 937,627,648 | yes |
| Borrowed-CRS B, energy merit | 46.064 | 19 | 8.69474e-4 | 1,864,040,448 | yes |

Raw evidence: `../bench_results/tangent_nocopy_20260925/` and
`../bench_results/tangent_energy_20260925/`, including executable/input hashes.
A/B residual-merit outer counts, total tangent inner counts (2125/978), accepted
directions (72/77), fallback counts (2/2) and final residuals agree with the
original copied-operator runs. Private peaks fell by approximately 288 MB and
290 MB. The storage correction has evidence of numerical equivalence and lower
memory, but **not of a measured speedup** in these runs.

External CPU activity was observed during the follow-up (a short process-CPU
sample showed Docker/Python/WSL and other applications active). No unrelated
process was stopped. Baseline timings also changed between groups/dates.
All wall times here are feasibility diagnostics, not controlled adoption data.

B + energy accepts 16 of 18 tangent attempts, falls back to secant twice and
uses 379 tangent inner iterations. Its cold-start reduction versus its control
is 34.277%, with a true converged final state. This is a promising candidate for
the 12-step experiment, not evidence of warm-sweep speed, waveform quality or
an exact energy/residual gradient relationship. It remains opt-in.

Added 32 tangent regressions combining actual material-image slides, both polar
storage orientations, periodic/antiperiodic seams and r=0 with Dirichlet axis.
The cached tangent retains its allocation, matches a fresh assembly after each
slide and matches finite-difference Jv: maximum relative error 3.55914e-11 on
Windows. The sliding fixture seeds static/nonlinear mu for the moved image and
then refreshes B/H/mu; it is not a substitute for the end-to-end warm sweep.
Windows and Linux CTest pass 3/3; Windows release-contract passed for the
borrowed-CRS implementation. macOS remains untested.

`benchmarks/solver_speed/ieej_12step_tangent_energy.yaml` differs from the primary
secant input only in `jacobian: tangent` and `line_search_objective: energy`.
Use candidate B (`--trial-preconditioner tangent-amg`) and the same current
executable for both paths. First run one complete baseline/trial pair with
`--repeats 3 --stop-after-runs 2`; pause to check strict convergence before
resuming the remaining pairs. External-load observations prevent treating a
single pair as the final performance gate.

Still pending: three paired timings and medians, 4-thread/cold/35H300 checks,
large-case tangent warm-sweep cost and memory, long-sweep retention,
full-waveform gates, and macOS build/numerical regression.
The numerical gates above do not waive any of these acceptance requirements.
