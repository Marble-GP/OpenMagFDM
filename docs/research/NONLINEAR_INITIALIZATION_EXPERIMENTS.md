# Nonlinear initialization experiments

Branch: `feature/nonlinear-initialization-experiments`, from `e80f20a`.
Updated requested sequence: direct warm-start full-revolution/large-displacement
validation -> angle prediction -> safeguarded Anderson combination. Magnetic
circuit initialization is deferred, not part of this sequence. These are separate
ablations, not a simultaneous change to the accepted solver. No public YAML,
material law, convergence tolerance, release defaults or AMG reuse changes.
Priority 4 remains rejected/OFF; the baseline is tangent-AMG + energy, mask 6.

## A. Direct transient warm start (12-step validation passed; full sweep pending)

Research environment `OMFDM_NK_INITIALIZATION=linear|direct` (default linear).
The standard NK initializer solves a linear problem each step even when a
previous Az is available. The direct variant instead uses the immediately
previous strictly converged solution as Az, in fixed coordinates, and skips
that linear initialization. It does not extrapolate or rotate Az.

Eligibility is one-shot, analyzer-local and set immediately before a transient
solve; it is consumed at NK entry. First steps/chunk starts, a failed preceding
step, wrong-sized/nonfinite history, coarsened solves and nonlinear materials
without valid B-H tables retain normal initialization. Explicit DD/file initial
states keep precedence. The material values and source maps are current; B/H/mu
and the true residual are recomputed before using the direct seed. A nonfinite
probe restores the old Az/mu and uses the linear initializer. A finite seed is
an experimental starting point, NOT a convergence guarantee or a claim that it
is better than the linear seed. No residual improvement claim is made without
measuring both methods. Final fresh-field residual and existing rollback gates
are unchanged. Accepted-field reuse is not introduced.

`[NKINIT]` records requested mode, eligibility, actual usage, seed residual and
initialization time. This time includes the nonlinear setup before the NK loop;
end-to-end comparisons include probe, initialization, output and process exit.

The preserved `build_tangent_workspace/Release/MagFDMsolver.exe` is not rebuilt.
New builds use `build_nonlinear_initialization/`, matching the prior MSVC/vcpkg
dependencies. The dependency installation step is disabled to preserve those
libraries. Windows numerical fixtures cover both polar layouts, periodic and
antiperiodic seams, Cartesian, invalid history, ineligible/no-table fallback and
one-shot invalidation. An actual sliding sweep is still required after these
small fixtures; they do not alone establish speed or waveform quality.

Before large trials: run CTest and the existing benchmark unit tests. Compare
the new LINEAR path against the preserved candidate's numerical histories.
Then alternate LINEAR and DIRECT for three 12-step pairs (AB/BA/AB), same
binary, 12 threads, chunk 1, tolerance 1e-3, max 300, mask 6, Anderson OFF.
Stop on unconverged results or lack of actual direct starts. Record outer/inner
iterations, initialization/AMG/Krylov costs, memory, load and all input hashes.
Different initializers need not have identical iteration histories. Do not
use a partial sweep to claim periodic EMF equivalence; advance only a useful
candidate to the existing 248-point waveform gate.

Windows Release build and CTest passed 3/3 on 2026-10-10, including 20 direct
initialization/fallback fixtures and the existing 324 stencil / 192 tangent /
32 axis-sliding checks. Runtime DLL hashes match the preserved control.
Linux/macOS have not been rebuilt for this branch.

Reproduce the first paired experiment with the benchmark Python environment:

```text
python benchmarks/solver_speed/run_initialization_pairs.py --exe build_nonlinear_initialization/Release/MagFDMsolver.exe --reference ../bench_results/selected_secant_20261009_01 --output ../bench_results/direct_initialization_20261010_01 --prepare-only
python benchmarks/solver_speed/run_initialization_pairs.py --exe build_nonlinear_initialization/Release/MagFDMsolver.exe --reference ../bench_results/selected_secant_20261009_01 --output ../bench_results/direct_initialization_20261010_01 --resume
```

The control compares all 12 Az/mu TIFF hashes and numerical histories against
the archived candidate. Direct runs must converge at all 12 steps and actually
skip initialization at the 11 eligible subsequent steps. A failed gate stops
further launches for review. No automatic predictor/Anderson/circuit experiment
is launched until these results have been reviewed.

### Parser-only resume after first pair

The first pair completed (528.150 seconds / 353 outer iterations for LINEAR,
328.005 seconds / 211 iterations for DIRECT, both 12/12 converged). The runner
then stopped because its line-start-only parser missed 11 NKINIT records
appended to updateMu output. These are real direct starts, not solver failures.
The reported `running` state was stale because the error handler only wrote
`last_error.json`. Do not treat this single pair as a validated speed claim.

The corrected parser searches for the NKINIT marker independently of line
position; a synthetic inline-log regression covers all 12 records. Errors now
also publish `stopped_for_review`. Original summary/progress/error files and
the original runner are archived before reaggregation. The original manifest,
process logs and measurements are unchanged. `--resume --accept-runner-change`
permits only an explicitly audited runner-hash change, checking every other
pin (binary, DLLs, inputs, measurement engine, helpers, power plan, settings)
unchanged. `parser_fix_resume.json` records both runner hashes. Completed
measurements are revalidated/reused; the remaining four processes run normally.
This resumed session must be identified when reviewing timing variability.

### Completed three-pair results (reviewed 2026-10-11)

All six runs completed: 72/72 steps strictly converged. Within each method,
iteration/residual histories repeat at log precision. All LINEAR controls also
matched the preserved candidate's 12 Az/mu TIFF hashes. DIRECT used 11 warm
starts per run; first steps retained linear initialization. Priority 4 stayed OFF.

| Pair | LINEAR seconds | DIRECT seconds | Reduction |
|---|---:|---:|---:|
| 1 | 528.150 | 328.005 | 37.90% |
| 2 (resumed) | 531.208 | 327.230 | 38.40% |
| 3 (resumed) | 544.480 | 341.017 | 37.37% |
| Median | 531.208 | 328.005 | **38.25%** |

Every run had 353 -> 211 outer iterations (40.2% reduction) and 4239 -> 3081
inner iterations. Maximum final residuals were 9.1956204992e-4 / 9.2812645201e-4.
Peak private memory remained about 1763 MiB. Initializer time fell from about
18–19 seconds to 2.6–3.0 seconds, much less than the approximately 200-second
total saving: fewer nonlinear iterations account for most of the improvement.
Non-solver load estimates were 1.74–2.18 busy cores. The interrupted session
and background load are recorded, not ignored; no cross-day cumulative speed
claim is inferred. No direct-start full-period waveform gate has yet passed.

Evidence: `../bench_results/direct_initialization_20261010_01/`; versioned
[summary](results/direct_initialization_20261010_summary.json) and
[provenance](results/direct_initialization_20261010_provenance.json).
Windows CTest 3/3 and release-contract passed. Benchmark Python tests passed
15/15 after the parser-only repair. Linux/macOS remain unverified for this change.

Historical distinction: `c2b5029` seeded the LINEAR AMGCL solve with current Az;
`bf07f15` reused nonlinear mu; `d37f02f` added prediction to the linear transient
path. `8449a46` / `a081875` already bypassed linear initialization for explicit
file/DD seeds, and `f41dc86` safeguarded transient history. This experiment
extends direct NEWTON initialization to normal strictly converged transient
history; it is not the first warm-start feature in the product.

## B. Angle prediction (not implemented)

Start with two strictly converged, adjacent, equal-angle-step states in the
same coordinates and compatible gauge: A_pred = A_prev + beta*(A_prev-A_older).
Compare beta=1 against direct previous-state initialization, with current-law
true-residual evaluation and a reject-to-direct safeguard. Reset history on
failure, restart/chunk boundary, incompatible grid/boundary/gauge or step size.
Keep initial validation overhead in total time. Rotor-frame transport is a
separate hypothesis, not bundled into this first predictor.

## C. Anderson combination (not implemented/evaluated here)

Use the selected initializer with existing safeguarded Anderson, one fixed
small depth; compare OFF/ON before changing other parameters. Preserve the
true-residual/energy acceptance, bounded correction and rollback. Count extra
QR/candidate evaluation costs, not merely outer iterations saved.

## D. Magnetic-circuit initializer (deferred; excluded from current plan)

Partition along flux paths (teeth, yoke, bridges, magnets, gap), not just RGB
connected components. Same material does not imply uniform saturation. Solve
a bounded nonlinear reluctance network with the original B-H data, use its
estimated B to seed mu, then run the unchanged fine-grid solver to convergence.
Do not confuse circuit scalar magnetic potential with Az. A direct Az lift
needs a separate compatible field reconstruction. Circuit construction/solve
cost counts toward total time. Initial focus is cold starts/geometry changes;
improving only the first of 248 steps may have little sweep benefit.

Prior work: `bf07f15` already reuses accepted transient mu; `d37f02f` predicts
the linear transient solver; `8449a46` permits explicit file/DD initial Az.
The old FAS/coarse-operator failures do not establish failure of coarse-only
initialization followed by a full fine-grid solve. Conversely, they prohibit
assuming coarse mu or frozen-mu residuals are accurate. Local homotopy evidence
already exists (`homotopy_poc`: 46 soft-stage + 8 true-stage vs 54 cold
iterations, with older tolerances); do not present that as a new speed win.
