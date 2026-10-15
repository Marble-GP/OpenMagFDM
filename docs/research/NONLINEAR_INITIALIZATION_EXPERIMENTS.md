# Nonlinear initialization experiments

Branch: `feature/nonlinear-initialization-experiments`, from `e80f20a`.
Updated requested sequence: direct warm-start full-revolution/large-displacement
validation -> angle prediction -> safeguarded Anderson combination. Magnetic
circuit initialization is deferred, not part of this sequence. These are separate
ablations, not a simultaneous change to the accepted solver. No public YAML,
material law, convergence tolerance, release defaults or AMG reuse changes.
Priority 4 remains rejected/OFF; the baseline is tangent-AMG + energy, mask 6.

## A. Direct transient warm start (12-step and full-sweep validation passed)

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

### Next gate: full revolution and large displacement (prepared 2026-10-11)

The completed direct-initialization work and three-pair evidence are committed
as `03a416f`. The next harness is `run_initialization_validation.py`; execution
instructions are in the benchmark README. It reuses the same preserved binary
and changes neither the solver nor public defaults.

| Test | Pixels / step | Mechanical degrees / step | Points per method | Quality gate |
|---|---:|---:|---:|---|
| Full revolution | 12 | 1.4516 | 248 | Flux, periodic EMF, harmonics |
| Large displacement 1 | 48 | 5.8065 | 12 | Aligned-angle flux only |
| Large displacement 2 | 124 | 15 | 12 | Aligned-angle flux only |
| Large displacement 3 | 248 | 30 | 12 | Aligned-angle flux only |

Each pair compares LINEAR and DIRECT with tangent AMG + energy + mask 6;
priority 4 and Anderson stay OFF. Large displacement can change iteration
history, so cross-method history equality is not required. Strict convergence
of every step and correct initializer selection are required. The large-step
cases do not claim accurately resolved EMF/harmonics, even if the sampled
positions span a revolution. Review total time, per-step iterations/residuals,
peak memory and background CPU load; one pair is not a repeated speed claim.

Binary/DLL/input/harness hashes and the power-plan GUID are pinned. Failures
preserve diagnostics and stop launches; no silent retry or solver change is
allowed. Reference and candidate must both be strictly converged before any
waveform acceptance. The new harness has unit and mocked orchestration tests
for malformed/missing initializer records, quality/convergence/memory failures,
prepare-only, strict resume, changed inputs and incomplete-attempt preservation.
Mocked solver runs are harness tests, not numerical/performance evidence.

Preparation at `../bench_results/direct_validation_20261011_01` validated the
preserved binary and inputs and generated all four configurations without
launching a solver. On Windows, the expanded benchmark Python suite passed
28/28 and the preserved direct-start build passed CTest 3/3 on 2026-10-11.
No rebuild was performed; Linux/macOS checks remain pending.

The prepared gate was executed on 2026-10-15; completed results follow.

### Completed full-revolution and large-displacement gate (2026-10-15)

The eight sequential runs finished at 07:47:46 JST after starting at 01:20:27
(6 h 27 min 20 s including harness overhead). `progress.json` is `complete`,
and all four pairs pass their numerical, initializer and memory gates.
All **568/568 steps strictly converged** below 1e-3. DIRECT used the previous
converged state at all 280 eligible steps; each first step retained linear
initialization. Priority 4 and Anderson remained OFF. No binary was rebuilt.

| Step angle / points | LINEAR seconds | DIRECT seconds | Time reduction | Outer iterations LINEAR -> DIRECT |
|---|---:|---:|---:|---:|
| 1.4516 degrees / 248 (full period) | 10,940.629 | 6,657.868 | **39.15%** | 7,531 -> 4,413 |
| 5.8065 degrees / 12 | 1,276.453 | 596.615 | 53.26% | 702 -> 385 |
| 15 degrees / 12 | 1,201.400 | 795.055 | 33.82% | 657 -> 460 |
| 30 degrees / 12 | 875.726 | 870.586 | 0.59% (not a demonstrated gain) | 589 -> 621 |

The full period falls from 182.34 to 110.96 minutes, saving 71.38 minutes.
Maximum outer iterations at one step fall from 63 to 33. This extends the
previous three-pair 12-step result (38.25% median reduction) to a complete
revolution with waveform validation, rather than extrapolating cold starts.
Each new displacement has only one timing pair, not three repeated pairs.

Full-period errors below are maxima across all three coils; all values in
this table are **percent**, not the dimensionless ratios in the JSON files.
The comparator uses identical periodic centered differences, includes the
wraparound, uses fundamental bin 2 and does not fit amplitude or phase.

| Quantity | Observed maximum | Limit |
|---|---:|---:|
| Flux normalized RMS difference | 0.001742% | 0.1% |
| EMF normalized RMS difference | 0.023169% | 0.5% |
| Flux harmonic amplitude difference / fundamental | 0.000362% | 0.1% |
| EMF harmonic amplitude difference / fundamental | 0.006172% | 0.1% |

The 48/124/248-pixel cases have maximum aligned-angle flux RMS differences
0.003604%, 0.001855% and 0.001636%, respectively (limit 0.1%). They do **not**
validate coarse-sampled EMF or harmonics. Reference and trial are both strictly
converged; this demonstrates agreement with the control, not JMAG accuracy.

Across all runs, peak RSS is at most 1.7264 GiB and peak private memory at most
1.7236 GiB. All paired limits pass (8 GiB and <=2x reference); small increases
remain, including about 20 MiB for the 30-degree pair. These process peaks do
not measure kernel pools or establish absence of a leak on their own.

Estimated non-solver busy cores (LINEAR / DIRECT) were 1.485 / 1.441 for full,
1.411 / 1.488 for 48 pixels, 1.419 / 1.338 for 124 pixels, and **3.022 / 2.390**
for 248 pixels. Light UI/MCP development and tests overlapped the early part;
heavy packaging, rebuilds and other solver experiments were deferred. Do not
describe these as completely idle measurements. At 30 degrees the 5.14-second
timing difference is inconclusive under unequal background load; outer
iterations increase by 5.43% and the worst step increases from 86 to 95.
The result supports reduced usefulness at large displacement, not a claim
of divergence or an angle-independent acceleration guarantee.

Decision: retain DIRECT as the selected research initialization candidate for
small-angle sweeps and advance to the separately gated angle predictor, then
safeguarded Anderson. Do not enable a new default or an angle-based heuristic
from this single model. No further long run started automatically; the circuit
initializer remains deferred. Linux/macOS numerical regression and broader
material/thread-count coverage remain outside this completed Windows gate.

Evidence: `../bench_results/direct_validation_20261011_01/` (original logs,
measurements and fields), versioned [summary](results/direct_validation_20261015_summary.json)
and [provenance](results/direct_validation_20261015_provenance.json). The latter
pins binary/DLL/input/harness hashes and hashes of all eight measurements,
logs, conditions and flux tables. Binary source provenance is distinguished
from the later harness-preparation commit; the executable is unchanged from
the previous experiment. On 2026-10-15, benchmark tests passed 28/28 and the
preserved Windows build passed CTest 3/3 without a rebuild.
The Windows release-contract smoke also passed using an isolated copy of the
same executable/DLLs; the pinned build and original benchmark outputs were
not overwritten. This is a solver contract check, not MCP packaged acceptance.

## B. Angle prediction (completed; rejected 2026-10-16)

Start with two strictly converged, adjacent, equal-angle-step states in the
same coordinates and compatible gauge: A_pred = A_prev + beta*(A_prev-A_older).
Compare beta=1 against direct previous-state initialization, with current-law
true-residual evaluation and a reject-to-direct safeguard. Reset history on
failure, restart/chunk boundary, incompatible grid/boundary/gauge or step size.
Keep initial validation overhead in total time. Rotor-frame transport is a
separate hypothesis, not bundled into this first predictor.

### Bounded implementation and safeguards

`OMFDM_NK_INITIALIZATION=predict` extends the research-only switch; the default
remains `linear`, with `direct` unchanged as the ablation control. Prediction
uses beta=1 in fixed coordinates. It first evaluates the direct seed with current
B/H/mu, then evaluates the extrapolation with its own current B/H/mu. Accept
only a finite prediction with a strictly smaller true residual; otherwise
restore the direct seed and recompute its fields. No frozen-mu merit, rotor-frame
transport, EW adjustment, line-search change or AMG hierarchy reuse is added.
First-loop and final true-residual checks remain. The extra probes are charged
to initialization time and process wall time, not hidden from the comparison.

Prediction eligibility is tracked with per-sweep/chunk metadata, reusing the
two existing solution vectors rather than retaining an additional long history.
Both preceding states must be adjacent and strictly converged. Failure, skipped
steps, restart/repeated analysis and a changed mesh/boundary/profile/gauge or
displacement signature invalidate eligibility. NK entry consumes both one-shot
flags before validating the experiment mode, including the exception path.
The first step of a chunk now explicitly cannot use a preceding-run direct seed.

The initial supported motion is deliberately one constant angular band on a
polar grid (both storage orientations), with at least one radial Dirichlet
boundary fixing the gauge. Rectangle/variable motion, multiple moving bands,
Cartesian motion and unanchored gauges retain the direct/linear fallback.
The signature includes resolved pixel displacement, grid coordinates, orientation,
boundary signs/scalars/profiles and the slide region/wrap mode. No-table materials,
coarsened states and explicit DD/file initial states retain their previous precedence.

`[NKINIT]` additionally records `prediction_eligible`, `prediction_attempted`,
`prediction_used` and `prediction_residual`. Residuals are logged at 17-digit
precision; stream precision is restored afterward. `direct_used=1` means the
direct transient bypass was eligible and evaluated, including when the final
seed is the accepted prediction. Use `prediction_used` to distinguish the latter.

Historical difference: `d37f02f` extrapolated the **linear AMG** initial guess
and found plain AR(1) more useful than rotated/smoothed guesses. This experiment
seeds the **nonlinear Newton** solve directly. A smaller initial residual does
not prove fewer Newton iterations: only end-to-end paired measurements can
decide whether the added predictor/probe cost is worthwhile.

### Verification and launch

Preserve `build_nonlinear_initialization/Release/MagFDMsolver.exe`. The new
`build_nonlinear_prediction/` uses the same MSVC 19.44 toolchain, optimization
settings and runtime DLL hashes, with dependency installation disabled.
Windows Release build, CTest 4/4 and release-contract smoke passed. CTest includes
56 predictor acceptance/rejection/fallback fixtures over both polar orientations,
periodic/antiperiodic seams and the axis, plus history/signature/exception checks.
An added actual sliding-sweep fixture covers both orientations, AA OFF/ON and
repeated use of the same analyzer (32 steps). Existing residual/Jv/AA regressions
remain enabled. Benchmark Python tests passed 38/38, including 10 new harness
tests with mock runs; mocks are not performance evidence. Linux/macOS unverified.

At 19:38 JST on 2026-10-15, launched `run_prediction_pairs.py --stage predictor`
at `../bench_results/angle_prediction_20261015_01` after prepare-only validation.
It runs DIRECT/PREDICT, PREDICT/DIRECT, DIRECT/PREDICT sequentially on the same
new binary: IEEJ-D, 12 points, 12-pixel steps, 12 threads, chunk 1, mask 6,
tangent-AMG + energy, tolerance 1e-3, max 300, Anderson OFF, priority 4 OFF.
The estimate is 30–40 minutes, not a completion guarantee. No speed claim yet.

The DIRECT controls must match the archived `1_direct` numerical histories and
all 12 Az/Mu TIFF hashes; each mode must repeat its own histories and hashes.
Every step must strictly converge and all expected initialization markers must
be present. Pair quality uses aligned flux RMS <=0.1%; this short sweep does
not establish periodic EMF/harmonic quality. Each run has a 7200-second / 8-GiB
guard; candidate peaks must also be <=2x control. Binary/DLL/input/scripts/power
plan are pinned. A mismatch or failed gate retains evidence and stops launches.
Incomplete attempts are not overwritten; no automatic restart or stage chaining.

Review per-pair and median time, iteration counts, predictor acceptance/rejection,
initialization cost, memory and background load before selecting the initializer.
If prediction is neutral or worse, retain DIRECT for the Anderson ablation.
Only useful candidates advance to the separate full-period quality gate and
large-displacement coverage; these have not been run for prediction yet.

### Completed three-pair review and rejection (2026-10-16)

All six processes finished successfully at 20:11 JST on 2026-10-15. All 72
steps strictly converged; maximum final residual was 9.2812645201e-4. Every
run used 211 outer and 3081 inner iterations. All 24 Az/mu TIFF hashes per
run match the archived DIRECT reference, as do step-status records and the
iteration/residual/inner-iteration/line-search/outcome signatures. Aligned
flux differences are zero for all three coils in every pair.

| Pair | DIRECT seconds | PREDICT seconds | Raw time reduction | Predictions used / attempted |
|---|---:|---:|---:|---:|
| 1 | 345.384 | 333.550 | 3.43% | 0 / 10 |
| 2 | 332.317 | 320.019 | 3.70% | 0 / 10 |
| 3 | 341.554 | 321.702 | 5.81% | 0 / 10 |
| Median | 341.554 | 321.702 | 5.81% | 0 / 30 overall |

Prediction was attempted at steps 3-12 and rejected every time: its true
residual was 2.2211-2.2323 times the direct seed residual. The fixed-coordinate
beta=1 extrapolation therefore never changed the accepted initial state.
The reject-to-direct safeguard worked; this is NOT a nonlinear-convergence
improvement. Total initialization cost increased in every pair (by 0.371,
0.832 and 0.810 seconds), including probes and restoring current-law fields.

Estimated non-solver busy cores were 3.344/2.970, 2.081/2.186 and 2.397/2.267
(DIRECT/PREDICT). The raw total-time differences cannot be attributed to an
accepted predictor, nor is a single background-load explanation established.
Do not credit the 5.81% median difference as an algorithmic speedup. Peak RSS
is about 1.725 GiB for both modes; these peaks alone are not a leak test.

User decision: reject this bounded predictor and retain DIRECT. The research
switch and regression fixtures remain for reproducibility, disabled in the
selected path; no public default changes. No damping/transport/acceptance-rule
search is added. This result does not rule out every possible angle predictor.
The raw `passed` field means convergence/numerical/safety gates passed, not
adoption. No new full-revolution or periodic EMF claim is made for prediction.

Evidence: original `../bench_results/angle_prediction_20261015_01/`, versioned
[summary](results/angle_prediction_20261015_summary.json) and
[provenance](results/angle_prediction_20261015_provenance.json). Original
outputs, manifest, scripts and binary are preserved; do not resume the finished
experiment merely to inspect it. The binary was built from base `e8df23f` plus
the then-uncommitted solver changes pinned in that manifest.

Windows CTest 4/4, benchmark Python tests 38/38 and the release-contract smoke
(isolated copy of the pinned executable/DLLs) were rerun on 2026-10-16.
Linux/macOS remain unverified for these initialization changes.

## C. Anderson combination (DIRECT selected; final ablation prepared 2026-10-16)

Use DIRECT (angle prediction rejected) with existing safeguarded Anderson, one fixed
small depth; compare OFF/ON before changing other parameters. Preserve the
true-residual/energy acceptance, bounded correction and rollback. Count extra
QR/candidate evaluation costs, not merely outer iterations saved.

`run_prediction_pairs.py --stage anderson --initializer direct|predict
--predictor-reference <completed predictor directory>` is the second, independent
stage. It requires a completed passing predictor experiment using the same binary
and uses the selected initializer's archived result as a numerical OFF control.
Only the AA block changes to `enabled: true, depth: 3, beta: 0.3`. Run OFF/ON,
ON/OFF, OFF/ON with the same physical inputs and convergence gates. Record QR/
candidate-evaluation time, accepted/rejected offers and actual evaluation counts;
the ON path must actually exercise Anderson. Existing correction bounds,
true-residual/energy checks, rejection reset and best-state rollback are unchanged.
The actual small-sweep OFF/ON test above passed, but no large-model Anderson
timing or adoption result is claimed before this second stage runs and is reviewed.

After the predictor rejection checkpoint `d82fee0`, the user authorized this
as the final bounded hypothesis comparison. Use the unchanged
`build_nonlinear_prediction/Release/MagFDMsolver.exe` (SHA256
`4a9a0f1d893234ff544fc51d46a61372b55dd347d96f6ae010144635a05906a1`)
and a new directory `../bench_results/direct_anderson_20261016_01`.
The solver implementation, EW/line search, AA correction/rollback safeguards,
AMG rebuild policy and all public defaults remain unchanged. Prediction and
priority 4 are OFF. The only paired configuration change is the AA block.

Four additional mocked orchestration regressions verify DIRECT selection after
a rejected predictor, prepare-only/AB-BA-AB ordering/duplicate-free resume,
rejection of AA ON with no evaluations or OFF with unexpected evaluations,
and stopping on an out-of-tolerance final residual without a speed claim.
The benchmark suite passed 42/42 on 2026-10-16. These mocks do not count as
real-model performance evidence. The preceding CTest 4/4 and isolated
release-contract smoke passed with the same binary; no rebuild is needed.

Acceptance review must include AA accepted/rejected counts, evaluation/QR
overhead, outer/inner iterations, total time including output completion,
aligned flux and memory/load, not just a `passed` summary flag. Only a useful
candidate proceeds to a separately reviewed full-period/large-displacement
gate. If it is not useful, retain DIRECT alone and close this hypothesis series;
do not start an unrestricted depth/beta search or revive angle prediction.

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
