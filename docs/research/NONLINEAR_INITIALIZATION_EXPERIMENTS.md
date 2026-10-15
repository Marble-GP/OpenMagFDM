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
