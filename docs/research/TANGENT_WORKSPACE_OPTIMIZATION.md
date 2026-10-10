# Tangent workspace experiments (2026-09-26)

Branch: `feature/tangent-workspace-optimization`, parent `6e6f644`.
Research only: no public YAML/default change, merge, tag, push or release.

## Reference and prior evidence

The immutable baseline source is the detached sibling worktree
`../OpenMagFDM_tangent_baseline_6e6f644`. Its `build_baseline/Release` is a
preserved **copy**, not a rebuild, of the validated `build_solver_tangent/Release`
binary and DLLs. Its copied CMake cache describes that original build directory.
Solver SHA256: `c83fd924d7e8cb7eef114f6f1b4740037f73388c4ab8d8afe72ea5349b4b03ec`,
verified against the September 25 manifest. Do not reconfigure this archived
build directory. Trial builds use `build_tangent_workspace` with the same MSVC,
Release/AVX2/FP flags and dependency paths.

The completed September 25 sweep in
`../bench_results/tangent_energy_sweep_20260925` was one pair, not three medians:

| Path | Wall seconds | Converged steps | Outer iterations | Peak private bytes |
|---|---:|---:|---:|---:|
| Optimized secant | 732.5634 | 12/12 | 1263 | 1039679488 |
| Tangent AMG + energy merit | 656.9744 | 12/12 | 353 | 1933287424 |

The preliminary reduction is 10.3184%, below the 15% adoption target. Tangent
Jacobian work was 201.746 s; 325/341 tangent directions were accepted and 16
used the checked secant fallback. External load and full-waveform validation
remain outstanding. Energy is a merit function, not a proven potential of R.

| Hypothesis | Prior implementation | Difference in this branch |
|---|---|---|
| 1: tangent preparation | Existing material LUT/PCHIP caches; repeated per-cell lookups and per-neighbor norm/chain calculations | Resolve material references once per NK solve; update two compact chain coefficients per cell each outer iteration; retain shared stencil/AD |
| 2: lazy fallback | Corrected secant copy built on every tangent attempt | Build the identical copy/correction only after tangent failure; retain ordinary secant residual/preconditioner matrix |
| 3: Krylov storage | New FGMRES(20) and buffers every outer iteration | Reuse storage within one NK solve only, zero initial direction, fresh AMG each iteration; release before fallback and solve completion |
| 4: accepted fields | Recompute B/H/mu at every iteration start | Reuse only accepted LS state and only with valid B-H tables for every nonlinear material; invalidate at writes, AA, rollback; force final evaluation |

This does not repeat `65e160a`'s stale numerical AMG reuse or `b0b5ed9`'s
frozen-mu candidate residual. No Krylov-subspace recycling, material-law change,
new line-search tuning, ILUT rescreen, or AMR/DD change is included.

## Switches and lifetime

`OMFDM_TANGENT_OPTIMIZATIONS` is a research-only integer mask 0..15, default 0:
bit 1 = priority 1, bit 2 = priority 2, bit 4 = priority 3, bit 8 = priority 4.
Thus 6 is priorities 2+3, 7 is 1+2+3, and 15 includes all four. Invalid masks
fail explicitly. Existing `OMFDM_TANGENT_PRECONDITIONER=tangent-amg` selects B.
No YAML item was added. No optimization is automatically enabled by tangent mode.

All workspaces are owned by an individual NK invocation, never global or shared
between chunks. Material references are valid only while the image and material
maps are immutable inside that invocation; slides/another solve create a new
workspace. Shape/boundary changes rebuild material preparation. Numeric chain
coefficients are refreshed every iteration and zero derivatives retain graph
connectivity. FGMRES storage contains no borrowed operator or preconditioner.

For priority 4, no-table `H=B/mu` is history dependent, so skipping the update
would change the existing iteration. Such solves are ineligible. AA and LS
failure conservatively invalidate reuse. First and final evaluations are never
skipped. Failure paths preserve the existing best evaluated state.

## Measurement and reproduction

Keep builds/tests separate in time from measurements. Use the existing isolated
Python environment `build_solver_speed/bench_py312/Scripts/python.exe` with
the pinned benchmark requirements. Paths below are relative to the repository;
the runner resolves them before launching a child.

```powershell
build_solver_speed/bench_py312/Scripts/python.exe benchmarks/solver_speed/run_workspace_screen.py `
  --baseline ../OpenMagFDM_tangent_baseline_6e6f644/build_baseline/Release/MagFDMsolver.exe `
  --baseline-repo ../OpenMagFDM_tangent_baseline_6e6f644 `
  --trial build_tangent_workspace/Release/MagFDMsolver.exe --trial-repo . `
  --config benchmarks/solver_speed/ieej_12step_tangent_energy.yaml `
  --image ../bench_results/superlinear_poc/20260531_ISEEJ-D-model_section2_polar.png `
  --output ../bench_results/workspace_cold_20260926 --steps 1 --threads 12 --masks 0 1 2 4 6
```

Use a new output directory with `--steps 12 --masks 1 6 7` for the sweep screen.
Compare the selected mask against the same mask plus 8 separately; do not
interpret cold-start ranking alone as a sweep adoption decision. Resume checks
all inputs, binary/DLL hashes and masks. Incomplete directories are preserved.
Screening stops for review after a nonconverged run. The diagnostic guard is
600 s per requested step and 8 GiB; a guarded run is never a speed success.

`run_comparison.py` additionally accepts `--baseline-preconditioner`,
`--baseline-optimizations` and `--trial-optimizations` for paired medians.
For the selected mask, alternate baseline/trial three times and obtain three
same-condition secant controls. Keep the physical inputs and output settings
unchanged. Source identities, flags, input/binary/DLL hashes and CMake cache are
saved. Results include OS/sample peaks and approximate non-solver busy-core
load (system user+kernel CPU seconds minus last sampled child CPU seconds).
This estimate is not a frequency/thermal-control guarantee.

NKPROF retains existing aggregates and adds material preparation, AD-row
assembly, fallback construction, FGMRES allocation/reset and preconditioner
times; workspace allocations/reuses and field updates/reuses/invalidations are
counted. New timings are **nested breakdowns**, not additional wall time to sum.
In lazy mode fallback construction is inside the linear phase. AMG destruction,
field history and other unclassified costs remain included in iteration and
end-to-end wall times. First-call costs are included, not amortized away.

## Verification and adoption status

Implementation commits are separated by hypothesis; all switches remain off.
Numerical tests compare cached/fresh tangents, changing iterates, actual slides,
FGMRES fresh/reused solves including zero RHS, tolerance/size changes, invalid
inputs/setup failure, and 64 end-to-end optimization combinations. They cover
both polar orientations, antiperiodicity, Cartesian, energy/residual merit,
AA, non-idempotent materials, LS exhaustion and max-iteration rollback.

Acceptance remains the parent research protocol: true converged steps, smooth
Jv error <=1e-5, normalized residual equivalence <=1e-11, reproducible paired
timings, >=15% speed reduction versus optimized secant, memory <=2x secant and
<=8 GiB. Target no increase versus current tangent peak. Only selected candidates
advance to 4-thread/35H300 and 248-point periodic flux/EMF/harmonic comparisons.
No full-waveform or adoption pass is inferred from the short screening runs.

At implementation checkpoint `ca1027a`, Windows Release and Linux GCC builds
passed, CTest passed 3/3 on each, and the Windows release-contract smoke passed.
Windows numerical evidence: 324 stencil comparisons, max normalized residual
error 1.84761e-17; 192 tangent directions, max best FD-Jv error 2.48439e-7;
32 axis/sliding cases, max FD-Jv error 3.55914e-11. The 64 optimization
comparisons exercised 40 field reuses, 168 Krylov reuses and 8 LS-exhaustion
cases, with matching final Az/mu/residual and outer iteration counts.

macOS is not available locally and remains unverified; no remote CI is
dispatched by this local-only experiment.

## Cold-start ablation results

Evidence: `../bench_results/workspace_cold_20260926/` (manifest at `c1fc3eb`,
solver implementation `ca1027a`). All six runs strictly converged in 19 outer
iterations with final residual 8.6947404067e-4. For every trial, the recorded
per-iteration residual, inner iteration count, LS trials and outcome match the
immutable baseline **at log precision**. This is not a full-field/waveform gate.

| Case | Wall seconds | Peak private MiB | FGMRES allocations / reuses | Estimated other busy cores |
|---|---:|---:|---:|---:|
| Immutable baseline | 42.658 | 1778.1 | 18 / 0 (old implementation) | 4.85 |
| Instrumented, mask 0 | 42.490 | 1778.8 | 18 / 0 | 4.31 |
| Priority 1, mask 1 | 46.887 | 1818.5 | 18 / 0 | 5.49 |
| Priority 2, mask 2 | 47.974 | 1696.8 | 18 / 0 | 5.90 |
| Priority 3, mask 4 | 43.083 | 1777.5 | 3 / 15 | 4.30 |
| Priorities 2+3, mask 6 | 45.019 | 1696.2 | 3 / 15 | 5.62 |

All cases used two secant fallbacks. Priority 2 reduced fallback construction
from 425 ms (mask 0, eager) to 49 ms (mask 2, only two failures), and lowered
private peak by about 82 MiB. Priority 3 eliminated 15/18 workspace allocations;
three allocations remain because storage is deliberately released at each
failed tangent direction. Combining 2+3 preserves this memory reduction.

Priority 1 reduced material preparation from 464 to 406 ms but AD-row time
increased from 8071 to 10040 ms; its retained compact arrays cost about 40 MiB
at the measured peak. This is **not** a demonstrated overall improvement.
Other phases also slowed during that run, so separate implementation costs
from scheduling/load variation before deciding whether to retain it.

Non-solver activity was significant and variable: sampled processes included
Docker, another Python process and Task Manager. None was stopped. These are
diagnostic runs: **no wall-time speedup or adoption is established**. The next
bounded screen at that checkpoint was 12 steps for immutable baseline, masks
1, 6 and 7, in that order. Its results and the subsequent retest follow below.

## September 26 sweep and October 8–9 idle retest

The first 12-step screen (`../bench_results/workspace_sweep_20260926/`)
measured baseline / mask 1 / mask 6 / mask 7 at 563.739 / 660.341 / 649.574 /
639.169 seconds. All cases converged in 353 outer iterations, with identical
Az and mu TIFF hashes. Estimated other busy cores increased from 2.82 to
3.86 / 4.19 / 4.87, so this was diagnostic, not an adoption comparison.

The user then made the Windows machine idle. Evidence is preserved in
`../bench_results/workspace_idle_retest_20261008_01/`: original and resume
manifests, archived runner sources, per-run measurements/logs, and final
`summary.json`. No unrelated processes were stopped and the Balanced power
plan was unchanged. Conditions: IEEJ-D, 12 steps, 12 threads, chunk 1,
tolerance 1e-3, maximum 300 outer iterations, tangent-AMG and energy merit.
Wall time includes process initialization and output completion.

The compact final aggregate is also versioned in
[`results/workspace_idle_retest_20261008_summary.json`](results/workspace_idle_retest_20261008_summary.json).

| Case | Cycle 1 seconds | Cycle 2 seconds | Cycle 3 seconds | Median seconds | Median peak private MiB |
|---|---:|---:|---:|---:|---:|
| Original tangent (`6e6f644`) | 532.838 | 551.928 | 557.293 | 551.928 | 1844.188 |
| Instrumented, mask 0 | 549.683 | 563.602 | 565.751 | 563.602 | 1845.957 |
| Priorities 2+3, mask 6 | 543.188 | 536.334 | 497.645 | 536.334 | 1761.988 |

Orders were original/off/opt23, off/opt23/original, opt23/original/off.
The user interrupted the first cycle-3 opt23 attempt after 121.440 seconds
and 3 completed steps. That attempt remains unchanged and is excluded.
Cycle 3 uses `3_opt23_retry1`, followed by the remaining two runs. Resumption
could introduce thermal/frequency effects; these were not measured.

All nine accepted runs converged: 108/108 steps, 353 outer iterations per run.
Per-iteration residuals, inner counts, line-search trials/outcomes and step
status match at log precision. All 12 Az and 12 mu TIFFs of each run have
identical SHA256 hashes to their corresponding first-baseline files.
This short-sweep equivalence is not a full-revolution EMF quality pass.

Mask 6 reduces median wall time by **4.84% versus mask 0**, or **2.83% versus
the original tangent**, and median private peak by **83.97 MiB versus mask 0**.
Paired reductions versus mask 0 are 1.18%, 4.84%, and 12.04%; the last includes
the interruption/resumption. Do not advertise a stable 12% improvement.
Estimated other busy cores ranged from 0.57 to 1.06. The instrumented OFF
median is itself 2.12% slower than original; its cause is not isolated.

Work counters support the intended mechanism: every OFF run allocates FGMRES
storage 341 times; mask 6 allocates 27 times and reuses it 314 times. All runs
use 16 secant fallbacks. Fallback construction drops from 8.04–8.22 to
0.328–0.344 seconds; workspace allocation/reset drops from 8.58–8.73 to
1.046–1.084 seconds. AMG numerical hierarchies are still rebuilt each iteration.

Executable SHA256 identities (the binaries were preserved, not rebuilt):

- Original: `c83fd924d7e8cb7eef114f6f1b4740037f73388c4ab8d8afe72ea5349b4b03ec`
- Trial, core checkpoint `ca1027a`: `034c389d315326835aefc2f578fd656043177e5ba906964bf1fcd7221b9ce6a0`

Decision: retain priorities 2+3 as the research candidate; keep priority 1
off. Next compare masks **6 and 14**, isolating priority 4 (accepted B/H/mu
reuse), with the same trial executable and inputs in three paired cycles.
Priority 4 already exists in commit `f5dbb90`; this is its large-case evaluation,
not a change to material laws or acceptance criteria. Require actual field
reuse, identical numerical histories/fields, and measure allocation/memory
and total time. Stop for review on failed convergence or equivalence.

Still pending: priority 4 large-case results, same-session secant controls,
4-thread/35H300 checks, full periodic waveform and long-run memory gates.
Neither the >=15% secant speed gate nor the full adoption protocol is passed.
No defaults, public YAML, release settings, tags or main-branch changes.

## Priority 4 rejected; selected-candidate comparison (2026-10-09)

User decision: **reject priority 4**. It is not part of the selected candidate
or credited optimization progress. Keep its historical evidence to avoid
repeating the hypothesis, without enabling it in further experiments.

The completed mask 6/14 experiment is in
`../bench_results/workspace_field_reuse_20261009_01/`. All 72 steps converged;
353 outer iterations per run, identical logged numerical histories, and all
144 Az/mu TIFF files matched corresponding reference hashes. Mask 14 reused
fields 341 times per run; profiled field updates fell from 1476 to 1147.
Wall-time medians were 494.335 / 487.283 seconds (1.43% reduction), but paired
reductions were -4.54%, +1.41%, +1.43%. This does not satisfy reproducible
three-pair improvement; limited benefit does not justify adopting the added
state management. No removal/rebuild of the preserved experimental binary is
needed: its switch remains OFF, as do priority 1 and public opt-in defaults.

The next experiment, `run_selection_sweep.py`, compares **optimized secant, mask 0** with
**tangent-AMG + energy merit + priorities 2+3, mask 6**, using the same preserved
binary and physical/output settings. The secant control uses its established
default residual merit; only candidate Jacobian/merit/optimization selection
differs. Three pairs use AB/BA/AB order, 12 steps, 12 threads, chunk 1,
tolerance 1e-3 and max 300. Convergence/repeat histories are checked within
each method; different algorithms are not required to have identical histories.

Waveform selection requires all 72 steps to converge, >=15% median wall-time
reduction, shortening in each pair, candidate peak RSS/private <=2x its paired
secant control and <=8 GiB. Failure stops after timing and is recorded, not
used to justify a full-waveform claim. Passing selection launches one 248-step
secant reference and one candidate with unchanged outputs and 12-pixel slides.
The 2976-pixel angular dimension is verified to equal 248*12. Both full sweeps
must converge before the existing periodic flux/EMF/harmonic comparator runs.
Priority 4 is excluded throughout. No automatic release or default changes.

Four-thread, 35H300, cross-platform and long-run retention review remain separate
requirements; a timing/waveform pass alone is not final release acceptance.

## Selected candidate: timing and waveform gates passed (reviewed 2026-10-10)

Evidence: `../bench_results/selected_secant_20261009_01/`. The six timing runs
and two full-revolution runs completed without guard termination; all **568/568
steps strictly converged**. The reviewed harness hashes match the manifest.
Priority 4 was disabled throughout (zero field reuses); priority 1 was also OFF.
The preserved binary is still core checkpoint `ca1027a`, SHA256
`034c389d315326835aefc2f578fd656043177e5ba906964bf1fcd7221b9ce6a0`.

| Timing comparison | Optimized secant, seconds | Candidate, seconds | Reduction |
|---|---:|---:|---:|
| 12 steps, pair 1 | 627.840 | 491.908 | 21.65% |
| 12 steps, pair 2 | 639.984 | 501.579 | 21.63% |
| 12 steps, pair 3 | 635.226 | 494.767 | 22.11% |
| 12 steps, median | 635.226 | 494.767 | **22.11%** |
| 248 steps, one pair | 13400.433 | 11086.565 | **17.27%** |

Each short secant run took 1263 outer iterations versus 353 for the candidate;
within-method logged histories repeated exactly. Full sweeps took 26863 versus
7531 outer iterations, with maximum final relative residuals 9.9999517721e-4
and 9.948413339e-4 respectively. Full-sweep speed is one pair, not a repeated
median; its saving is 2313.868 seconds (38.56 minutes). Estimated non-solver
busy cores were 0.72–0.90 in the short runs and 0.723/0.751 in the full runs.

Full-sweep peak private memory was 0.981 GiB (secant) versus 1.724 GiB
(candidate); peak RSS was 0.985 versus 1.727 GiB. Both the paired 2x and 8-GiB
limits passed. Candidate sampled-private maxima by elapsed-time quarter were
1.72250, 1.72241, 1.72185 and 1.72075 GiB: no sustained growth is apparent in
this sweep. This is not proof against all leaks or an OS pool-memory audit.

All three flux channels passed the periodic waveform comparator, without
alignment/fitting, using fundamental bin 2 and identical centered differences
including wraparound. Worst differences across channels:

| Metric | Measured maximum | Acceptance limit |
|---|---:|---:|
| Flux normalized RMS difference | 0.007109% | 0.1% |
| EMF normalized RMS difference | 0.060288% | 0.5% |
| Harmonic amplitude difference / fundamental (flux or EMF) | 0.014119% | 0.1% |

These compare against a converged optimized-secant reference, not JMAG or
physical measurements; absolute accuracy is not established by this test.

Versioned evidence:

- [Three-pair timing summary](results/selected_secant_20261009_summary.json)
- [Full-waveform metrics](results/selected_secant_20261009_waveform_quality.json)
- [Full-run measurements and input provenance](results/selected_secant_20261009_provenance.json)

Decision: **retain tangent-AMG + energy + priorities 2+3 as the adoption
candidate**. The primary Windows 12-thread timing and full-waveform gates
passed. Priority 4 remains rejected and is not credited toward these gains.
Remaining: 4-thread/35H300 regression, unverified platform checks (particularly
macOS), and release integration/default-setting review. Existing Windows/Linux
numerical tests are historical evidence, not newly executed platform builds.
The benchmark harness/comparator unit suite was rerun successfully (11 tests).
No main merge, public-default change, tag or release is implied by this result.
