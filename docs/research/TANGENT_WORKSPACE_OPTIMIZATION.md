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
bounded screen is 12 steps for immutable baseline, masks 1, 6 and 7, in that
order. Priority 4's large-case evaluation follows selection from that screen.

Still pending: sweep timings, repeated controlled medians with secant controls,
priority 4 large-case comparison, secondary cases, periodic waveform and
long-run memory gates. No optimization has passed the adoption protocol.
