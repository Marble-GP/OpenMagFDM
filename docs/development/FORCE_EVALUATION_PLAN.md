# Single-step electromagnetic force correction and method selection

Date: 2026-11-11 (Asia/Tokyo)
Status: the primary Amperian correction is implemented on `fix/ampere-force-evaluation`.
See [verification and remaining limits](AMPERIAN_FORCE_VERIFICATION.md). The audit below
describes the v1.7.4 baseline; broader acceptance gates remain a roadmap, not a claim
that every geometry, contact traction or material model has been certified.
Baseline: v1.7.4 / `ccf2c4572d345f30bed66b4ef66690f2e88b2144`.

## Requirements and decision

Compute net force and torque from **one step's converged magnetic state, geometry, materials and prescribed currents/magnetization**. Production evaluation must not read previous/next steps, displace the model and solve again, or smooth the force history to conceal errors. A comparison suite may independently solve different grids or positions; each force evaluation still consumes only its own snapshot.

The provisional first implementation candidate is a **body-local, source-complete Amperian evaluation with consistent interface treatment**. Do not replace it by the existing Maxwell line integral merely because the continuum formula is correct. Benchmark against single-solution local virtual work and spatially averaged stress methods before selecting the production default. No candidate is currently demonstrated to have the smallest error across nonlinear motor models.

Select the validated method by analytical/reference error, ghost-force rejection, worst-case mesh/seam sensitivity, and convergence, not smooth-looking curves or agreement between two implementations sharing the same error. If performance differs by supported geometry, document that scope instead of claiming one universally best algorithm. Never pick whichever method happens to return the smallest force.

Keep this correction separate from 3D development and voltage differentiation. Preserve existing YAML and the field solve where possible. Accuracy changes in formerly defective force outputs are intentional.

## Issue history checked

[Issue #1](https://github.com/Marble-GP/OpenMagFDM/issues/1) and all three comments were read. It reports force/torque ripple on an inert mu_r=1 test piece whose displacement does not change energy. Its configured field-producing regions are separate current sources. The comments describe adopting distributed bound currents to remove ghost forces, show magnetic/nonmagnetic comparisons, and correct the energy/co-energy sign convention. Issue #2 also reports a historical static-path omission, so static and transient paths must both be tested.

The issue's reported sub-1% agreement is historical evidence, not a universal bound; numerical benchmark tables and original result arrays are not supplied in those comments. Also, **mu_r=1 alone does not imply zero force or zero magnetization**: a current-carrying conductor or prescribed permanent magnet is not an inert marker. Exact zero is required only when the selected object has no constitutive magnetic response, no remanence and no free current and does not alter the model.

The issue's intended formula includes `J_free + curl(M)`. The current implementation omits free current and prescribed permanent magnetization, so its failures do not by themselves invalidate the intended Amperian approach.

## Implementation audit

Default calls remain `calculateForceDistributedAmperian` in both `MagFDMsolver.cpp` (static) and `MagneticFieldAnalyzer.cpp` (transient). Export reads `force_results_amperian` directly.

| Finding | Confirmed consequence | Correction requirement |
|---|---|---|
| Global curl integrated over independently dilated material masks | Identical Az but summed force doubles after color splitting | Define one mechanical-body union; compute source ownership before differentiation |
| Canonical antiperiodic sign ignored in force curl | Equivalent full/half models give opposite torque signs | Shared signed indexing and correct physical component basis |
| Free current omitted | Native conductor test: -14.4 N/m reference, approximately zero output | Include selected free current once |
| Prescribed magnetization omitted | Native PM test: approximately 11.46 N*m/m reference, approximately zero output | Construct total M from the actual constitutive/source convention |
| Force radius shifted by half a cell | Approximately 5% change in one polar diagnostic | Use the solver's actual coordinates and explicit quadrature |
| Per-depth torque labeled N*m; uncomputed material energy exported as zero | Ambiguous conversion and unavailable values appear physical | Explicit units, reference origin, quality and missing-value handling |

These defects were reproduced on the official v1.7.4 Windows executable in ten small cases during the 2026-11-06 review. The external report's exact 60% error remains unverified because its model/data are unavailable; do not apply a 1/1.6 correction.

### Candidate replacement code also needs correction

- `sampleFieldsAtPhysicalPoint` and `sampleFieldsAtPolarPoint` pass the antiperiodic flag to permeability as well as B. Mu is even under the boundary mapping. An isolated compiled test of the unchanged `bilinearInterpolatePolar` helper returned **0 for constant mu=1** halfway across the seam when called that way.
- The same helper loses whole-sector wrap parity: an antiperiodic field sampled one sector beyond theta=0 returned **+1 instead of -1**. Field interpolation and material interpolation need separate parity contracts.
- The old Shell routine sets G=1 inside and 0 outside, but sums **+T grad(G)**. This convention requires **-T grad(G)**. This is a source-level sign finding, not a new native Shell benchmark.
- That routine calls mu_r near 1 air, clips shell pieces intersecting other materials, and uses Cartesian metrics. It is not a validated polar replacement and must not be activated unchanged.
- `webui/public/app.js::loadForceData` uses `parseFloat(...) || 0` and sums material rows automatically. It must preserve invalid/unavailable results and distinguish body totals from component diagnostics.

Existing numerical regression tests mainly validate magnetic operators, sources and convergence; force CSV existence is not a physical force validation.

## Candidate A: body-local Amperian evaluation

For a mechanical body b, let chi_b select its regions. Form M_b = chi_b M_total and J_free,b = chi_b J_free. For an isolated body extend M_b by zero outside it, and retain the complete support of its curl, including interface currents:

```
J_bound,b = curl(M_b)
force_per_depth = integral (J_free,b + J_bound,b) cross B_consistent dA
torque_per_depth = integral (x-origin) cross [(J_free,b + J_bound,b) cross B_consistent] dA
```

This is not the current global-curl-then-dilate algorithm. The integration may cover derivative support outside the body, but must not acquire another body's source current. Summing algebraic partitions should reproduce the union result. Publish net mechanical-body force; forces on artificial contacting material partitions require separate mechanical interpretation and are not certified by additivity alone.

### Constitutive and interface rules

1. Use total M = B/mu0 - H with H obtained from the actual constitutive law and source convention, including prescribed remanence. Do not use the magnitude `H_map` as a vector. Establish how solver `Mx_map`, `My_map` and `Jz_mag_map` represent the source before adding terms; do not count remanence twice or substitute differential permeability for secant constitutive response.
2. An inert material with exact mu_r=1, zero prescribed magnetization and zero free current must produce exact zero from the source definition. Do not zero real PMs/coils by a permeability threshold or subtract an empirical force offset.
3. In the continuum representation, bound volume current and interface sheet current both matter. If using a distributed curl of zero-extended M, do not add the sheet term a second time. If separating faces, use K_b = M_b cross n and consistently handle current-sheet field jumps and the average limiting field rather than arbitrarily choosing the inside or outside value.
4. Build a conservative face/circulation representation or a discrete summation-by-parts form. Use shared face values and pairwise cancellation so arbitrary color partitions do not create duplicated loads. A weak/discrete-by-parts form can avoid explicitly differentiating a discontinuous cell magnetization, but still needs independent accuracy tests for B reconstruction and boundary terms.
5. Apply the same quadrature to force and moment. If using a dipole-force reformulation instead of resolved bound currents, derive its intrinsic magnetic torque term; do not omit it or add it on top of a current formulation that already includes it.
6. Verify net self-force/self-torque of isolated sources, rotational/translational symmetry, nonlinear response and corner/interface convergence. Fixing missing source terms alone is insufficient.
7. Do not use arbitrary Gaussian smoothing as a default correction. It changes interface/source distribution. Any explicit regularization must preserve integrated currents, include its support, be reported, and converge with refinement.

### Limited one-snapshot experiment already completed

An internal JavaScript postprocessor used existing official-v1.7.4 Cartesian Az/J arrays and known fixture materials. It selected each body's sources before central differentiation and integrated the complete compact support. It read no other step and performed no new magnetic solve.

| Check | Experimental value | Reference / interpretation |
|---|---:|---|
| Current-carrying square in uniform external By | Fx = -14.3999999999998 N/m | -14.4 N/m analytical |
| Uniformly magnetized square in external By | Torque = 11.4591559026163 N*m/m | 11.4591559026165 N*m/m analytical |
| Permeable strip, one color | Fx = 0.0333681898663390 N/m | Partition invariance reference, not independent accuracy reference |
| Same strip, sum of two algebraic source partitions | Fx = 0.0333681898663383 N/m | No double counting; individual partition loads are not certified physical contact loads |
| Inert marker in a nonuniform field from other sources | Fx=Fy=torque=0 exactly | Required ghost-force invariant |

These are deliberately simple interior fixtures. They show that correcting the intended method is viable; they do not validate production code or prove smallest error for nonlinear moving machines. Earlier polar data also show that a seam/coordinate repair alone does not eliminate all discretization error.

## Other single-step candidates and their role

| Method | One-step use | Expected benefit | Main limit / decision |
|---|---|---|---|
| Body-local Amperian, conservative/weak interface treatment | Yes | Structural zero for inert targets; arbitrary geometry; complete source attribution | First implementation candidate; M/B derivative and interface error require comparison |
| Local virtual work / discrete energy-coordinate derivative | Yes, if derived from the converged state and a consistent variational operator | Energy-consistent load and no subtraction of neighboring solved energies | Strong challenger; do not assume current FDM residual is a gradient of exported energy |
| Weighted-volume Maxwell / eggshell | Yes | Spatial averaging avoids dependence on one jagged contour | Competitor/diagnostic; validate sign, weighting, air support, ghost force and metrics |
| Arkkio air-gap torque | Yes | Spatial averaging across a cylindrical air gap can reduce mesh sensitivity | Strong motor-torque competitor for radial-flux geometry; not a generic body-force or unrolled axial-flux formula |
| Single air-contour Maxwell integral | Yes | Simple independent comparison with analytic geometry | Diagnostic baseline, not presumed most accurate and not a privileged ground truth |
| Difference of energies across steps/perturbed solves | No | Historical diagnostic | Excluded from production force evaluation by requirement |

A local virtual-work method is not the existing timeline difference. It differentiates the energy/action with respect to a virtual geometric displacement at the same state. For illustration only, if a consistent linear discretization has action Pi(a,q)=0.5*a^T*K(q)*a-a^T*b(q), the load at a stationary solution is -partial(Pi)/partial(q), including source and boundary terms. No neighboring field solution is required. Nonlinear materials and permanent magnets require the correct energy/source functional. An integer image shift or `round()` cannot be differentiated to obtain a physical virtual displacement; continuous face/metric variations must be defined first. Do not claim this method implemented by differentiating `system_total_energy` alone.

A weighted stress method and virtual work may be related formulations, not statistically independent validations. Comparisons must include analytical problems and refinement, not agreement alone. If an auxiliary linear/weighting solve is proposed, it must use only the current state and be identified explicitly; adjacent magnetic solutions remain forbidden.

## Mechanical body, coordinates and compatibility

- Add an optional method-neutral `force_evaluation` block, with named bodies, explicit membership, reference origin and method. Preserve legacy `calc_force` syntax and its input compatibility; do not silently merge different mechanical parts. Material names selecting both rotor and stator need a region/component selector.
- A rotor is the union of its core and magnets. Do not sum a body total with its constituent rows or unrelated stator loads. Overlapping user bodies must be explicitly labeled non-additive or rejected for automatic totals.
- Use the current step's geometry after motion. Cache keys include body membership, geometry and boundary topology. Both static and transient execution call the same dispatcher.
- Preserve `r_coords`, global `theta_offset`, both polar storage orientations and physical torque origins. Antiperiodic local field components change sign; mu, masks and weights do not. Rotate global vectors using the basis at each physical angle.
- Full-machine torque about the polar origin can use validated repeated sectors. Net Cartesian force requires a rotated vector sum, not multiplication by the number of sectors. Sector cut surfaces/source support must be accounted for consistently.
- Raw outputs are per depth: N/m and N*m/m. Optional depth conversion occurs once. Unrolled shaft torque requires mean radius times tangential force times depth and validated repeat count; rectangular image-center moment is a different quantity.
- Invalid force does not become zero. Retain useful field outputs and emit force status/reason, field residual, method/version and available diagnostics. Update CLI exit/status behavior, REST/MCP and WebUI together rather than inventing an isolated numerical exit code.
- Preserve old results as legacy data; do not relabel their method or rewrite them. New outputs require a method-neutral body table and unit/status metadata. Compatibility export must not masquerade body totals as independently additive materials.

## Implementation stages

1. **Regression baseline:** turn the issue ghost-force scenario and the confirmed omissions/duplication/seam failures into physical acceptance tests. Keep the experimental datasets outside release artifacts; commit only intentionally maintained small test generators/kernels when implementing.
2. **Common numerical infrastructure:** independently test physical coordinates, signed sampling, body masks, source reconstruction and source-support accounting. Separate these kernels from the monolithic analyzer; reuse Eigen/AMGCL only where needed.
3. **Amperian correction candidate:** implement complete body-local sources and a conservative interface treatment, with the simple central form retained only as a comparison. Evaluate one state only. Reproduce all limiting cases before motor sweeps.
4. **Fair challengers:** implement an independently checked weighted/Arkkio benchmark and assess whether a discrete-energy/local-work formulation is compatible with the actual FDM operator. Do not activate old Shell code unchanged. Record unsupported cases honestly.
5. **Method selection:** compare candidates on the same fields, grids, geometry, materials and sources. Select the default only after the gates below; document applicability and unresolved errors. If no method passes, keep the force feature explicitly unvalidated rather than choosing a visually smooth output.
6. **Interfaces and integration:** route static/transient/export/UI/MCP through the selected method, add body configuration/diagnostics and preserve old YAML/result reading. Run full field regressions to show unchanged Az, B and flux for unchanged inputs.

Use a correction branch from stable main when changing production code; do not mix v2/3D work. No release is authorized by this planning document.

## Acceptance matrix and selection rule

Use absolute-plus-relative tolerances with a declared physical reference scale; relative error at zero torque is meaningless. The following are proposed initial gates, not claims already met by production code:

- **Issue #1:** inert nonmagnetic/no-source marker in strong nonuniform fields, irregular boundaries, translations and rotations. Structural zero in Amperian/source-based kernels; other methods must meet a strict normalized absolute ghost-force budget (initial target 1e-10 of an active-load reference) without case-specific offset subtraction.
- **Analytical force/torque:** coil in known external field, PM dipole torque, linear magnetic attraction and isolated-source self-load. Initial resolved-grid target 1% relative error plus appropriate absolute tolerance; include sign and origin dependence.
- **Partition/body invariance:** identical field/body with different material-color subdivisions; normalized discrepancy <=1e-10 where identical arithmetic support is available. No double-counting body/components.
- **Full/half/periodic/antiperiodic:** asymmetric placement near seams, both polar orientations, nonzero angular offset and torque origin; target <=1e-8 for mathematically matching snapshots, accounting for solver residual. Symmetric cases alone are insufficient.
- **Refinement and jagged geometry:** at least three physically consistent resolutions in both polar axes; vary gap cells, material contrast, corners, contour/weight width and grid alignment. Compare error envelopes rather than assuming monotonic error at every rasterization.
- **Nonlinear/load coverage:** unsaturated and saturated pure iron, prescribed PMs including recoil mu_r=1, zero/load current, inner/outer rotors, straightened models and multiple rotor positions. Each force evaluation sees one snapshot only. Test field-tolerance tightening separately from mesh refinement.
- **One-step independence:** evaluate a selected step in isolation with all other result files unavailable; reproduce the batch value. Changing order or neighboring step data must not change force.
- **Quality and failure:** unconverged fields, narrow gaps, invalid reconstruction, ambiguous/contacting bodies and unsupported symmetry must produce diagnostics, not fabricated values. No history smoothing or dropped outlier without a recorded reason.
- **Independent references:** analytical solutions first; matched external solver/reference cases where available. Maxwell/Arkkio consistency is useful evidence, not proof by itself. If energy-reference experiments are used in development, they are not production inputs; match fixed source conditions and PM energy conventions.

Rank admissible candidates by worst-case normalized reference error over predeclared fixtures, then grid/seam sensitivity and runtime/memory. All structural invariants must pass before averaging errors. Freeze acceptance thresholds before evaluating a holdout motor case. If the Amperian correction remains the best-supported method, retain it as default; if another single-step method demonstrably wins for a geometry class, document the evidence and scope.

## Current status and evidence

Completed: issue/comment review, active/alternative implementation audit, unchanged-helper C++ test, limited one-snapshot body-local Amperian experiment, and this selection/implementation plan.

Not completed: production correction, nonlinear motor comparison, demonstrated lowest-error method, revised UI/export, or release validation. This distinction is intentional.

Internal evidence (not repository/release data): `qa/force_review_20261106/REVIEW.md`, `qa/force_review_20261106/v174/`, and `qa/force_plan_20261111/{issue-1.json,helper-results.txt,ampere-candidate.cjs,ampere-candidate.json}`.

## Sources

- Project issue and history: https://github.com/Marble-GP/OpenMagFDM/issues/1
- Distributed-current report: https://github.com/Marble-GP/OpenMagFDM/issues/1#issuecomment-3677673149
- FEMM weighted stress formulation: https://www.femm.info/doku/doku.php?id=weightedstresstensor
- COMSOL, electromagnetic forces and Arkkio assumptions: https://doc.comsol.com/6.4/doc/com.comsol.help.acdc/acdc_ug_modeling.06.20.html
- Fu et al., local energy-coordinate derivative with permanent magnets, DOI: https://doi.org/10.1109/TMAG.2004.824774 (author-uploaded paper: https://www.researchgate.net/publication/3110671_Magnetic_Force_Computation_in_Permanent_Magnets_Using_a_Local_Energy_Coordinate_Derivative_Method)
- Henrotte and Hameyer, variational force-density formulation: https://www.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaabcfydq
- MIT 6.685 electromagnetic forces and co-energy: https://web.mit.edu/course/6/6.685/www/chapter1.pdf

Finite-element references guide the continuum/variational reasoning. They do not establish accuracy for this FDM implementation without the tests above.
