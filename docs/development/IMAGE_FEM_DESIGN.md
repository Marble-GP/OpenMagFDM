# Image-based 2D FEM solver — design specification

Date: 2026-11-17 (Asia/Tokyo)
Status: draft, not implemented. Baseline: v1.7.5 (`226e1b0`).
Scope owner: solver. Related: [force evaluation plan](FORCE_EVALUATION_PLAN.md),
[Amperian force verification](AMPERIAN_FORCE_VERIFICATION.md).

This document specifies an optional finite-element engine that solves the same
2D magnetostatic problem from the same colour-coded model image as the existing
finite-difference (FDM) engine. It defines the mesh pipeline (region extraction,
Douglas-Peucker boundary sampling, attraction/repulsion interior sampling,
constrained Delaunay triangulation and quality improvement), the Galerkin
discretisation, the residual-driven local refinement loop, the YAML/CLI/WebUI
integration and the verification gates.

Everything here is a proposal. No part of it is a claim of achieved accuracy or
performance. Numbers in the acceptance gates are targets to be validated, not
measured results.

## 0. Terms

| Term | Meaning |
|---|---|
| FDM | Existing finite-difference engine (`MagneticFieldAnalyzer`, row stencils). |
| FEM | Proposed finite-element engine using triangle elements. |
| Mask | Integer raster with one material id per source-image pixel. |
| Contour | Closed polyline on pixel boundaries separating two material ids. |
| CDT | Constrained Delaunay triangulation; material boundaries are forced edges. |
| CVT | Centroidal Voronoi tessellation (Lloyd relaxation); used as a stable point-smoothing step. |
| MMS | Method of manufactured solutions; analytic solution with a derived source term. |
| ZZ | Zienkiewicz-Zhu gradient-recovery error estimator. |
| NVB | Newest-vertex bisection mesh refinement. |

## 1. Goal and non-goals

Goal: solve 2D magnetostatics on an unstructured mesh derived from the model
image so that material interfaces, thin features and graded refinement are
represented without a uniform grid, and expose it as a selectable engine with
unchanged input files and unchanged result consumers.

Non-goals for the first delivery (Phases 0-3):

- Replacing the FDM engine or changing its results.
- Polar coordinates (`coordinate_system: polar`), the pre-warped polar image and
  the `polar_domain` block. Rejected with a clear error in FEM mode.
- Transient image sliding, domain decomposition, chunked parallel steps.
- 3D, electric circuits, voltage-driven coils.
- Reinterpreting anti-aliased colour blends as graded material unless the
  sub-pixel boundary option in section 5.1 is accepted.

### Mapping to the original proposal

| Proposal item | Where it is specified | Deviation |
|---|---|---|
| Edge detection to split regions | 5.1, 5.2 | Uses the existing exact-RGB mask plus marching squares instead of Canny; avoids double edges and threshold tuning. |
| Node sampling that converges by distance | 5.4 | Poisson-disk density derived from the boundary distance transform. |
| Douglas-Peucker boundary sampling with spacing and tolerance limits | 5.3 | DP is followed by arc-length resampling, because DP alone does not bound the spacing. |
| Minimum-energy interior sampling with boundary attraction and pair repulsion | 5.4 | Same energy, finished by CVT for numerical stability, then quality-refined. |
| Weak form / Galerkin solve | 6 | P1 `A_z`, exact element tangent for nonlinear media, reused AMGCL solve. |
| Local re-iteration where the residual is high | 7.3 | Kept as a Schwarz-type accelerator only; the global residual and a final global solve remain the acceptance criterion. |

## 2. Existing pipeline facts the design must respect

Verified on v1.7.5. Line references are informative, not contracts.

| Fact | Location |
|---|---|
| Image is a CLI argument `MagFDMsolver <config.yaml> <image.png> [out]`; there is no image key in YAML. `image_path:` in YAML is documentation only. | `MagFDMsolver.cpp:593-617`, `webui/public/app.js:4352-4359` |
| Image is read BGR→RGB and flipped once (`y down -> y up`). | `MagneticFieldAnalyzer.cpp:527-535,1294-1296` |
| Material assignment is exact RGB equality per pixel; no tolerance, no connected-component labelling, no contour extraction. Unmatched pixels stay `mu=MU_0`, `jz=0`. | `MagneticFieldAnalyzer.cpp:1288-2056` |
| Optional `anti_aliasing: true` interpolates mu harmonically on mixed-colour pixels between declared material RGBs. | `MagneticFieldAnalyzer.cpp:1967-2045,2484-2592` |
| Cartesian grid spacing is `mesh.dx`, `mesh.dy` in metres; unknown keys are read directly, no C++ schema validation. | `MagneticFieldAnalyzer.cpp:537-664`, `docs/CONFIGURATION.md:38-42` |
| Boundary types are `dirichlet`, `robin` (`alpha*Az + beta*dAz/dn = gamma`), `periodic` (`value: -1` antiperiodic). | `MagneticFieldAnalyzer.cpp:665-757`, `docs/CONFIGURATION.md:43-58` |
| Assembly is a row stencil only; no element matrices and no weak-form assembly exist. | `MagneticFieldAnalyzer_stencil.cpp:7-57` |
| Linear solve: AMGCL smoothed-aggregation + CG for `n > 10000`, otherwise SparseLU, with LU fallback. | `MagneticFieldAnalyzer.cpp:6928-7070` |
| Nonlinear solve: Newton-Krylov (secant default, tangent optional) + Armijo line search + Eisenstat-Walker; B-H interpolation is monotone PCHIP. | `MagneticFieldAnalyzer_nonlinear_newton.cpp`, `MagneticFieldAnalyzer_nonlinear.cpp:311-455` |
| Transient is integer image-slide steps, no `dt` and no time integration. | `MagneticFieldAnalyzer.cpp:845-1005,12383-...` |
| Force evaluation is the Amperian volume integral for regular grids; there is no FEM-compatible force path. | `AmperianForce.h:27-126` |
| Results are written as per-field rasters (`Az/Mu/H/Jz/EnergyDensity/.../step_0001.tiff|csv`) plus `Forces/step_0001.csv`. | `MagneticFieldAnalyzer.cpp:10463-10635` |
| No feature-flag mechanism exists; engine selection needs a new key and a new branch in `solve()`. | `MagneticFieldAnalyzer.cpp:6833-6878` |
| ctest suites: `input_snapshot`, `amperian_force`, `chunk_threads`, `solver_numerics`, `nonlinear_prediction_sweep`, `polar_magnetization_curl`. | `CMakeLists.txt:102-141` |

Consequence: the FEM engine is greenfield for mesh and assembly; solver
infrastructure (config parsing, material parsing, B-H tables, AMGCL/SparseLU,
error/exit conventions, result layout) is reused.

## 3. Architecture

```
config.yaml + model.png
        |
        v
[1] material mask  (exact RGB + optional AA weight, same rules as FDM)
        |
        v
[2] contours       (marching squares on the mask, sub-pixel option)
        |
        v
[3] boundary nodes (Douglas-Peucker + arc-length resampling, topology guards)
        |
        v
[4] interior nodes (density function from boundary distance; attraction/repulsion
                    energy minimisation, relaxed by CVT)
        |
        v
[5] CDT + quality  (constrained Delaunay, Ruppert-style refinement, smoothing,
                    min-angle/coverage gates)
        |
        v
[6] assembly       (P1 stiffness per element, PM surface currents, free J)
        |
        v
[7] solve          (reuse AMGCL/SparseLU; linear or Newton with exact tangent)
        |
        v
[8] refinement     (ZZ/residual estimate -> marking -> NVB -> re-solve)
        |
        v
[9] export         (sample onto the FDM raster and mesh artifact; same folders,
                    same CSV headers, same convergence status semantics)
```

Components to add:

| Component | Responsibility |
|---|---|
| `MeshBuilder` | steps 1-5, deterministic, unit-testable without a solve |
| `FemAssembler` | element stiffness, sources, constraints, residual and Jacobian |
| `FemSolver` | linear/nonlinear iteration, refinement loop |
| `FemExport` | raster sampling, mesh artifact, force/flux hooks |

Existing reused pieces: `loadConfig`/`loadImage`/`setupMaterialProperties`
(material parsing), `evaluateMu` PCHIP, `solveLinearSystemImpl` (generalise its
input from `RowMatrix` to accept FEM CSR), `InputSnapshot`, the CLI contract and
`MagneticFieldAnalyzer::exportResults` layout.

Engine selection: a new top-level `solver` block; absent or `engine: fdm` keeps
v1.7.5 behaviour bit-for-bit. `solve()` branches before any stencil or polar
setup, and FEM mode rejects unsupported combinations before allocating.

## 4. Input specification

```yaml
solver:
  engine: fem            # fdm (default) | fem
  fem:
    mesh:
      boundary_tolerance: 0.5   # m; Douglas-Peucker tolerance, must be << dx*minimum_feature_px
      min_spacing: 0.002        # m; lower bound for boundary/interior edge length
      max_spacing: 0.010        # m; upper bound, typically dx*10
      boundary_attraction: 0.6  # 0..1, density grading strength toward material boundaries
      min_angle_deg: 25         # quality gate; Ruppert refinement target
      max_growth_ratio: 1.4     # adjacent element size ratio
      max_nodes: 200000         # hard budget; exceeding it fails, never silently coarsens
      subpixel_boundaries: false # anti-aliased mixed pixels -> contour at 0.5 membership
    refinement:
      enabled: false
      marking: dorfler          # dorfler | threshold
      theta: 0.5                # Dörfler bulk fraction
      max_rounds: 8
      target_relative_error: 1.0e-3
      local_solve: false        # Phase 3b overlapping Schwarz acceleration
```

The FEM engine reuses the existing `nonlinear_solver` block verbatim
(`tolerance`, `max_iterations`, `line_search_*`, `jacobian`); no duplicate
nonlinear keys are introduced.

Rules:

- Defaults are derived from `mesh.dx` when the block omits them
  (`min_spacing = dx`, `max_spacing = 10*dx`, `boundary_tolerance = 0.5*dx`).
- Physical units are metres at every mesh key; the source image contributes
  topology and per-pixel material ids only. Pixel (i,j) covers
  `[i*dx,(i+1)*dx] x [j*dy,(j+1)*dy]` after the existing flip, matching the FDM
  cell-centre convention.
- Unknown keys inside `solver.fem` are rejected with an error (stricter than the
  legacy parser) so that typos cannot silently downgrade the mesh.
- `coordinate_system: polar`, `transient.enabled: true`,
  `domain_decomposition.enabled: true`, `mesh` missing, or an image whose
  dimensions differ from earlier steps are rejected with
  `FEM_UNSUPPORTED` (exit 1) and a message naming the offending key.
- `export.format: tiff|csv|both` and `export.precision` keep their meaning for
  the raster export. The mesh artifact is always written as JSON + binary arrays
  under `<out>/Mesh/`.
- The WebUI must add the block to `webui/public/yaml-schema.json` and
  `yaml-intellisense.js`; the C++ parser remains authoritative.

## 5. Mesh pipeline specification

### 5.1 Material mask

Reuse the FDM material rules so that FEM and FDM agree on what every pixel is:
exact RGB equality after the single vertical flip, default `rgb: [255,255,255]`,
unmatched pixels are material `air` (`mu=MU_0`, `jz=0`). Implement the mask in
the same pass that builds `mu_map`/`jz_map` (or a shared helper) to avoid two
divergent interpretations. The mask stores a 32-bit material index, not a colour.

Anti-aliasing: if any material declares `anti_aliasing`, compute per-material
membership weights with the existing line-segment test and harmonic-mu intent.
Two modes:

- `subpixel_boundaries: false` (default): mixed pixels take the material with the
  largest membership; the mesh boundary follows pixel edges. Deterministic
  tie-break by declaration order. Document this as an accuracy limitation.
- `subpixel_boundaries: true`: the contour is extracted from the continuous
  membership field at 0.5 using bilinear interpolation, giving boundaries inside
  mixed pixels. Only one membership field per material pair is supported; a pixel
  claiming three materials is an error.

### 5.2 Contour extraction (marching squares)

For each material id, build a binary indicator on the pixel-centre lattice and
extract closed contours with marching squares.

- Cells with exactly two crossings are unambiguous. Ambiguous saddle cells
  (four crossings) are resolved by the cell-centre average, not by a global
  rule, so a diagonal thin feature is not cut. The rule is fixed and tested.
- Orientation: material interior on the left; outer boundary counter-clockwise,
  holes clockwise. Nested contours are paired by containment (ray casting).
- Vertices sit on pixel edges at the midpoints between two pixel centres. The
  lattice point `(i,j)` is the pixel centre at `((i+0.5)*dx, (j+0.5)*dy)`; a
  contour vertex between centres `(i,j)` and `(i+1,j)` is at
  `((i+1)*dx, (j+0.5)*dy)`. This keeps the material raster aligned with the FDM
  cell-centre convention.
- Triple points (three materials meeting) are detected as vertices claimed by
  three contours and merged with a snap tolerance of `1e-6 m`. The merged vertex
  is inserted into each contour; no contour may pass through a triple point
  without a node.
- Features thinner than one pixel cannot be represented; the mask-level filter
  reports their count and area. Zero-area or self-touching contours are dropped
  with a warning that states the lost area in square metres.

### 5.3 Boundary simplification and resampling

Pipeline per closed contour:

1. Douglas-Peucker with tolerance `boundary_tolerance`. Implementation must be
   iterative (no deep recursion on long contours).
2. Arc-length resampling with target spacing `h_target = lerp(min_spacing,
   max_spacing, local_density)` where `local_density` is 0 near a narrow feature
   or a triple point. Resampling inserts *and* removes points, which DP alone
   cannot do; this is how the "no interval above the limit" requirement is met.
3. Validate, and fail the mesh (not the run) on violation:
   - every segment stays inside the parent material mask (rasterise segment
     midpoints and both quarter points);
   - Hausdorff distance to the unsimplified contour `<= boundary_tolerance`;
   - no self-intersection within one contour and no crossing between contours of
     different materials (segment intersection test, O(n log n) sweep);
   - no material coverage loss greater than 0.1 % of the domain area.
4. Snap near-coincident vertices (gap smaller than `0.25*min_spacing`) so that
   no zero-length edges or sliver gaps reach the triangulator; snap distance is
   recorded in the mesh artifact.

DP alone must never be the final sampler: it guarantees the tolerance but not
the spacing floor or the maximum edge length.

### 5.4 Interior sampling

Initial interior points: Poisson-disk sampling (Bridson) with a spatially varying
radius derived from the boundary distance transform of the mask. This is the
discrete form of the proposal's "attraction to the boundary": the target point
density is

```
rho(x) = 1 / h(x)^2
h(x) = max_spacing - attraction * (max_spacing - min_spacing) * (1 - g(d(x)))
```

where `d(x)` is the distance to the nearest material boundary and `g` rises
monotonically from 0 at the boundary to 1 beyond a grading band (default band:
`max_spacing`). At `boundary_attraction = 0` the mesh is uniform at
`max_spacing`; at `1.0` elements reach `min_spacing` at the interfaces and grade
outward to `max_spacing`. This is the "grading" the proposal asks for, made
explicit and testable.

Relaxation: the attraction/repulsion energy

```
E = sum_boundary w_b |x - proj_boundary(x)|^2 + sum_pairs w_r / |x_i - x_j|
```

is minimised with damped explicit steps (fixed step cap `0.25*h`, no oscillation
by accepting a step only if `E` decreases) for a bounded iteration count, then
finished by 2-4 Lloyd/CVT iterations with density `rho(x)`. The CVT step is the
convex, well-studied limit of the same idea and prevents the point set from
locking into a poor local minimum. The final positions are what reaches the
triangulator; the energy value is exported for QA but is not a convergence
criterion for the field solve.

### 5.5 Triangulation and quality

- Constrained Delaunay triangulation with every simplified contour as a forced
  edge (CDT, not plain Delaunay, which would delete thin features).
- Ruppert-style refinement: insert the circumcentre of any triangle whose
  circumradius/edge ratio or minimum angle violates `min_angle_deg`, unless it
  falls outside the domain; then enforce the material-boundary constraints.
  Termination is guaranteed for the angle bound and is additionally capped by
  `max_nodes`; hitting the cap is an error with the violating element count.
- Element material id comes from the material of the triangle centroid in the
  mask; every triangle must lie entirely inside one mask material, otherwise the
  mesh is rejected ("non-conforming material element"). This prevents a
  boundary from cutting through an element.
- Laplacian/CVT smoothing afterwards may move only interior vertices whose
  one-ring lies in a single material and that are not on a contour or a triple
  point. Boundary vertices slide along their contour only.
- Quality gates before assembly: min angle `>= min_angle_deg` (default 25°), max
  angle `<= 130°`, no inverted/zero-area triangle, size ratio across an edge
  `<= max_growth_ratio`, every boundary edge shared by exactly two elements (one
  for the outer boundary), every contour present as a mesh edge.

Triangulator choice (decision, section 13): a compact permissively licensed CDT
implementation or a vendored one. The license must be recorded in
`third_party/` and the release notes. GPL-only options are not acceptable for
the current distribution model; Triangle (Shewchuk) is only acceptable if its
non-commercial terms fit the project, which must be decided before Phase 0 ends.

### 5.6 Mesh artifact and determinism

Write `<out>/Mesh/mesh.json` (vertices, triangles, material ids, edge flags,
contour map, triple points, quality summary) and binary `vertices.bin`,
`triangles.bin` for fast reload. The artifact hash goes into
`run-inputs.json`. Given the same YAML and image, the mesh must be byte-identical:
fixed iteration counts, stable contour ordering (sorted by bounding box then
area), no dependence on `unordered_map` iteration order, no random seeds unless
recorded.

## 6. Discretisation and solver

### 6.1 Governing equation and weak form

2D magnetostatics with out-of-plane current density `J_z` and in-plane
magnetisation `M`:

```
B = curl(A_z zhat) = ( dA/dy, -dA/dx )
curl( nu curl(A) ) = J_free + curl(M),    nu(|B|) = H/B
```

Test function `v` in `H^1_0` (Dirichlet on the outer boundary):

```
integral_Omega nu grad(A) . grad(v) dA
  = integral_Omega J_z v dA
  + integral_Omega M . (grad(v) x zhat) dA + boundary terms for prescribed M
```

`nu` is element-wise constant evaluated at the element centroid. Because the
finite-element unknown is `A` and `B` is known per element, the constitutive
relation is used in the inverse direction: the loaded B-H table is monotone
(checked at load), so `H(B)` and hence `nu(B) = H(B)/B` and `dnu/dB` come from
the inverse PCHIP of the same table as the FDM engine. Remanence enters as
equivalent surface current density `K = M x n` on the boundaries of each
magnetised region (line integral on the contour), matching the existing
`curl(M)` convention in `AmperianForce.h`. The sign convention is fixed in code
by one explicit formula and validated against the circular-PM analytic case
(section 10.3) before any other verification is trusted.

### 6.2 Elements and assembly

- P1 (linear) triangles, one unknown per vertex, `A_z` nodal.
- `B` is constant per element: `B_T = sum_i A_i (dphi_i/dy, -dphi_i/dx)`.
- Element stiffness `K_e[i][j] = nu_T * area_T * grad(phi_i) . grad(phi_j)`.
- Sources: `J_z` area integral over triangles whose mask material has a
  non-zero `jz`; PM surface currents as contour line integrals.
- Assembly into a CSR matrix `Eigen::SparseMatrix<double, RowMajor>` so that
  `solveLinearSystemImpl` can be reused for the linear solve (AMGCL above
  10 000 unknowns, SparseLU below, same tolerances and fallbacks).
- Dirichlet: eliminate the constrained rows with a symmetric penalty-free
  elimination (identity row + zero column, adjust RHS) to keep the matrix
  symmetric for CG.
- Robin and periodic boundary conditions: reuse the FDM semantics. Robin is a
  natural boundary term assembled on boundary edges; periodic/antiperiodic is a
  constraint mapping of the two boundary vertex sets. Both are Phase 2; Phase 1
  implements Dirichlet only and rejects other types with `FEM_UNSUPPORTED`.

### 6.3 Nonlinear iteration

Reuse the `nonlinear_solver` block (`tolerance`, `max_iterations`,
`line_search_*`, `jacobian`) without duplicating keys.

- Newton with the exact element tangent: for element `T`,
  `d(nu_T K_T)/dA_j` includes `dnu/dB * (dB_T/dA_j)` with `dnu/dB` from the
  PCHIP derivative. Element 3x3 blocks are assembled into the Jacobian.
- The existing `secant` option is not meaningful for FEM; `jacobian: secant` in
  FEM mode produces a warning and falls back to the element tangent.
- Armijo line search on the residual norm with the existing defaults; on line
  search failure, fall back to Picard with `relaxation` for that Newton step.
- Convergence: `||R|| / ||f|| < tolerance` (same default 5e-4) plus a final
  re-evaluation that sets the same `last_nonlinear_converged_` flag the FDM
  engine uses. Non-convergence exits 2, as today.
- Initial guess: zero on the first static solve, previous solution interpolated
  onto the new mesh after refinement (barycentric projection).

### 6.4 Error and exit semantics

| Condition | Behaviour |
|---|---|
| Unsupported combination (polar, transient, DD, missing image) | exit 1, `FEM_UNSUPPORTED`, message names the key |
| Mesh quality gate failure | exit 1, `FEM_MESH_FAILED`, artifact with the violating elements is written for diagnosis |
| `max_nodes` exceeded | exit 1, `FEM_MESH_TOO_LARGE` |
| Linear solve failure | reuse existing fallback chain; exit 1 if all fail |
| Nonlinear non-convergence | exit 2, best evaluated state exported, as today |
| Refinement did not reach target but rounds exhausted | exit 0 with a warning and `refinement.status: "rounds_exhausted"` |

## 7. Adaptive refinement

### 7.1 Indicators

Two indicators, both computed per element:

- ZZ recovery: build nodal gradients by area-weighted averaging of element
  gradients, then `eta_K = || grad(A) - G(grad(A)) ||_K`. Simple, no edge data,
  suitable as the default.
- Residual: `eta_K^2 = h_K^2 || f_K ||^2 + sum_edges h_e || [nu dA/dn] ||^2`
  for elements touching a material boundary or a source region. This is the more
  physical indicator; the jump term is assembled from the same edge data the
  refinement needs anyway.

Global estimate `eta = sqrt(sum eta_K^2)`; relative error uses the energy norm
`||A||_E = sqrt(A^T K A)` before the nonlinear update. Estimator correctness is
validated by the MMS convergence test (section 10.2): the observed convergence
order must match the theoretical one, otherwise the indicator is not trusted.

### 7.2 Marking and refinement

- Dörfler marking with `theta` (default 0.5): mark the smallest element set whose
  `eta_K^2` covers `theta` of the total; Phase 1 may use simple threshold
  marking if Dörfler needs more implementation time.
- NVB (newest-vertex bisection) with closure, so the mesh stays conforming.
  Material interfaces are refined by splitting the interface edge, which keeps
  both parent triangles in their material; the children inherit the material id.
  Refinement must never create a triangle crossing a contour.
- Loop: solve -> estimate -> mark -> refine -> warm start -> solve, until
  `eta_rel <= target_relative_error` or `max_rounds`. Every round is recorded in
  `<out>/Mesh/refinement.json` (nodes, elements, eta, rounds, residuals).

### 7.3 Local iteration (Phase 3b)

"Extract only the high-residual region and iterate locally" is admissible only
as an accelerator, never as the final answer, because elliptic coupling makes a
purely local update inconsistent with the global residual.

- Build overlapping subdomains as the element neighbourhood (one or two layers)
  of marked elements.
- Solve the subdomain problem with Dirichlet data from the current global
  solution (multiplicative Schwarz), or with the global matrix restricted
  (additive Schwarz).
- Track the *global* residual norm after each local sweep; stop when the
  reduction per work unit no longer beats a global AMG solve.
- Always perform one global solve before export and before the estimator is
  recomputed. The exported state must satisfy the global tolerance.

The proposal's expected benefit (cheap iterations on hot spots) is plausible;
the spec deliberately makes the acceptance criterion "global residual per unit
work", not "local field looks smoother".

## 8. Output specification

Downstream consumers (WebUI plots, flux linkage, force CSVs, result history)
read rasters and CSV headers. FEM results are therefore **sampled onto the FDM
raster** rather than changing the consumer contract:

- The export grid is the image pixel lattice at `dx`,`dy` (the same
  `nx x ny` used by the FDM engine). Vertex values are barycentrically
  interpolated; element-wise `Mu`, `H`, `Jz`, `EnergyDensity` are evaluated at
  cell centres inside their element. Points outside the mesh are written as
  missing values (not zero) and counted in the status file.
- `Az/step_0001.tiff|csv` keeps the FDM orientation (already flipped once) and
  the same value convention (Wb/m).
- `FluxLinkage/flux_linkage.csv` and `flux_linkage_status.csv` keep their
  headers; contour flux is integrated from the sampled `A` with the same
  quadrature the FDM path uses, and the status records `engine: fem`.
- `Forces/step_0001.csv`: Phase 1 writes the same header with
  `status: unavailable` and zeros, never fabricated numbers, until the Phase 4
  force implementation is validated. Existing readers already preserve
  unavailable values (v1.7.4 contract).
- `conditions.json` gains `solver.engine`, mesh statistics and the refinement
  summary. `run-inputs.json` records the mesh hash and all FEM parameters.
- The mesh artifact under `<out>/Mesh/` is the source of truth for QA and for
  re-running a solve without re-meshing (`--mesh <path>`, Phase 1 optional).

## 9. Integration

- CLI: unchanged invocation `MagFDMsolver config.yaml image.png out/`.
  `--version` unchanged. Add `--engine fem|fdm` as an override that takes
  precedence over YAML and prints the effective engine in `log.txt`.
- WebUI: the run form gets an engine selector only after Phase 3; job submission
  stays the three-argument spawn. Add `solver` to `yaml-schema.json` and the
  intellisense docs. The results view needs no change if section 8 holds.
- MCP: `analysis_start` keeps its schema; the returned status includes
  `engine`, `mesh.nodes`, `refinement.rounds`. No new tool.
- Failure surfaces: the WebUI must show `FEM_UNSUPPORTED` and
  `FEM_MESH_FAILED` messages verbatim; the panel must not retry a deterministic
  mesh failure.

## 10. Test and verification plan

### 10.1 Mesh tests (`fem_mesh_pipeline`)

Synthetic images, each with an expected assertion:

| Fixture | Assertion |
|---|---|
| Single filled rectangle | one closed contour, four segments after DP, coverage loss <= 0.1 %, min angle gate |
| Annulus | outer CCW, hole CW, 2 contours, no crossing, materials assigned per element |
| Thin gap (1-2 px) | gap preserved or explicit `FEM_MESH_FAILED`; never silently merged |
| Triple point | one shared vertex, three contours, watertight |
| Anti-aliased boundary | with `subpixel_boundaries: false` deterministic majority; with `true`, boundary moves inside the pixel and no three-material pixel is silent |
| DP tolerance sweep | spacing floor/ceiling always respected, Hausdorff bound always met, no self-intersection |
| Attraction sweep | element size decreases monotonically toward the boundary as `boundary_attraction` rises |
| Determinism | two runs produce identical mesh hashes |

### 10.2 Manufactured solutions (`fem_mms`)

Choose `A(x,y)` analytic (e.g. a smooth trigonometric function, not the linear
polynomial that P1 reproduces exactly), derive `J_z = -div(nu grad A)`, solve on
a sequence of uniformly refined meshes, and measure the error in L2 and H1.
Acceptance: observed orders `>= 1.9` (L2) and `>= 0.9` (H1) for P1, on a
homogeneous material. This test is the guard against a wrong sign, a wrong
Jacobian or a broken assembly: without it, agreement with FDM could hide a
shared convention error.

### 10.3 Analytic cases (`fem_analytic`)

- Infinite solenoid (uniform interior field, zero exterior) with Dirichlet box.
- Two parallel wires: field at a set of probe points vs Biot-Savart.
- Circular permanent magnet in air: analytic series solution for the field on
  the axis and at a ring of points; validates the `M x n` source convention.
- Rectangular bar in a uniform external field: demagnetisation factor vs the
  reference value; validates the interface treatment of a linear permeable body.
- Nonlinear ring (B-H table): compare against a fine-mesh reference solution of
  the same model, with the PCHIP table frozen in both.

Tolerances are set from mesh convergence (two successive refinements), not from
a single run: report the extrapolated value and the observed order.

### 10.4 Cross-check against FDM

Run the existing configurations (`benchmarks/solver_speed/ieej_12step.yaml` and
the internal fixtures) through both engines:

- Global relative L2 difference of `B` on the export raster: target <= 2 % on
  smooth problems, with the FEM result expected to converge to the analytic
  limit and FDM showing the larger error.
- Interface-local error: FEM must not be worse than FDM on any measured
  interface segment; report the worst segment, not only the average.
- Flux linkage and total energy: agreement within the same tolerance band as
  the field; a systematic offset indicates a source or sign error.
- Force: deferred to Phase 4; the Phase 1-3 export must mark forces unavailable.

### 10.5 Performance budget (`fem_performance`)

Record wall time, peak RSS and nodes for the IEEJ model on the reference
machine. Acceptance targets (to be refined after Phase 1):

- FEM with <= 50 000 unknowns solves in <= 1.5x the FDM runtime for the same
  model at the same `dx`.
- Export adds no more than 10 % to the solve time.
- Adaptive refinement rounds must reduce `eta_rel` by at least 30 % per round on
  the first three rounds, or the run reports `rounds_exhausted`.

### 10.6 Regression and release gates

Add ctest suites `fem_mesh_pipeline`, `fem_mms`, `fem_analytic`,
`fem_nonlinear`, `fem_adaptive`, `fem_export_compat` following the existing
fixture-directory convention. Release requires: all existing suites unchanged,
new suites green on Windows/Linux/macOS CI, WebUI tests unchanged, and the
FDM pathway byte-identical for a fixed config/image (a checksum test).

## 11. Phases and estimates

Estimates assume one experienced developer, part-time, and are deliberately
coarse.

| Phase | Content | Estimate | Exit gate |
|---|---|---|---|
| 0 | mask, marching squares, DP+resampling, CDT, quality gates, mesh artifact, fixtures | 3-4 weeks | `fem_mesh_pipeline` green, deterministic hashes |
| 1 | P1 linear assembly, Dirichlet, raster export, MMS + analytic linear cases, CLI/engine switch | 3-4 weeks | MMS orders, solenoid/wire/bar cases |
| 2 | nonlinear PCHIP Newton, Robin/periodic, IEEJ cross-check, convergence/exit semantics | 3-5 weeks | `fem_nonlinear`, FDM cross-check tolerances |
| 3 | ZZ/residual estimators, Dörfler + NVB, refinement loop, local Schwarz (3b) | 3-5 weeks | `fem_adaptive`, per-round eta reduction |
| 4 | force evaluation (Amperian contract), flux/energy parity, transient study, WebUI/MCP exposure | 4-6 weeks | `fem_export_compat`, force cases from `amperian_force_test` |

Total: roughly four to six months to a product-quality static FEM path,
including verification. A throwaway prototype of Phase 0-1 can be built in
about three weeks and is the recommended first step; it either proves the
pipeline or exposes the topology problems early.

## 12. Risks and alternatives

| Risk | Impact | Mitigation |
|---|---|---|
| Contour topology breaks on real motor images (thin bridges, triple points) | mesh failure or silently wrong material map | coverage/Hausdorff guards, fixtures, explicit failure instead of approximation |
| CDT library license incompatible | blocked release | decide license before Phase 0; fallback to a permissive compact CDT implementation |
| Point-energy sampling converges to a poor local minimum | bad element quality | CVT finish and Ruppert refinement; quality gates are hard gates |
| Anti-aliased materials have no exact FEM analogue | systematic interface error | sub-pixel boundary option; otherwise documented limitation |
| Newton tangent from a non-monotone B-H table | divergence | monotonicity check at load, Picard fallback, same exit semantics as FDM |
| Polar/transient/DD expectations | user surprise | explicit `FEM_UNSUPPORTED` errors, docs, no silent fallback |
| Local refinement gives a false sense of accuracy | wrong engineering decisions | global residual criterion, global solve before export, estimator validated by MMS |
| Maintenance of a second solver | doubled validation surface | shared material/B-H/linear-solver/export code, FDM checksum regression |

Alternatives considered:

- Adaptive FDM only (local grid refinement on the existing stencil): lower risk,
  keeps the validated kernel, but keeps staircase interfaces and cannot grade
  the mesh arbitrarily. This is the fallback if the mesh pipeline proves
  unstable on real images.
- Isogeometric or cut-cell FDM: attractive for interfaces but the cut-cell
  assembly is as much work as FEM without the element-quality tooling.
- Calling an external solver (FEMM-like): rejected for distribution, control and
  reproducibility reasons.

## 13. Open questions

1. CDT library: **resolved** — `artem-ogre/CDT` (MPL-2.0), vendored under
   `third_party/CDT/` (header-only, robust predicates). Phase 0 uses it.
2. Is `subpixel_boundaries` in scope for the first release, or is exact-match
   rasterisation acceptable with a documented limitation?
3. Should FEM mode support the `anti_aliasing` materials as graded elements
   (isoparametric interface) or require exact-match images?
4. Is a mesh cache keyed by (config, image hash, FEM parameters) acceptable, or
   must every run rebuild the mesh for auditability? (Recommendation: rebuild by
   default, cache only with an explicit flag.)
5. Which force method to validate first: element-wise Amperian integration or a
   Maxwell stress contour? The existing force plan requires benchmarking both
   before selecting a default.
6. Target hardware and the real performance envelope: the performance budgets
   in section 10.5 are placeholders until Phase 1 measures the reference model.
7. Whether transient support is a later project or a hard requirement for
   adoption: rebuilding the mesh per slide step is expensive, while remapping
   materials on a fixed mesh keeps interfaces locked to the old geometry.
