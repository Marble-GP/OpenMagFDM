# OpenMagFDM v1.6.1 configuration reference

This document is the normative reference for analysis YAML in v1.6.1. The
WebUI autocomplete file `webui/public/yaml-schema.json` must describe the same
contract. Historical research knobs may remain parseable for reproducibility,
but only the keys documented here are supported for production use.

## Compatibility and units

- Existing integer `slide_region_start/end` values retain pixel-index meaning.
- Decimal or exponent-form slide bounds (`0.05`, `5e-2`) mean physical metres.
- `polar_domain.r_start/r_end`, Cartesian `mesh.dx/dy`, and physical point
  coordinates are metres.
- Pixel ranges use zero-based image coordinates. Band end values follow the
  existing sliding implementation; verify the resolved range printed at start.
- User variables in `variables:` may be referenced as `$name` throughout the
  configuration. Runtime tokens such as `$step` are evaluated at solve time.

## Top-level structure

```yaml
coordinate_system: cartesian  # cartesian | polar
variables: {}
omp: { threads: 0 }
mesh: {}
polar_domain: {}
boundary_conditions: {}
polar_boundary_conditions: {}
material_presets: {}
materials: {}
flux_linkage: []
nonlinear_solver: {}
domain_decomposition: {}
transient: {}
export: {}
```

## Coordinates and boundaries

Cartesian analyses require `mesh.dx`, `mesh.dy`, and four boundaries. Polar
analyses require `polar_domain` and four polar boundaries.

```yaml
polar_domain:
  r_start: 0.0            # 0 <= r_start < r_end
  r_end: 0.10
  r_orientation: horizontal
  theta_range: 2*pi
polar_boundary_conditions:
  inner: { type: dirichlet, value: 0.0 }
  outer: { type: dirichlet, value: 0.0 }
  theta_min: { type: periodic, value: 1.0 }
  theta_max: { type: periodic, value: 1.0 }
```

`r_start: 0` includes the axis. It is supported only with a Dirichlet inner
boundary; Neumann and Robin conditions are singular at the axis and fail fast.
For a periodic pair use `value: 1.0`; use `value: -1.0` for anti-periodicity.

## Materials, presets and magnetisation

Every image colour used by the model should map to one `materials` entry.

```yaml
material_presets:
  NdFeB_N40:
    mu_r: 1.05
    magnetization: { Br: 1.26, pattern: parallel, angle: 0 }
materials:
  air: { rgb: [255, 255, 255], mu_r: 1.0, jz: 0.0 }
  magnet:
    rgb: [255, 200, 0]
    preset: NdFeB_N40
    calc_force: true
```

Material values in the analysis override preset values. Nonlinear iron may be
defined with a B-H table/formula or the supported `mu_r` forms. The WebUI
material-library manager merges the selected library before launching the
solver. Magnetisation patterns include `parallel`, `parallel_array`,
`halbach_continuous`, `polar_anisotropy`, and `custom`; consult autocomplete
for the pattern-specific parameters.

## Nonlinear solver

Recommended v1.6.1 settings:

```yaml
nonlinear_solver:
  enabled: true
  solver_type: newton-krylov
  max_iterations: 100
  tolerance: 1.0e-3
  eisenstat_walker:
    enabled: true
    gamma: 0.9
    alpha: 2.0
    eta_min: 1.0e-6
    eta_max: 0.1
  anderson:
    enabled: false
    depth: 5
    beta: 0.3
  verbose: false
  export_convergence: true
```

The stiff reference B-H model has a residual plateau around `5e-3`; requesting
`1e-5` commonly exhausts the iteration limit without improving engineering
accuracy. OpenMagFDM nevertheless treats only the authored `tolerance` as
convergence: reaching `max_iterations` retains diagnostic fields but exits with
code `2`, so an unconverged sweep cannot silently appear successful.

Eisenstat-Walker reduces the work of early inner AMGCL solves; it does not relax
the outer `tolerance`. With `eta_max: 0.1`, difficult B-H curves can need more
than 60 outer iterations. Increase `max_iterations`, choose a tolerance justified
by the model, or disable Eisenstat-Walker for a slower comparison run. Do not
compare fields from two runs unless both report convergence. The assembled
tangent Jacobian remains an opt-in research path; it did not reduce wall time
on the IEEJ-D reference case.

The stable baseline is Newton-Krylov with `anderson.enabled: false`. Anderson
acceleration is an experimental opt-in: each candidate is rebuilt with its own
field and permeability, then accepted only when the true nonlinear residual
decreases without increasing magnetic energy. The solver retries with a smaller
beta, rolls back and restarts the history on rejection, and disables Anderson
for the current solve after three consecutive rejections. If the iteration
limit is reached, OpenMagFDM exports the best evaluated state rather than the
last unchecked update.

Transient warm starts use only states that pass a bounded residual-quality
gate; a divergent step cannot seed the next one. When flux linkage is enabled,
`FluxLinkage/flux_linkage_status.csv` records `strict_converged`,
`state_reusable`, iteration count and residual for each row while the established
`flux_linkage.csv` schema remains unchanged.

## Transient motion

The preferred syntax is `slides:`. The legacy single-slide keys remain
backward compatible.

```yaml
transient:
  enabled: true
  enable_sliding: true
  total_steps: 100
  parallel_chunks: 3
  slides:
    - name: rotor
      kind: band
      direction: vertical
      region_start: 0.05       # decimal -> metres
      region_end: 0.09
      angle_deg: 0.25          # polar rotation; overrides pixels_per_step
      pixels_per_step: 1
      wrap_mode: auto
```

- `direction: vertical` bounds columns and moves content vertically;
  `horizontal` bounds rows and moves content horizontally.
- Metre bounds are legal only when the bounded axis is Cartesian x/y or polar
  r. A polar theta-axis band must use integer pixels.
- `angle_rad` takes precedence over `angle_deg`; either takes precedence over
  `pixels_per_step` and is polar-only.
- Cartesian supports multiple independent band/rectangle slides. Polar
  currently applies only `slides[0]` and prints a warning for additional ones.
- `wrap_mode`: `auto`, `periodic`, `antiperiodic`, or `vacuum`.
- `parallel_chunks: 2..4` improves long-sweep throughput but creates one cold
  first step per chunk and multiplies memory use by the chunk count.

## Windows CPU use

The packaged Windows solver uses the current MSVC OpenMP runtime. Sparse AMG
matrix-vector products are usually limited by memory bandwidth, so all logical
processors need not remain at 100%. In automatic mode OpenMagFDM uses physical
cores on Windows; `omp.threads` or `OMP_NUM_THREADS` can override this. On the
12-core/24-thread Ryzen reference machine, 12 threads completed the 1.34M-DOF
nonlinear step about 5% faster than 24 threads. For long independent sweeps,
`parallel_chunks: 2..4` is normally a larger throughput improvement than forcing
SMT, at the cost of multiplying solver memory use.

Rectangle motion is Cartesian-only:

```yaml
slides:
  - name: mover
    kind: rectangle
    rect: [100, 50, 200, 150]
    dx: 0.5
    dy: "$speed*sin(2*pi*$step/$N_step)"
    vacuum_rgb: [255, 255, 255]
```

## Domain decomposition

DD is an optional polar accuracy mode, not a general speed switch.

```yaml
domain_decomposition:
  enabled: true
  bands:
    - [0, 52, 4, 4]
    - [52, 330, 1, 1]
    - [330, 450, 4, 4]
  robin_p: 12.0
  overlap: 4
  max_outer: 8
  max_inner: 3
  tol: 1.0e-3
  relax: 1.0
```

Bands are `[c0, c1, cf_r, cf_theta]`. They must tile the full radial range.
Keep air gaps, magnets, coils and saturated teeth at `1,1`, and place band
interfaces inside iron. Experimental theta patching/coarse-space/gap-link
knobs remain off by default and are not supported through the WebUI.

## Force, flux and output

`calc_force: true` selects materials for the default Distributed Amperian
force calculation in both static and transient analyses. Legacy Maxwell-stress
implementations remain research APIs but are not the default export path.

`flux_linkage` supports a point/path form and an area-averaged material form in
both Cartesian and polar coordinates. In polar coordinates the average uses the
physical-area Jacobian (`r * dr * dtheta`). A material entry may define both
sides or only the coil cross-section present in an antiperiodic half-period
domain:

```yaml
flux_linkage:
  - { name: phase_U, material_a: coil_U_pos, material_b: coil_U_neg }
  - { name: phase_V_half, material_a: coil_V_pos } # +mean(Az) over this section
  - { name: phase_W_half, material_b: coil_W_neg } # -mean(Az) over this section
export:
  format: tiff
  precision: double
  async: true
  async_queue_depth: 2
  tiff: { compression: deflate, predictor: 3 }
```

TIFF/float64/async is the default. `format: csv` is retained for compatibility;
`format: both` writes both. `transient.export_fields` may restrict output to
`Az`, `Mu`, `H`, `Jz`, `InputImg`, `BoundaryImg`, `Forces`, and
`EnergyDensity`. `async_queue_depth` must be in `1..16`; a larger queue can
overlap more disk I/O but retains more full-field snapshots, so the default is
recommended unless storage profiling shows that writes are the bottleneck.

## Removed/deprecated configuration

The old material-level `coarsen`, `coarsen_ratio`, top-level `coarsening`, and
custom Galerkin/JFNK tuning keys are ignored with warnings. Remove them from
new configurations. AMGCL native multigrid is the production linear solver;
`domain_decomposition` is the supported variable-resolution accuracy mode.
