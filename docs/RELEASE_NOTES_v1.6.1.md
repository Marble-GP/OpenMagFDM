# OpenMagFDM v1.6.1 release notes

Status: release candidate on `feature/webui-template-v1.6.1`.

v1.6.1 consolidates the v1.6 solver work into a safer, self-describing user
experience. It is backward compatible with integer pixel-based slide regions.

## Solver fixes

- Restored default banded DD performance after an inactive OpenMP region had
  serialized nested AMGCL work (IEEJ-D regression: ~862 s back to ~57 s).
- Allowed `polar_domain.r_start: 0` with a mandatory Dirichlet inner boundary
  and finite-volume half-cell regularisation at the axis.
- Added physical-metre slide bounds: decimal/exponent literals mean metres;
  integer literals retain pixel meaning. Bounds are validated, swapped and
  clamped with explicit log messages.
- Static and transient analyses now use Distributed Amperian force calculation
  consistently for the default `Forces` export.
- `conditions.json` preserves slide bounds in authored units and records the
  unit marker and multi-slide metadata.
- Added `MagFDMsolver --version` and unified project/package version 1.6.1.
- Nonlinear transient warm starts now reuse permeability only when the material
  RGB at that cell is unchanged; sliding one permeable material over another no
  longer leaks the previous material's permeability across the boundary.
- Removed the legacy polar plateau rule that accepted residuals up to 10x the
  requested tolerance. A Newton-Krylov iteration-limit result is retained for
  diagnostics, clearly marked `NOT CONVERGED`, and returns exit code `2`.
- Windows automatic OpenMP selection now uses physical cores while preserving
  explicit `omp.threads` and `OMP_NUM_THREADS` overrides. This avoids SMT
  contention in memory-bound AMG kernels.
- Corrected the polar permanent-magnet source curl. Cartesian magnetisation at
  each theta neighbour is now converted with that neighbour's own polar basis,
  and theta derivatives wrap consistently across periodic and anti-periodic
  seams. The old source generated an O(M/r) fictitious volume current even for
  uniform Cartesian magnetisation, which could cause non-physical harmonics in
  transient flux-linkage and derived EMF waveforms.
- Added analytic numerical regressions for both polar storage orientations,
  periodic and anti-periodic seams, second-order angular convergence, radial
  derivatives and non-periodic endpoint differences. All release-platform
  builds now execute these tests before packaging.

## WebUI

- Refreshed nonlinear templates for Eisenstat-Walker, realistic tolerance and
  100-iteration headroom.
- Added standalone Cartesian/Linear YAML generation with material detection.
- Added physical-unit slide templates and default-on detected air-gap slide.
- Added user-data ZIP export/import, including optional analysis results and a
  lazy file picker suitable for thousands of files.
- Unified File Manager listing for YAML configs, uploaded images, material
  libraries and analysis results, with open/download/delete actions,
  filter-aware select-all and cross-category bulk deletion, and the user-data
  import/export controls in their natural location.
- Added Image Properties, polar/cartesian coordinate previews, improved
  magnetisation preview, zoom/pan and pixel rulers.
- Fixed preset merge, polar magnetisation orientation, overlay alignment and
  viewer repaint issues.
- Standalone WebUI packages now include `sample_config.yaml` and
  `general_materials.yaml`.
- Polar slide generation maps the detected source-image air-gap radius through
  the actual warp interval (so changing output `nr` does not move the band),
  and defaults to the inner side.
- With `nonlinear_solver.verbose: false`, implementation-selection and AMGCL
  residual lines are suppressed; each transient step reports elapsed seconds
  and iteration count instead. `conditions.json` is written before solver
  initialization, and full-disc `r_start: 0` is valid in the viewer.
- New user directories receive the bundled `general_materials.yaml` library.

## Recommended nonlinear settings

```yaml
nonlinear_solver:
  enabled: true
  solver_type: newton-krylov
  max_iterations: 100
  tolerance: 1.0e-3
  eisenstat_walker: { enabled: true }
```

## Upgrade notes

- Existing integer slide bounds do not change meaning.
- Use a decimal point when authoring metres (`0.05`, not `50`).
- Remove deprecated `coarsen`, `coarsen_ratio`, `coarsening`, and old custom
  Galerkin/JFNK keys from new configurations.
- DD remains optional and polar-only. Treat it as an accuracy mode; keep the
  active electromagnetic band at full resolution.
- Automation should treat solver exit code `2` as nonlinear nonconvergence.
  Eisenstat-Walker accelerates inner solves but does not loosen the configured
  outer tolerance; compare exported fields only after both runs converge.
- See `docs/CONFIGURATION.md` for the normative configuration contract.

## Validation status

The pre-refresh branch head `dde0b7a` built successfully on Linux, macOS and
Windows, including standalone WebUI packages, the Windows installer and the
Windows AMGCL smoke step. The final release-candidate head requires a fresh CI
run. v1.6.1 adds a Windows release-contract smoke for version output, static
force export, export defaults and slide-unit metadata, plus cross-platform
analytic regression coverage for the polar permanent-magnet source curl;
broader machine-level numerical regression remains benchmark-driven.
