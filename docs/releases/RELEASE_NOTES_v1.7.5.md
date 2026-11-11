# OpenMagFDM v1.7.5

OpenMagFDM v1.7.5 corrects the electromagnetic force evaluation, adds residual
noise reduction for flux/EMF display plots, and makes MCP connections recover
across server restarts without re-issuing tokens.

## Highlights

- **Corrected electromagnetic force:** The single-state Amperian force
  evaluation no longer produces ghost forces on non-magnetic bodies
  (mu_r=1, no current, no magnetization). Source ownership is assigned before
  differentiation so overlapping material masks cannot double-count, induced
  face currents stay consistent with the magnetic-field operator, and polar
  signs, nodal coordinates, storage orientation and the force-vector basis
  are corrected. See the
  [Amperian force verification](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.5/docs/development/AMPERIAN_FORCE_VERIFICATION.md).
- **Flux/EMF display estimates:** Flux-linkage and back-EMF plots can reduce
  residual-induced differentiation noise with a robust local polynomial fit or
  a repeating-state blend over explicitly declared full-state periods
  (rotation, translation or combined). Off by default, display data only —
  solver results, convergence and exported files are unchanged. Estimates
  exclude non-converged steps through `flux_linkage_status.csv`, keep missing
  values missing, and run in a cancellable Web Worker. Specifications and
  limits are in the WebUI's Flux/EMF estimate help and `EMF_FILTER.md`.
- **MCP restart recovery:** A listener that lost its port to another process
  now retries binding automatically and shows the retry state in Help &
  Integrations. The stdio bridge re-reads the saved port on reconnect and
  waits up to 30 s for a server that starts later. Authorization refusals are
  reported separately from transport outages, so a restart is no longer
  mistaken for a token problem. Existing tokens keep working across
  restarts.
- **Durable MCP credentials:** Token and settings writes are acknowledged only
  after reaching the filesystem, and corrupt or missing private settings fail
  loudly instead of silently resetting authorization state.

## Upgrade

Stop the server before replacing binaries and public assets. Preserve workspace
directories and OS-private account, job and MCP settings. Keep the same
installation path to retain the existing management profile. Existing 2D YAML
and result files need no conversion. MCP client settings survive the upgrade;
re-pairing or new tokens are only needed if private settings were already
lost. Solver results keep their stored force values; recompute analyses that
need the corrected force evaluator.

See the [CLI manual](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.5/docs/CLI.md),
[REST API manual](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.5/docs/REST_API.md)
and [operations manual](https://github.com/Marble-GP/OpenMagFDM/blob/v1.7.5/docs/OPERATIONS.md).

## Packages and validation

The tagged workflow builds Windows, Linux and macOS solver packages, standalone
WebUI packages and the Windows installer. WebUI-only packages require a separate
native solver. The macOS WebUI package targets Intel x64 and requires Rosetta 2
on Apple Silicon.

Pre-release validation passed 179 Node regression tests, all native CTest
suites including the Amperian force regression set, and the packaged MCP
validation in CI. The flux/EMF estimate is display-only and its accuracy on a
real motor remains unverified; synthetic-signal checks, raw overlays and
harmonic-retention limits are documented in `EMF_FILTER.md`.
Experimental GPU rendering and 3D solver features remain excluded from stable
packages.
