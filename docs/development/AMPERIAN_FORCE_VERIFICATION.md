# Amperian force correction and verification

Date: 2026-11-11. Branch: `fix/ampere-force-evaluation`.
Baseline: v1.7.4, `ccf2c4572d345f30bed66b4ef66690f2e88b2144`.
Status: implemented and locally tested; not a release or a universal accuracy certification.

## What changed

The default force evaluator remains Amperian. It uses one solved snapshot of Az,
permeability, prescribed magnetization and free current. It performs no additional
field solves, time differencing, trajectory fitting or smoothing.

For each selected material b, the current is

```
delta_nu_b = chi_b * (1/mu - 1/mu0)
J_b = chi_b * J_free + curl(chi_b * M_source)
      + div(delta_nu_b * grad(Az))
F_b = integral(J_b cross B) dArea
```

`M_source` is the magnetization actually used on the solver's right-hand side.
Its current convention is Br/mu0, so reconstructing it as Br/mu would be wrong.
The induced term is curl of the induced magnetization, expressed through the
face gradient and arithmetic face reluctivity used by the field operator
(equivalently, harmonic face permeability). This avoids the inconsistent old
sequence of cell-centre B, discontinuous cell-centre M, and a second derivative.
It includes surface currents without tracing jagged boundary normals.

Source ownership is assigned **before** differentiation. The full resulting
current support is integrated, with no dilation over a global current map.
Splitting a body into otherwise identical material colours therefore preserves
its summed force. Individual loads on touching materials are source partitions,
not certified mechanical contact tractions; sum the rows for the assembly.

Additional fixes:

- Include previously omitted free current and prescribed permanent magnetization.
- Use the solver's `r_start + i*dr` and Cartesian nodal coordinates, not half-cell offsets.
- Apply polar anti-periodic signs to fields and source components, never permeability.
- Respect both polar storage orientations and the global theta offset for force vectors.
- Recalculate the field for the public default `step=-1` call rather than reusing an uninitialized cache.
- Use compensated accumulation and 15-digit scientific CSV output.
- Label torque as `N*m/m` (per unit axial depth). Existing WebUI parsing recognizes column names by prefix.

There is no change to the field equations, material constitutive laws, YAML schema,
or selected `calc_force` materials. The former optional C++ smoothing parameter
now rejects nonzero values explicitly. Magnetized selected bodies touching a
nonperiodic physical boundary are rejected: the absent exterior cannot supply a
closed-body force. Enlarge the domain or select an interior body. For accuracy,
leave several air cells between the body and external boundaries.

## Verification

Built with MSVC 19.51, Release, x64. All six CTest suites passed, including the new
`amperian_force` suite and existing field-operator, nonlinear/sliding, threading,
input-snapshot and polar-magnetization regressions. Generated inputs, outputs,
builds and detailed logs are outside the repository under
`../qa/ampere_fix_20261111/`, with an ignore-all `.gitignore`.

The committed regression constructs its own models and checks the public native
solver and exported CSV, not just the numerical helper:

| Check | Result |
|---|---|
| Current-carrying mu_r=1 conductor in 0.1 T | Fx = -14.4 N/m; v1.7.4 returned approximately zero |
| mu_r=1 permanent magnet in transverse field | Torque = 11.4591559026 N*m/m, matching the derivative of U=-m dot B; v1.7.4 returned approximately zero |
| Inert mu_r=1, J=0, M=0 marker in a nonuniform field | Exactly zero force and torque |
| Two touching colours versus one colour, identical properties | Summed force/torque agree within 1e-12 absolute; old summed Fx was doubled |
| Static versus three current steps (zero, positive, negative) | Correct zero and direction reversal; transient error <1e-5 N/m, allowing iterative field residual |
| Full circle versus anti-periodic half circle | Half-sector torque times two agrees within 1e-10 absolute |
| Horizontal/vertical polar storage and rotated coordinate frame | Torque invariant; force vector rotates correctly |
| Unequal Cartesian pitches, simultaneous free and PM sources | Analytic force and torque agree within 1e-10 absolute |
| Magnetization truncated by a physical boundary | Explicit rejection |

Polar rotor comparisons (all torques per unit depth):

| Model | Amperian | Arkkio | Air-domain virtual work | Amperian/Arkkio difference |
|---|---:|---:|---:|---:|
| Linear, 40 radial x 64 angular nodes | -0.00986288378616 | -0.00986469782523 | -0.00986288378746 | 0.0183892% |
| Linear, 79 x 128 nodes | -0.00956315620741 | -0.00956343648196 | -0.00956304442162 | 0.00293069% |
| Nonlinear B-H test, 40 x 64 nodes | -0.00992317490860 | -0.00992500003894 | -0.00992317490955 | 0.0183892% |

The old cell-centre-M formulation, even after fixing its seam and radius bugs,
gave -0.00952709756337 on the coarse linear model, about 3.42% from Arkkio.
The shipped v1.7.4 half-circle result had the opposite sign. Thus the change
addresses both correctness defects and interface discretization error.

Arkkio is evaluated over a verified source-free air annulus, with trapezoidal
radial averaging. The energy reference uses an angular shear of that annulus,
theta_new = theta + epsilon*g(r), with g'=-1/gap_width. At fixed nodal Az it
evaluates the transformed gradient energy on radial faces at +/-epsilon and
takes -dW/dangle. The body moves rigidly and only the air annulus deforms; no
additional solve or other time step is needed. The PM test supplies a separate
analytic interaction-energy reference. Arkkio and air virtual work express the
same continuum stress identity, so agreement is a consistency check, not three
independent physical ground truths.

A naive finite shift of rasterized reluctivity was also explored internally.
It produced substantial discretization error and was not used as an oracle or
added to production. Likewise, the existing WebUI difference of system energy
between arbitrary excitation steps is not a valid force reference.

In the Cartesian thin-iron diagnostic, corrected Fx is 0.0263829 N/m, compared
with air-contour values 0.0260897--0.0261841 N/m (varying contour distance).
The old value was 0.0333682 N/m. This illustrates why accuracy claims must be
geometry-specific; the smaller polar discrepancy is not universal.

Ten native cases were also rerun against official v1.7.4 inputs. Maximum absolute
Az difference was 3.47e-18, consistent with build/rounding differences. Within
each regression, force evaluation leaves Az exactly unchanged.

## Remaining limits and use

- The external motor model behind the reported 60% error was not provided. Its
  particular error has not been reproduced or certified fixed; no empirical
  scaling factor is applied.
- Refining the polar raster changes the torque itself by about 3% in this test,
  despite small method-to-method differences. Check geometry and field mesh
  convergence, gap resolution, nonlinear residual and total assembly ownership.
- This is a macroscopic total-body Lorentz/Amperian force, not magnetostrictive
  stress or the mechanical traction between contacting magnetic materials.
- The nonlinear test is one isotropic B-H fixture, not a saturation/PM/contact
  acceptance matrix. General nonlinear, loaded-PM and very thin-gap models need
  further model-specific validation.
- The production evaluator does not automatically find an Arkkio annulus or
  substitute a reference result. Those references live in the regression tests.
- Legacy alternative Shell/interpolation methods audited in the original plan
  are unchanged and are not activated by this correction.
- Per-material magnetic energy is still an uncomputed legacy zero placeholder,
  now stated in the CSV comments. `_SYSTEM_TOTAL` remains the existing energy
  diagnostic and is not substituted for a physically constrained energy derivative.

Run after building with `BUILD_TESTING=ON`:

```
ctest --test-dir <build> -C Release --output-on-failure
```

Windows execution requires the native dependency DLL directory on PATH. See
[the original audit and plan](FORCE_EVALUATION_PLAN.md) for issue history and
broader acceptance work. No release tag or published binary is changed here.
