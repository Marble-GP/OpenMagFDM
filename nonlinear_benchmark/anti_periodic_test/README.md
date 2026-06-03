# Anti-periodic θ-boundary benchmark (Phase B.4)

A numerical regression test for the polar anti-periodic boundary condition
in `MagneticFieldAnalyzer.cpp` (matrix-build-time sign flip at L10130-10340
and post-solve interpolation at L6617-6620 of the pre-v1.5 layout — line
numbers drift, the helpers there are `is_antiperiodic` and the seam
recompute).

## Test design

Anti-periodic BC means `Az(θ_max, r) = −Az(θ_min, r)` on the cut. The
canonical equivalence is:

> A half-circle solve `[0, π]` with anti-periodic BC at `θ_min` and
> `θ_max` reproduces the same field as a full-circle solve `[0, 2π]`
> with periodic BC and an extra opposite-sign source at `θ + π`.

So we set up two yamls:

- `full_circle_ref.yaml` — full circle, periodic θ-BC (`value: 1.0`),
  two coil patches: `+jz` at `θ = π/4` and `−jz` at `θ = π/4 + π`.
- `half_circle_ap.yaml` — half-circle `θ ∈ [0, π]`, anti-periodic
  θ-BC (`value: −1.0`), single coil patch `+jz` at `θ = π/4`.

If the algebra is correct, then for every grid cell:

- `Az_full(r, θ) = Az_half(r, θ)`     for `θ ∈ [0, π]`
- `Az_full(r, θ) = −Az_half(r, θ − π)` for `θ ∈ [π, 2π]`

`compare_az.py` loads the two TIFFs, applies the reconstruction, and
asserts `L2(diff) / ||Az_full|| < 1e-3`. (Tolerance is loose because the
two solves are on different grids — the ntheta of the half-circle setup
is exactly half of the full-circle setup, so cells align without
interpolation, but discretization noise still produces non-zero
residuals.)

## Files

- `make_test_image.py` — writes `full_ring.png` (360×20) and
  `half_ring.png` (180×20) with the coil patches at the right indices.
- `full_circle_ref.yaml` — full-circle reference solver config.
- `half_circle_ap.yaml` — half-circle AP-BC solver config.
- `compare_az.py` — reconstructs the full-circle field from the
  half-circle solution and compares against the reference.
- `run_test.sh` — driver script: generates images, runs the solver
  twice, runs the Python comparison, prints PASS/FAIL.

## Running

```bash
cd nonlinear_benchmark/anti_periodic_test
bash run_test.sh /path/to/MagFDMsolver
```

The script exits non-zero if `L2 error / ||Az_ref|| > 1e-3` so it can be
wired into CI. Expected output on a healthy build:

```
[setup]      writing full_ring.png (360x20) and half_ring.png (180x20)
[full]       running MagFDMsolver on full_circle_ref.yaml ...
[half]       running MagFDMsolver on half_circle_ap.yaml ...
[compare]    L2(reconstructed − reference) / ||Az_ref|| = 4.21e-05
             PASS  (tolerance 1.0e-03)
```

## If the test fails

The most likely failure modes, in order of probability:

1. **Sign flip in the wrong direction** — `Az_full(π+δ) = +Az_half(δ)`
   instead of `−Az_half(δ)`. Check the `is_antiperiodic ? -1.0 : 1.0`
   wiring in the polar matrix builder.
2. **Row mapping off by one** at the seam. The half→full reconstruction
   pairs `θ_half = δ` with `θ_full = π + δ`; if the ntheta of the
   half solve is N, the full ntheta must be 2N. The grid match is
   asserted at the top of `compare_az.py`.
3. **Source not symmetric in the reference** — `+jz` and `−jz` must
   live at exactly `(θ, r)` and `(θ + π, r)` respectively. Off-by-one
   indices in `make_test_image.py` shift one of the patches and ruin
   the equivalence.

When debugging, dump `Az_full[0::2, :]` vs `Az_half[:, :]` for the
`θ ∈ [0, π]` half — they should agree to solver precision.
