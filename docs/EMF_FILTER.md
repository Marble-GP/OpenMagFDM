# Flux and Back-EMF display estimates

This experimental WebUI feature processes **display data only**. It does not
change flux CSVs, magnetic fields, solver convergence, input YAML or exported
solver results. It uses multiple steps; this is separate from the single-state
electromagnetic force evaluator.

## Use

1. Load a result and add **Flux Linkage Timeline** or **Back-EMF Timeline**.
2. Open **Plot Configure** and find **Flux / EMF estimate** for that plot.
3. Choose a processing mode, set the fit window and select **Apply**.

| Mode | Behavior |
|---|---|
| Off — raw data | Default. Flux is unchanged; the derivative uses adjacent samples and their actual step spacing. |
| Robust local fit | Fit cubic polynomials to neighbouring flux samples with adaptive weights, then differentiate the polynomials. |
| Repeating-state blend + local fit | Combine matching phase positions in explicitly declared complete state periods, then fit and differentiate in the periodic coordinate. |

The fit window is an odd number of samples from 5 to 51; the default is 9.
Larger windows suppress more variation, including potentially real harmonics.
Raw and estimated curves remain separate legend entries. Hover over an estimate
to compare its raw value. Turn processing Off to return to the original display.
Settings apply to the current plot/result and reset when that plot uses a
different result; they are not persistent solver configuration.

## Rotation, translation and repeat equivalence

The **full-state period**, in solver steps, must include repetition of geometry,
all excitations and boundary conditions. For rotation, equal rotor angle alone
does not establish equivalence if current phase/amplitude differs. For
translation, equal position modulo a spatial pitch is insufficient if excitation
or material state changes. Combined motions require their complete joint repeat.
Electrical symmetry can shorten a period only when the same measured flux
quantity, coil identity and sign also repeat. Anti-periodic sign reversal is not
automatically inferred; specify the full same-sign period.

The initial implementation requires at least two complete periods and more
repeated phase positions than the selected fit window. It does not infer a
period from the waveform, estimate current-phase equivalence, rotate/translate
field arrays, or blend arbitrary visually similar steps. Such automatic matching
needs explicit per-step motion and excitation metadata.

A coarse consistency check withholds the repeating-state estimate if the RMS
spread of matched flux samples exceeds 5% of the observed flux range. This is
a heuristic guard, not proof that two physical states match. Startup transients,
ramps and changing operating points should use local mode or raw data.

## Numerical and convergence handling

Processing acts on flux **before differentiation**, not on an already noisy EMF
curve. Fits use distance weights, a scaled local coordinate, reorthogonalized QR
and Huber residual weights. Polynomial-fit residuals are different from the
magnetic solver's equation residuals: neither is automatically the physical
error of the flux quantity.

When `FluxLinkage/flux_linkage_status.csv` is available, samples with
`strict_converged != 1`, or missing status rows, are excluded from the estimate.
Positive reported solver residuals additionally supply bounded heuristic weights
`max(0.1, 1/(1+(residual/median_positive_residual)^2))`. These are not inverse
error variances. Older results without this file can still be estimated, with
**convergence metadata unavailable** displayed. An inaccessible or malformed
status file withholds the estimate while retaining raw curves.

Missing flux values remain missing rather than becoming zero. Fits do not fill
excluded points or cross step gaps larger than 1.5 times the median spacing.
Endpoints use local one-sided fits; only explicitly declared repeating-state
mode wraps phase. Derivatives are evaluated at the actual midpoint between
adjacent step coordinates. Nonuniform step intervals are divided out.

The current result CSV has step coordinates, not physical timestamps. The
Back-EMF plot therefore remains **-dPhi/dstep [Wb/m per step]**, not voltage in V.
Converting to voltage needs the appropriate physical time scale and axial-depth
/ flux-linkage normalization. A smooth curve does not supply those missing units.

Work is performed in a cancellable Web Worker where available; stale requests
cannot overwrite a newly requested plot. Filtering is bounded to 100,000 points
per series, 64 series and 500,000 total samples. Without workers, only small
datasets are filtered. Worker/limit errors retain raw data and show the reason.

## Verification and limits

Numerical tests cover exact cubic differentiation on irregular coordinates,
endpoint handling, missing data, non-converged steps, metadata failure, state
period mismatch, harmonic retention, result-scoped controls, raw overlays and
worker cancellation. The full Node regression suite passed (173 tests).

For an artificial 288-sample signal with period 72, unit-amplitude sine and
0.01-amplitude non-periodic sinusoidal noise, derivative RMS error was:

| Method | RMS error against the analytic derivative |
|---|---:|
| Raw adjacent difference | 0.0107368 |
| Local fit, window 9 | 0.00136282 |
| Four-period blend, window 9 | 0.0000384584 |

This is a synthetic example, **not** a measured improvement on a motor model.
A clean fifth-harmonic test retains amplitude within 5% at this window/period;
other harmonics, windows and sampling rates have different attenuation.
The plotted RMS difference from raw data measures the effect of processing,
not uncertainty or an accuracy bound. Correlated discretization errors may
repeat exactly with the machine position and will not average away.

An isolated preview server was prepared, but live visual inspection was blocked
by unavailable browser automation and a Windows activation failure. DOM/control
and trace-generation behavior was tested automatically; visual acceptance and
comparison with a tighter-tolerance motor solve remain pending. Generated preview
data and scripts are outside the repository under `qa/emf_filter_20261111`.

For background, see [SciPy's Savitzky-Golay derivative interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)
and [MathWorks' locally weighted and robust smoothing description](https://www.mathworks.com/help/curvefit/smoothing-data.html).
The browser implementation is a local weighted cubic fit, not a dependency on
either library or a claim to reproduce all of their algorithms.
