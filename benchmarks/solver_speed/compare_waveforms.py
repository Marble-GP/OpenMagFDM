"""Quality gate for a complete, aligned, strictly converged rotation sweep.

Both inputs are solver result directories. EMF uses the same periodic centered
difference in both runs; dt defaults to one step, which cancels in relative
metrics. No time shifting, fitting or amplitude rescaling is performed.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(root, points):
    with (root / "FluxLinkage/flux_linkage.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    with (root / "FluxLinkage/flux_linkage_status.csv").open(encoding="utf-8-sig", newline="") as f:
        status = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    if len(rows) != points or steps != list(range(points)):
        raise ValueError("A complete sweep with consecutive steps [0, points) is required")
    if len(status) != points or [int(s["step"]) for s in status] != steps:
        raise ValueError("Missing or misaligned solver statuses")
    if any(s["strict_converged"] != "1" or not np.isfinite(float(s["residual"])) for s in status):
        raise ValueError("Unconverged/nonfinite reference or trial: waveform gate is not evaluable")
    names = [c for c in columns if c != "step"]
    values = np.asarray([[float(r[name]) for name in names] for r in rows])
    if not names or not np.all(np.isfinite(values)):
        raise ValueError("Missing/nonfinite flux values")
    return names, values


def metrics(reference, trial, fundamental):
    scale = np.linalg.norm(reference)
    if scale <= np.finfo(float).tiny:
        raise ValueError("Zero reference waveform: normalized quality metrics undefined")
    spectrum = 2 * np.abs(np.fft.rfft(reference)) / reference.size
    candidate = 2 * np.abs(np.fft.rfft(trial)) / reference.size
    if fundamental < 1 or fundamental >= len(spectrum) or spectrum[fundamental] <= 1e-12 * spectrum.max():
        raise ValueError("Reference fundamental is missing or too small")
    harmonics = np.arange(len(spectrum))
    selected = (harmonics != 0) & (harmonics != fundamental)
    diff = np.abs(candidate - spectrum) / spectrum[fundamental]
    return dict(normalized_rms=float(np.linalg.norm(trial-reference)/scale),
                max_harmonic_difference_over_fundamental=float(np.max(diff[selected])),
                harmonic_difference_over_fundamental=diff.tolist())


def compare(baseline, trial, points=248, fundamental=2, dt=1.0):
    if points < 4 or dt <= 0:
        raise ValueError("points >=4 and dt >0 required")
    names, base = load(baseline, points)
    trial_names, test = load(trial, points)
    if names != trial_names:
        raise ValueError("Flux channel names/order differ")
    base_emf = -(np.roll(base, -1, axis=0)-np.roll(base, 1, axis=0))/(2*dt)
    test_emf = -(np.roll(test, -1, axis=0)-np.roll(test, 1, axis=0))/(2*dt)
    channels = {}
    for i, name in enumerate(names):
        flux = metrics(base[:, i], test[:, i], fundamental)
        emf = metrics(base_emf[:, i], test_emf[:, i], fundamental)
        passed = flux["normalized_rms"] <= 0.001 and emf["normalized_rms"] <= 0.005
        passed &= max(flux["max_harmonic_difference_over_fundamental"],
                      emf["max_harmonic_difference_over_fundamental"]) <= 0.001
        channels[name] = dict(flux=flux, emf=emf, passed=bool(passed))
    return dict(points=points, fundamental_bin=fundamental, timestep=dt,
                passed=all(c["passed"] for c in channels.values()), channels=channels)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("trial", type=Path)
    ap.add_argument("--points", type=int, default=248)
    ap.add_argument("--fundamental", type=int, default=2, help="mechanical-period pole-pair harmonic")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    try:
        result = compare(a.baseline, a.trial, a.points, a.fundamental, a.dt)
    except (ValueError, OSError, KeyError) as e:
        result = dict(passed=None, reason=str(e))
    text = json.dumps(result, indent=2, allow_nan=False)
    if a.output:
        with a.output.open("x", encoding="utf-8") as f:
            f.write(text)
    print(text)
    raise SystemExit(0 if result["passed"] is True else 2)


if __name__ == "__main__":
    main()
