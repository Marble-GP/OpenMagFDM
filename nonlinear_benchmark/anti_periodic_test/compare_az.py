"""Compare the half-circle anti-periodic solve against the full-circle
reference (Phase B.4). Returns exit 0 on PASS, 1 on FAIL.

The full-circle reference Az has shape (ntheta_full, nr).
The half-circle AP solve produces Az with shape (ntheta_half, nr).
Reconstruction:
  Az_full_reconstructed[θ, r] = Az_half[θ, r]                 for θ ∈ [0, π]
  Az_full_reconstructed[θ, r] = -Az_half[θ - π, r]            for θ ∈ [π, 2π]

We then compute L2 norm of (reconstructed - reference) divided by
the L2 norm of the reference. PASS if ratio < 1e-3.

Usage:
    python compare_az.py <full_output_dir> <half_output_dir>

The output dirs are the per-yaml result folders that the solver writes
into. The TIFF lives at <dir>/Az/step_0000.tiff.
"""
import sys
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:
    print("FAIL  tifffile not installed -- pip install tifffile", file=sys.stderr)
    sys.exit(2)


TOLERANCE = 1e-3


def load_az(dir_path):
    p = Path(dir_path) / "Az" / "step_0000.tiff"
    if not p.exists():
        # Some solver builds write a different padding; try a few.
        candidates = sorted((Path(dir_path) / "Az").glob("step_*.tiff"))
        if not candidates:
            raise SystemExit(f"FAIL  no Az TIFF under {p.parent}")
        p = candidates[0]
    arr = tifffile.imread(str(p))
    return arr.astype(np.float64), p


def reconstruct_full(az_half, ntheta_full):
    """Mirror-and-negate the half-circle AP solve to a full-circle field."""
    ntheta_half, nr = az_half.shape
    if 2 * ntheta_half != ntheta_full:
        raise SystemExit(
            f"FAIL  half ntheta ({ntheta_half}) is not exactly half of "
            f"full ntheta ({ntheta_full}); regenerate the images so ntheta_full = 2 x ntheta_half."
        )
    out = np.empty((ntheta_full, nr), dtype=np.float64)
    out[:ntheta_half, :] = az_half
    out[ntheta_half:, :] = -az_half
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    full_dir, half_dir = sys.argv[1], sys.argv[2]

    az_full, full_path = load_az(full_dir)
    az_half, half_path = load_az(half_dir)

    print(f"[compare]    full: {full_path}  shape={az_full.shape}")
    print(f"[compare]    half: {half_path}  shape={az_half.shape}")

    az_recon = reconstruct_full(az_half, az_full.shape[0])

    diff = az_recon - az_full
    ref_norm = np.linalg.norm(az_full)
    diff_norm = np.linalg.norm(diff)
    ratio = diff_norm / max(ref_norm, 1e-30)
    print(f"[compare]    L2(reconstructed - reference) / ||Az_ref|| = {ratio:.3e}")

    if ratio < TOLERANCE:
        print(f"             PASS  (tolerance {TOLERANCE:.1e})")
        sys.exit(0)
    else:
        print(f"             FAIL  (tolerance {TOLERANCE:.1e})")
        # Quick diagnostic: half of the reconstructed field that should
        # match the reference verbatim (the un-flipped part).
        ntheta_half = az_half.shape[0]
        half_diff = az_recon[:ntheta_half, :] - az_full[:ntheta_half, :]
        half_ratio = np.linalg.norm(half_diff) / max(ref_norm, 1e-30)
        print(f"             ... un-flipped half error ratio = {half_ratio:.3e}")
        print(f"             ... if un-flipped is large too, the issue is geometry, "
              f"not the sign flip")
        sys.exit(1)


if __name__ == "__main__":
    main()
