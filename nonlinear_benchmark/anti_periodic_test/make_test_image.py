"""Generate the two warped polar images for the anti-periodic BC benchmark.

  full_ring.png  - 360 rows (theta = 0..2pi) x 20 cols (r), with a +jz coil
                   patch at theta=45 deg and a -jz coil patch at theta=225 deg.
  half_ring.png  - 180 rows (theta = 0..pi)  x 20 cols (r), with only the
                   +jz coil patch at theta=45 deg.

The solver reads r along the image x-axis (r_orientation: horizontal) so
the image layout is (ntheta_rows, nr_cols).
"""
from PIL import Image

WIDTH_NR     = 20      # nr cells (radial direction, image cols)
HEIGHT_FULL  = 360     # ntheta_full (one cell per degree)
HEIGHT_HALF  = 180     # ntheta_half (theta in [0, pi])

# Coil patch: width 4 cells radially, height 4 cells in theta direction.
PATCH_W = 4
PATCH_H = 4
R_START = 8            # leave a few cells of air at r_inner / r_outer

# Colours (must match the yaml `materials.<name>.rgb`):
AIR       = (255, 255, 255)
COIL_POS  = (255,   0,   0)   # +jz
COIL_NEG  = (  0, 255,   0)   # -jz


def paint_patch(img, row_start, col_start, h, w, colour):
    px = img.load()
    for j in range(row_start, row_start + h):
        for i in range(col_start, col_start + w):
            px[i, j] = colour


def build(height, two_coils):
    img = Image.new("RGB", (WIDTH_NR, height), AIR)
    # +jz coil at theta = 45 deg. Centre row = height * 45 / 360.
    row_pos = int(round(height * 45 / 360)) - PATCH_H // 2
    paint_patch(img, row_pos, R_START, PATCH_H, PATCH_W, COIL_POS)
    if two_coils:
        # -jz coil at theta = 225 deg. Only present in the full-circle yaml,
        # where it stands in for the "ghost coil" the anti-periodic BC
        # would induce in the half-circle solve.
        row_neg = int(round(height * 225 / 360)) - PATCH_H // 2
        paint_patch(img, row_neg, R_START, PATCH_H, PATCH_W, COIL_NEG)
    return img


if __name__ == "__main__":
    full = build(HEIGHT_FULL, two_coils=True)
    full.save("full_ring.png")
    print(f"  wrote full_ring.png  size={full.size}  (+jz at row "
          f"{int(round(HEIGHT_FULL * 45 / 360))}, -jz at row "
          f"{int(round(HEIGHT_FULL * 225 / 360))})")

    half = build(HEIGHT_HALF, two_coils=False)
    half.save("half_ring.png")
    print(f"  wrote half_ring.png  size={half.size}  (+jz at row "
          f"{int(round(HEIGHT_HALF * 45 / 360))})")
