#!/usr/bin/env python3
import numpy as np

# Read first step image to check core position
data = []
with open("output_20251115_122217/StressVectors/step_0001.csv", 'r') as f:
    for line in f:
        if line.startswith('#') or 'i_pixel' in line:
            continue
        values = line.strip().split(',')
        try:
            i = int(values[0])
            j = int(values[1])
            ds = float(values[6])
            material = values[12].strip()
            if ds > 0:  # Boundary pixel
                data.append([i, j, material])
        except:
            pass

data = np.array(data)
i_pixels = data[:, 0].astype(int)
j_pixels = data[:, 1].astype(int)

print("=== Core Position in Image ===")
print(f"Image size (from config): ntheta=500 (j-direction), nr=500 (i-direction)")
print(f"r_orientation: horizontal → i=r-index, j=θ-index")
print(f"\nBoundary pixel ranges:")
print(f"i-pixels (r-direction): [{i_pixels.min()}, {i_pixels.max()}]")
print(f"j-pixels (θ-direction): [{j_pixels.min()}, {j_pixels.max()}]")

# For theta_range = π/3 with 500 pixels, periodic boundaries are at j=0 and j=500 (wraps to 0)
# Core should be centered at j=250 (30°) for symmetry
j_center = (j_pixels.min() + j_pixels.max()) / 2
theta_range_deg = 60.0
j_center_deg = j_center / 500 * theta_range_deg

print(f"\nCore θ-center: j={j_center:.1f} → θ={j_center_deg:.2f}° (expected 30° for symmetry)")
print(f"Core θ-width: {j_pixels.max() - j_pixels.min()} pixels")

# Check slide region
print(f"\n=== Slide Region (from config) ===")
print(f"slide_region: i ∈ [110, 390] (width 280 pixels)")
print(f"Core i-range: [{i_pixels.min()}, {i_pixels.max()}]")

# For perfect periodicity:
# Core left edge should be at j=0 or very close
# Core right edge should be at j=theta_width where theta_width divides 500 evenly
print(f"\n=== Periodicity Check ===")
print(f"For perfect periodicity, core edges should be at θ=0° and θ=60°")
print(f"Current left edge:  j={j_pixels.min()} → θ={j_pixels.min()/500*60:.2f}°")
print(f"Current right edge: j={j_pixels.max()} → θ={j_pixels.max()/500*60:.2f}°")
