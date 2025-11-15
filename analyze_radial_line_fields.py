#!/usr/bin/env python3
import numpy as np

# Read stress vector data
data = []
with open("output_20251115_122217/StressVectors/step_0001.csv", 'r') as f:
    for line in f:
        if line.startswith('#') or 'i_pixel' in line:
            continue
        values = line.strip().split(',')
        try:
            data.append([
                float(values[2]), float(values[3]),  # x, y
                float(values[4]), float(values[5]),  # fx, fy
                float(values[6]),  # ds
                float(values[7]), float(values[8]),  # nx, ny
                float(values[9]), float(values[10]),  # Bx, By
                float(values[11]),  # B_mag
            ])
        except:
            pass

data = np.array(data)
x, y, fx, fy, ds, nx, ny, Bx, By, B_mag = data.T

# Boundary only
mask = ds > 0
x_b, y_b, fx_b, fy_b, ds_b, nx_b, ny_b, Bx_b, By_b, B_mag_b = x[mask], y[mask], fx[mask], fy[mask], ds[mask], nx[mask], ny[mask], Bx[mask], By[mask], B_mag[mask]

r_b = np.sqrt(x_b**2 + y_b**2)
theta_b = np.arctan2(y_b, x_b)

# Normal in polar coords
n_r = nx_b * np.cos(theta_b) + ny_b * np.sin(theta_b)
n_theta = -nx_b * np.sin(theta_b) + ny_b * np.cos(theta_b)

# Classify boundary types
is_circular = np.abs(n_r) > 0.8  # r=const (inner/outer)
is_radial = np.abs(n_theta) > 0.8  # theta=const (left/right)

# Radial lines
x_rad = x_b[is_radial]
y_rad = y_b[is_radial]
r_rad = r_b[is_radial]
theta_rad = theta_b[is_radial]
Bx_rad = Bx_b[is_radial]
By_rad = By_b[is_radial]
B_mag_rad = B_mag_b[is_radial]
fx_rad = fx_b[is_radial]
fy_rad = fy_b[is_radial]
ds_rad = ds_b[is_radial]
n_theta_rad = n_theta[is_radial]

# Left and right
is_left = n_theta_rad < 0
is_right = n_theta_rad > 0

print("=== Magnetic Field on Radial Boundaries ===")
print(f"\nLeft side (θ ≈ 0°～20°):")
print(f"  |B| range: [{B_mag_rad[is_left].min():.4f}, {B_mag_rad[is_left].max():.4f}] T")
print(f"  |B| mean:  {B_mag_rad[is_left].mean():.4f} T")

print(f"\nRight side (θ ≈ 40°～49°):")
print(f"  |B| range: [{B_mag_rad[is_right].min():.4f}, {B_mag_rad[is_right].max():.4f}] T")
print(f"  |B| mean:  {B_mag_rad[is_right].mean():.4f} T")

print(f"\nMagnetic field asymmetry: {B_mag_rad[is_right].mean() / B_mag_rad[is_left].mean():.4f}")

# Check force direction
print(f"\n=== Force on Radial Boundaries ===")
f_mag_left = np.sqrt(fx_rad[is_left]**2 + fy_rad[is_left]**2)
f_mag_right = np.sqrt(fx_rad[is_right]**2 + fy_rad[is_right]**2)

print(f"\nLeft side:")
print(f"  |f| mean: {f_mag_left.mean():.2f} N/m")
print(f"  fx mean: {fx_rad[is_left].mean():+.2f} N/m")
print(f"  fy mean: {fy_rad[is_left].mean():+.2f} N/m")

print(f"\nRight side:")
print(f"  |f| mean: {f_mag_right.mean():.2f} N/m")
print(f"  fx mean: {fx_rad[is_right].mean():+.2f} N/m")
print(f"  fy mean: {fy_rad[is_right].mean():+.2f} N/m")

# Torque per unit length
dtorque_left = (x_rad[is_left] * fy_rad[is_left] - y_rad[is_left] * fx_rad[is_left])
dtorque_right = (x_rad[is_right] * fy_rad[is_right] - y_rad[is_right] * fx_rad[is_right])

print(f"\n=== Torque per Unit Length ===")
print(f"Left:  mean dτ/ds = {dtorque_left.mean():+.2f} N")
print(f"Right: mean dτ/ds = {dtorque_right.mean():+.2f} N")
print(f"Ratio (right/left): {dtorque_right.mean() / dtorque_left.mean():.4f}")

# Check r distribution on left vs right
print(f"\n=== Radial Position Distribution ===")
print(f"Left side  r: [{r_rad[is_left].min():.4f}, {r_rad[is_left].max():.4f}] m, mean={r_rad[is_left].mean():.4f} m")
print(f"Right side r: [{r_rad[is_right].min():.4f}, {r_rad[is_right].max():.4f}] m, mean={r_rad[is_right].mean():.4f} m")
