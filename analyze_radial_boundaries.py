#!/usr/bin/env python3
import numpy as np

# Read stress vector data
data = []
with open("output_20251115_122217/StressVectors/step_0001.csv", 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if not line.startswith('#') and 'i_pixel' not in line:
            values = line.strip().split(',')
            try:
                data.append([
                    int(values[0]), int(values[1]),  # i, j
                    float(values[2]), float(values[3]),  # x, y
                    float(values[4]), float(values[5]),  # fx, fy
                    float(values[6]),  # ds
                    float(values[7]), float(values[8]),  # nx, ny
                    float(values[9]), float(values[10]),  # Bx, By
                ])
            except:
                pass

data = np.array(data)
x, y, fx, fy, ds, nx, ny, Bx, By = data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6], data[:, 7], data[:, 8], data[:, 9], data[:, 10]

# Boundary only
mask = ds > 0
x_b, y_b, fx_b, fy_b, ds_b, nx_b, ny_b, Bx_b, By_b = x[mask], y[mask], fx[mask], fy[mask], ds[mask], nx[mask], ny[mask], Bx[mask], By[mask]

r_b = np.sqrt(x_b**2 + y_b**2)
theta_b = np.arctan2(y_b, x_b)

# Normal in polar coords
n_r = nx_b * np.cos(theta_b) + ny_b * np.sin(theta_b)
n_theta = -nx_b * np.sin(theta_b) + ny_b * np.cos(theta_b)

# Radial lines (θ=const)
is_radial = np.abs(n_theta) > 0.8

theta_rad = theta_b[is_radial]
r_rad = r_b[is_radial]
n_theta_rad = n_theta[is_radial]
fx_rad = fx_b[is_radial]
fy_rad = fy_b[is_radial]
x_rad = x_b[is_radial]
y_rad = y_b[is_radial]
ds_rad = ds_b[is_radial]

# Classify into left (n_theta < 0) and right (n_theta > 0)
is_left = n_theta_rad < 0
is_right = n_theta_rad > 0

print("=== Radial Line Boundary Analysis ===")
print(f"Total radial line pixels: {len(theta_rad)}")
print(f"Left side (n_θ < 0):  {is_left.sum()} pixels")
print(f"Right side (n_θ > 0): {is_right.sum()} pixels")

# Torque contribution
dtorque_rad = (x_rad * fy_rad - y_rad * fx_rad) * ds_rad
torque_left = dtorque_rad[is_left].sum()
torque_right = dtorque_rad[is_right].sum()

print(f"\n=== Torque Contribution ===")
print(f"Left side:  {torque_left:+.6f} N·m")
print(f"Right side: {torque_right:+.6f} N·m")
print(f"Total:      {torque_left + torque_right:+.6f} N·m")
print(f"Difference: {torque_left - torque_right:+.6f} N·m")

# Check theta distribution
print(f"\n=== Theta Distribution ===")
print(f"Left side  θ: [{np.degrees(theta_rad[is_left].min()):.2f}, {np.degrees(theta_rad[is_left].max()):.2f}] deg")
print(f"Right side θ: [{np.degrees(theta_rad[is_right].min()):.2f}, {np.degrees(theta_rad[is_right].max()):.2f}] deg")

# Check if left and right are symmetric
theta_range = 1.0471975512  # π/3
theta_left_mean = theta_rad[is_left].mean()
theta_right_mean = theta_rad[is_right].mean()
print(f"\nMean θ (left):  {np.degrees(theta_left_mean):.4f} deg")
print(f"Mean θ (right): {np.degrees(theta_right_mean):.4f} deg")
print(f"Expected symmetric positions: 0 deg and {np.degrees(theta_range):.4f} deg")
print(f"Difference from expected: {np.degrees(abs(theta_right_mean - theta_range)):.4f} deg")
