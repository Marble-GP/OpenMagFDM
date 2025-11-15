#!/usr/bin/env python3
import numpy as np

# Read stress vector data
data = []
with open("output_20251115_122217/StressVectors/step_0001.csv", 'r') as f:
    lines = f.readlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if not line.startswith('#'):
            header_idx = i
            break
    
    header = lines[header_idx].strip().split(',')
    for line in lines[header_idx+1:]:
        if line.startswith('#'):
            continue
        values = [float(x) if x.replace('.','').replace('-','').replace('e','').replace('+','').isdigit() 
                  else x for x in line.strip().split(',')]
        data.append(values)

data = np.array(data, dtype=object)

# Extract columns
i_pixel = data[:, 0].astype(int)
j_pixel = data[:, 1].astype(int)
x = data[:, 2].astype(float)
y = data[:, 3].astype(float)
fx = data[:, 4].astype(float)
fy = data[:, 5].astype(float)
ds = data[:, 6].astype(float)
nx = data[:, 7].astype(float)
ny = data[:, 8].astype(float)

# Extract boundary points only (ds > 0)
mask = ds > 0
x_b = x[mask]
y_b = y[mask]
fx_b = fx[mask]
fy_b = fy[mask]
ds_b = ds[mask]
nx_b = nx[mask]
ny_b = ny[mask]

print(f"=== Boundary Analysis ===")
print(f"Total boundary pixels: {len(x_b)}")

# Calculate r and theta
r_b = np.sqrt(x_b**2 + y_b**2)
theta_b = np.arctan2(y_b, x_b)

# Normal vector in polar coordinates
n_r = nx_b * np.cos(theta_b) + ny_b * np.sin(theta_b)
n_theta = -nx_b * np.sin(theta_b) + ny_b * np.cos(theta_b)

# Classify boundary types
# Circular arc: |n_r| > 0.8
# Radial line: |n_theta| > 0.8
is_circular = np.abs(n_r) > 0.8
is_radial = np.abs(n_theta) > 0.8
is_mixed = ~(is_circular | is_radial)

print(f"\n=== Boundary Type Distribution ===")
print(f"Circular arc (r=const): {is_circular.sum()} pixels")
print(f"Radial line (θ=const):  {is_radial.sum()} pixels")
print(f"Mixed/corner:           {is_mixed.sum()} pixels")

# Circular arc analysis
if is_circular.sum() > 0:
    print(f"\n=== Circular Arc Boundaries ===")
    r_circ = r_b[is_circular]
    ds_circ = ds_b[is_circular]
    print(f"r range: [{r_circ.min():.6f}, {r_circ.max():.6f}] m")
    print(f"ds range: [{ds_circ.min():.6f}, {ds_circ.max():.6f}] m")
    print(f"Mean ds: {ds_circ.mean():.6f} m")
    
    # Expected ds = r * dtheta
    dtheta = 1.0471975512 / 500
    ds_expected = r_circ * dtheta
    ds_error = ds_circ - ds_expected
    print(f"Mean ds error: {ds_error.mean():.6e} m ({100*ds_error.mean()/ds_circ.mean():.2f}%)")
    print(f"RMS ds error: {np.sqrt((ds_error**2).mean()):.6e} m")

# Radial line analysis
if is_radial.sum() > 0:
    print(f"\n=== Radial Line Boundaries ===")
    ds_rad = ds_b[is_radial]
    print(f"ds range: [{ds_rad.min():.6f}, {ds_rad.max():.6f}] m")
    print(f"Mean ds: {ds_rad.mean():.6f} m")
    
    # Expected ds = dr
    dr = (0.3 - 0.2) / (500 - 1)
    print(f"Expected dr: {dr:.6f} m")
    print(f"ds/dr ratio: {ds_rad.mean()/dr:.6f}")

# Torque contribution by type
print(f"\n=== Torque Contribution by Boundary Type ===")
dtorque = (x_b * fy_b - y_b * fx_b) * ds_b

torque_circ = dtorque[is_circular].sum() if is_circular.sum() > 0 else 0
torque_rad = dtorque[is_radial].sum() if is_radial.sum() > 0 else 0
torque_mix = dtorque[is_mixed].sum() if is_mixed.sum() > 0 else 0
torque_total = dtorque.sum()

print(f"Circular arc: {is_circular.sum():4d} pixels, torque = {torque_circ:+.6f} N·m ({100*torque_circ/torque_total:+.1f}%)")
print(f"Radial line:  {is_radial.sum():4d} pixels, torque = {torque_rad:+.6f} N·m ({100*torque_rad/torque_total:+.1f}%)")
print(f"Mixed/corner: {is_mixed.sum():4d} pixels, torque = {torque_mix:+.6f} N·m ({100*torque_mix/torque_total:+.1f}%)")
print(f"Total:        {len(x_b):4d} pixels, torque = {torque_total:+.6f} N·m")
