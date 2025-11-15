#!/usr/bin/env python3
import numpy as np
import glob
import os

# Read all force files
force_dir = "output_20251115_103835/Forces"
force_files = sorted(glob.glob(f"{force_dir}/step_*.csv"))

# Extract data
steps = []
torques_origin = []
torques_center = []
energies = []

for f in force_files:
    step = int(os.path.basename(f).replace("step_", "").replace(".csv", ""))

    # Read CSV manually
    with open(f, 'r') as file:
        lines = file.readlines()

        # Find header line (first line)
        header_line = lines[0].strip()
        header = header_line.split(',')

        # Find data line (skip comment lines starting with #)
        data_line = None
        for line in lines[1:]:
            if not line.strip().startswith('#'):
                data_line = line.strip()
                break

        if data_line is None:
            print(f"Warning: No data found in {f}")
            continue

        data = data_line.split(',')

        # Find column indices (exact match with units)
        try:
            # Look for columns with units in brackets
            torque_origin_idx = None
            torque_center_idx = None
            energy_idx = None

            for i, col in enumerate(header):
                if 'Torque_Origin' in col:
                    torque_origin_idx = i
                if 'Torque_Center' in col:
                    torque_center_idx = i
                if 'Magnetic_Energy' in col:
                    energy_idx = i

            if torque_origin_idx is None or torque_center_idx is None or energy_idx is None:
                print(f"Warning: Could not find required columns in {f}")
                print(f"  Header: {header}")
                continue

            torque_origin = float(data[torque_origin_idx])
            torque_center = float(data[torque_center_idx])
            energy = float(data[energy_idx])

            steps.append(step)
            torques_origin.append(torque_origin)
            torques_center.append(torque_center)
            energies.append(energy)
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse {f}: {e}")
            continue

if len(steps) == 0:
    print("ERROR: No data could be read from force files!")
    exit(1)

steps = np.array(steps)
torques_origin = np.array(torques_origin)
torques_center = np.array(torques_center)
energies = np.array(energies)

# Calculate energy derivative (approximate torque)
energy_diff = np.diff(energies)
angle_per_step = 2 * np.pi / len(steps)
torque_from_energy = -energy_diff / angle_per_step  # T = -dE/dθ

# Print statistics
print("=== Torque Analysis ===")
print(f"Total steps: {len(steps)}")
print(f"\nTorque (origin) statistics:")
print(f"  Mean: {np.mean(torques_origin):.6f} N·m")
print(f"  Std:  {np.std(torques_origin):.6f} N·m")
print(f"  Min:  {np.min(torques_origin):.6f} N·m")
print(f"  Max:  {np.max(torques_origin):.6f} N·m")
print(f"  P-P:  {np.max(torques_origin) - np.min(torques_origin):.6f} N·m")

print(f"\nTorque (center) statistics:")
print(f"  Mean: {np.mean(torques_center):.6f} N·m")
print(f"  Std:  {np.std(torques_center):.6f} N·m")
print(f"  Min:  {np.min(torques_center):.6f} N·m")
print(f"  Max:  {np.max(torques_center):.6f} N·m")
print(f"  P-P:  {np.max(torques_center) - np.min(torques_center):.6f} N·m")

print(f"\nMagnetic Energy statistics:")
print(f"  Mean: {np.mean(energies):.6f} J/m")
print(f"  Std:  {np.std(energies):.6f} J/m")
print(f"  Min:  {np.min(energies):.6f} J/m")
print(f"  Max:  {np.max(energies):.6f} J/m")
print(f"  P-P:  {np.max(energies) - np.min(energies):.6f} J/m")
print(f"  Variation: {100.0 * np.std(energies) / np.mean(energies):.2f}%")

# Find step 93 area
idx_93 = np.where(steps == 93)[0]
if len(idx_93) > 0:
    i = idx_93[0]
    print(f"\n=== Around Step 93 ===")
    for j in range(max(0, i-5), min(len(steps), i+6)):
        e_change = ""
        t_energy = ""
        if j > 0:
            de = energies[j] - energies[j-1]
            t_from_e = -de / angle_per_step
            e_change = f"ΔE={de:+8.4f}"
            t_energy = f"T(from E)={t_from_e:+9.4f}"
        print(f"Step {steps[j]:3d}: Torque={torques_origin[j]:+10.4f} N·m, Energy={energies[j]:10.6f} J/m  {e_change:20s} {t_energy}")

# Check for discontinuities
torque_diff = np.diff(torques_origin)
large_jumps = np.where(np.abs(torque_diff) > 10.0)[0]  # More than 10 N·m jump
if len(large_jumps) > 0:
    print(f"\n=== Large Torque Jumps (>10 N·m) ===")
    for idx in large_jumps[:15]:  # Show first 15
        print(f"Between step {steps[idx]:3d} and {steps[idx+1]:3d}: ΔT = {torque_diff[idx]:+10.4f} N·m")

# Calculate correlation between torque and energy derivative
if len(torque_from_energy) > 0:
    # Align arrays (energy derivative is one element shorter)
    correlation = np.corrcoef(torques_origin[:-1], torque_from_energy)[0, 1]
    print(f"\n=== Torque vs Energy Derivative ===")
    print(f"Correlation between calculated torque and -dE/dθ: {correlation:.4f}")

    # RMS difference
    rms_diff = np.sqrt(np.mean((torques_origin[:-1] - torque_from_energy)**2))
    print(f"RMS difference: {rms_diff:.4f} N·m")

    # Mean absolute difference
    mean_abs_diff = np.mean(np.abs(torques_origin[:-1] - torque_from_energy))
    print(f"Mean absolute difference: {mean_abs_diff:.4f} N·m")

print("\n=== Writing data to CSV for plotting ===")
with open("torque_timeseries.csv", "w") as f:
    f.write("step,torque_origin,torque_center,energy,torque_from_energy\n")
    for i in range(len(steps)):
        t_from_e = torque_from_energy[i] if i < len(torque_from_energy) else 0.0
        f.write(f"{steps[i]},{torques_origin[i]},{torques_center[i]},{energies[i]},{t_from_e}\n")
print("Data written to torque_timeseries.csv")
