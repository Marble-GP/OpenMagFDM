#!/usr/bin/env python3
import numpy as np
import glob
import os

# Read all force files
force_dir = "output_20251115_122217/Forces"
force_files = sorted(glob.glob(f"{force_dir}/step_*.csv"))

steps = []
torques = []
energies = []

for f in force_files:
    step = int(os.path.basename(f).replace("step_", "").replace(".csv", ""))
    
    with open(f, 'r') as file:
        lines = file.readlines()
        header_line = lines[0].strip()
        header = header_line.split(',')
        
        data_line = None
        for line in lines[1:]:
            if not line.strip().startswith('#'):
                data_line = line.strip()
                break
        
        if data_line is None:
            continue
        
        data = data_line.split(',')
        
        try:
            torque_origin_idx = None
            energy_idx = None
            
            for i, col in enumerate(header):
                if 'Torque_Origin' in col:
                    torque_origin_idx = i
                if 'Magnetic_Energy' in col:
                    energy_idx = i
            
            if torque_origin_idx is None or energy_idx is None:
                continue
            
            torque = float(data[torque_origin_idx])
            energy = float(data[energy_idx])
            
            steps.append(step)
            torques.append(torque)
            energies.append(energy)
        except (ValueError, IndexError) as e:
            continue

steps = np.array(steps)
torques = np.array(torques)
energies = np.array(energies)

# Calculate angle per step
theta_range = 1.0471975512  # π/3 rad from config
angle_per_step = theta_range / len(steps)

# Energy derivative
energy_diff = np.diff(energies)
torque_from_energy = -energy_diff / angle_per_step

print("=== Detailed Torque Analysis ===")
print(f"Total steps: {len(steps)}")
print(f"Theta range: {theta_range:.6f} rad = {np.degrees(theta_range):.2f} deg")
print(f"Angle per step: {angle_per_step:.6f} rad = {np.degrees(angle_per_step):.4f} deg")

print(f"\n=== Torque Statistics ===")
print(f"Mean:   {np.mean(torques):.6f} N·m")
print(f"Median: {np.median(torques):.6f} N·m")
print(f"Std:    {np.std(torques):.6f} N·m")
print(f"Min:    {np.min(torques):.6f} N·m (step {steps[np.argmin(torques)]})")
print(f"Max:    {np.max(torques):.6f} N·m (step {steps[np.argmax(torques)]})")

# Check for periodicity: compare first and last values
print(f"\n=== Periodicity Check ===")
print(f"First 5 torques: {torques[:5]}")
print(f"Last 5 torques:  {torques[-5:]}")
print(f"Difference (last - first): {torques[-1] - torques[0]:.6f} N·m")

# Check symmetry around zero
positive_torques = torques[torques > 0]
negative_torques = torques[torques < 0]
print(f"\n=== Symmetry Analysis ===")
print(f"Positive samples: {len(positive_torques)} (mean: {np.mean(positive_torques):.6f} N·m)")
print(f"Negative samples: {len(negative_torques)} (mean: {np.mean(negative_torques):.6f} N·m)")
print(f"Absolute mean (positive): {np.mean(np.abs(positive_torques)):.6f} N·m")
print(f"Absolute mean (negative): {np.mean(np.abs(negative_torques)):.6f} N·m")
print(f"Asymmetry: {np.mean(np.abs(positive_torques)) - np.mean(np.abs(negative_torques)):.6f} N·m")

# Correlation with energy derivative
correlation = np.corrcoef(torques[:-1], torque_from_energy)[0, 1]
print(f"\n=== Energy-Torque Consistency ===")
print(f"Correlation: {correlation:.6f}")
print(f"Mean torque (calculated): {np.mean(torques):.6f} N·m")
print(f"Mean torque (from -dE/dθ): {np.mean(torque_from_energy):.6f} N·m")
print(f"Offset: {np.mean(torques[:-1]) - np.mean(torque_from_energy):.6f} N·m")

# Integral of torque over one period (should be zero for conservative field)
total_work = np.sum(torques[:-1] * angle_per_step)
print(f"\n=== Conservative Field Check ===")
print(f"Work per cycle (∫T·dθ): {total_work:.6f} J/m")
print(f"Energy change (E_end - E_start): {energies[-1] - energies[0]:.6f} J/m")
print(f"Conservation error: {total_work + (energies[-1] - energies[0]):.6f} J/m")
