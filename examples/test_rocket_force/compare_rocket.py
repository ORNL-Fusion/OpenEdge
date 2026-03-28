#!/usr/bin/env python3
"""
Compare rocket force test results for eta = 0.0, 0.5, 1.0.

Reads SPARTA particle dump files and plots:
  1. Trajectories in (R, Z) plane — should show increasing R-deflection with eta
  2. R-velocity vs time — should show growing negative vR with eta
  3. Radius vs time — should be identical (same evaporation, rocket doesn't affect mass loss)

Usage:
    python3 compare_rocket.py
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys


def read_particle_dump(pattern):
    """Read all frames from a SPARTA particle dump file sequence.

    Returns dict with arrays: time, x, y, vx, vy, mass, temp, radius
    """
    files = sorted(glob.glob(pattern + '.*'))
    if not files:
        # Try single file
        if os.path.exists(pattern):
            files = [pattern]
        else:
            print(f"  No files matching {pattern}")
            return None

    times, xs, ys, vxs, vys, masses, temps, radii = [], [], [], [], [], [], [], []

    for fpath in files:
        with open(fpath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('ITEM: TIMESTEP'):
                step = int(lines[i+1].strip())
                i += 2
                continue
            if line.startswith('ITEM: NUMBER'):
                natoms = int(lines[i+1].strip())
                i += 2
                continue
            if line.startswith('ITEM: ATOMS') or line.startswith('ITEM: PARTICLES'):
                # Parse header to find column indices
                cols = line.split()[2:]  # skip "ITEM: ATOMS"
                col_map = {c: j for j, c in enumerate(cols)}

                for j in range(natoms):
                    parts = lines[i+1+j].split()
                    if len(parts) < len(cols):
                        continue
                    times.append(step * 1e-5)  # dt = 1e-5
                    xs.append(float(parts[col_map.get('x', 2)]))
                    ys.append(float(parts[col_map.get('y', 3)]))
                    vxs.append(float(parts[col_map.get('vx', 5)]))
                    vys.append(float(parts[col_map.get('vy', 6)]))
                    masses.append(float(parts[col_map.get('v_pmass', 8)]))
                    temps.append(float(parts[col_map.get('temp', 9)]))
                    radii.append(float(parts[col_map.get('radius', 10)]))
                i += 1 + natoms
                continue
            i += 1

    if not times:
        return None

    return {
        'time': np.array(times),
        'x': np.array(xs),
        'y': np.array(ys),
        'vx': np.array(vxs),
        'vy': np.array(vys),
        'mass': np.array(masses),
        'temp': np.array(temps),
        'radius': np.array(radii),
    }


# ======================================================================
# Load data
# ======================================================================
eta_values = [0.0, 0.5, 1.0]
colors = ['blue', 'orange', 'red']
labels = [r'$\eta=0$', r'$\eta=0.5$', r'$\eta=1.0$']

data = {}
for eta in eta_values:
    eta_str = f"{eta:.1f}".replace(".", "p")
    pattern = f"traj_eta_{eta_str}"
    d = read_particle_dump(pattern)
    if d is not None:
        data[eta] = d
        print(f"  eta={eta}: {len(d['time'])} frames, "
              f"t=[{d['time'].min():.3f}, {d['time'].max():.3f}] s")
    else:
        print(f"  eta={eta}: NO DATA")

if not data:
    print("No data found. Run the simulations first.")
    sys.exit(1)

# ======================================================================
# Plot
# ======================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Trajectories (R, Z)
ax = axes[0, 0]
for eta, c, lab in zip(eta_values, colors, labels):
    if eta in data:
        d = data[eta]
        ax.plot(d['x'], d['y'], color=c, label=lab, linewidth=1.5)
ax.set_xlabel('R [m]')
ax.set_ylabel('Z [m]')
ax.set_title('Droplet trajectory')
ax.legend()
ax.set_aspect('equal')

# 2. R position vs time
ax = axes[0, 1]
for eta, c, lab in zip(eta_values, colors, labels):
    if eta in data:
        d = data[eta]
        ax.plot(d['time'], d['x'], color=c, label=lab, linewidth=1.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel('R [m]')
ax.set_title('R position vs time')
ax.legend()

# 3. vR vs time
ax = axes[1, 0]
for eta, c, lab in zip(eta_values, colors, labels):
    if eta in data:
        d = data[eta]
        ax.plot(d['time'], d['vx'], color=c, label=lab, linewidth=1.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$v_R$ [m/s]')
ax.set_title('Radial velocity (rocket force effect)')
ax.legend()

# 4. Radius vs time (should overlap)
ax = axes[1, 1]
for eta, c, lab in zip(eta_values, colors, labels):
    if eta in data:
        d = data[eta]
        r_norm = d['radius'] / d['radius'][0] if d['radius'][0] > 0 else d['radius']
        ax.plot(d['time'], r_norm, color=c, label=lab, linewidth=1.5)
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$r_d / r_{d,0}$')
ax.set_title('Normalized radius (should overlap)')
ax.legend()

plt.suptitle('Rocket Force Validation: Single Droplet', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rocket_force_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved rocket_force_comparison.png")

# Print summary statistics
print("\n--- Summary ---")
for eta in eta_values:
    if eta in data:
        d = data[eta]
        dR = d['x'][-1] - d['x'][0]
        vR_final = d['vx'][-1]
        print(f"  eta={eta}: deltaR = {dR:.4f} m, vR_final = {vR_final:.4f} m/s, "
              f"r_final/r0 = {d['radius'][-1]/d['radius'][0]:.4f}")

plt.show()
