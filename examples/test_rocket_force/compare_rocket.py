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
import glob
import os
import sys

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def read_surf(filename):
    """Read a SPARTA 2D surface file. Returns (R, Z) arrays for plotting."""
    with open(filename, 'r') as f:
        text = f.read()

    # Find Points and Lines sections
    pts = {}
    in_points = False
    in_lines = False
    edges = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == 'Points':
            in_points, in_lines = True, False
            continue
        if stripped == 'Lines':
            in_points, in_lines = False, True
            continue
        if not stripped:
            continue
        parts = stripped.split()
        if in_points and len(parts) >= 3:
            try:
                pid = int(parts[0])
                pts[pid] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue
        elif in_lines and len(parts) >= 3:
            try:
                edges.append((int(parts[1]), int(parts[2])))
            except ValueError:
                continue

    # Chain edges into ordered point list
    ordered = [edges[0][0], edges[0][1]]
    used = {0}
    for _ in range(len(edges) - 1):
        tail = ordered[-1]
        for j, (a, b) in enumerate(edges):
            if j in used:
                continue
            if a == tail:
                ordered.append(b)
                used.add(j)
                break
            elif b == tail:
                ordered.append(a)
                used.add(j)
                break
    r = np.array([pts[i][0] for i in ordered])
    z = np.array([pts[i][1] for i in ordered])
    return r, z


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

common_t_end = min(d['time'][-1] for d in data.values())

# ======================================================================
# Plot — publication-quality 3-panel figure
# ======================================================================
if plt is not None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 1.8,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.top': True,
        'ytick.right': True,
    })

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    linestyles = ['-', '--', ':']

    # (a) Trajectory (R, Z) with wall and separatrix
    ax = axes[0]

    # Load and plot geometry
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wall_file = os.path.join(script_dir, 'wall.surf')
    core_file = os.path.join(script_dir, 'core.surf')
    if os.path.exists(wall_file):
        rw, zw = read_surf(wall_file)
        ax.plot(rw, zw, color='k', lw=2.0, label='Wall', zorder=1)
    if os.path.exists(core_file):
        rc, zc = read_surf(core_file)
        ax.fill(rc, zc, color='lightyellow', edgecolor='forestgreen',
                lw=1.5, label='Separatrix', zorder=1)

    for eta, c, lab, ls in zip(eta_values, colors, labels, linestyles):
        if eta in data:
            d = data[eta]
            ax.plot(d['x'], d['y'], color=c, label=lab, linestyle=ls, zorder=3)
    ax.set_xlabel(r'$R$ [m]')
    ax.set_ylabel(r'$Z$ [m]')
    ax.legend(frameon=False, fontsize=9)
    ax.set_aspect('equal')
    ax.text(0.03, 0.95, '(a)', transform=ax.transAxes, fontweight='bold',
            va='top', fontsize=12)

    # (b) Radial velocity vs time
    ax = axes[1]
    for eta, c, lab, ls in zip(eta_values, colors, labels, linestyles):
        if eta in data:
            d = data[eta]
            ax.plot(d['time'] * 1e3, d['vx'], color=c, label=lab, linestyle=ls)
    ax.set_xlabel(r'Time [ms]')
    ax.set_ylabel(r'$v_R$ [m/s]')
    ax.legend(frameon=False)
    ax.text(0.03, 0.95, '(b)', transform=ax.transAxes, fontweight='bold',
            va='top', fontsize=12)

    plt.tight_layout(w_pad=2.5)
    for fmt in ['png', 'pdf']:
        outname = f'rocket_force_comparison.{fmt}'
        fig.savefig(outname, dpi=300, bbox_inches='tight')
    print(f"\nSaved rocket_force_comparison.png and .pdf")
else:
    fig = None
    print("\nmatplotlib not available; skipping plot generation.")

# Print summary statistics
print("\n--- Summary ---")
print(f"  common_t_end = {common_t_end:.4f} s")
for eta in eta_values:
    if eta in data:
        d = data[eta]
        ic = np.searchsorted(d['time'], common_t_end, side='right') - 1
        ic = max(ic, 0)
        dR = d['x'][ic] - d['x'][0]
        vR_common = d['vx'][ic]
        vR_min = np.min(d['vx'])
        positive_radius = d['radius'][d['radius'] > 0.0]
        r0 = positive_radius[0] if len(positive_radius) else d['radius'][0]
        rratio = d['radius'][ic] / r0 if r0 > 0 else float('nan')
        print(f"  eta={eta}: deltaR = {dR:.4f} m, vR(t_common) = {vR_common:.4f} m/s, "
              f"vR_min = {vR_min:.4f} m/s, "
              f"r_final/r0 = {rratio:.4f}")

if plt is not None and "--show" in sys.argv:
    plt.show()
elif plt is not None:
    plt.close(fig)
