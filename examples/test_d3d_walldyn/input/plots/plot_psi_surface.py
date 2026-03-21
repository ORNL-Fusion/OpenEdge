#!/usr/bin/env python3
"""Plot psi contours with the flux surface and outer wall overlaid."""

import numpy as np
import re
import sys
import matplotlib.pyplot as plt


def read_equ(path):
    with open(path, 'r') as f:
        text = f.read()

    jm = int(re.search(r'jm\s*=\s*(\d+)', text).group(1))
    km = int(re.search(r'km\s*=\s*(\d+)', text).group(1))
    psib = float(re.search(r'psib\s*=\s*([^\s;]+)', text).group(1))

    def find_section(text, candidates):
        for c in candidates:
            idx = text.find(c)
            if idx >= 0:
                return idx + len(c)
        raise KeyError(f"Could not find any of {candidates}")

    def read_floats_from(pos, n):
        vals = []
        remaining = text[pos:]
        for m in re.finditer(r'[+-]?\d+\.\d*[eE][+-]?\d+|[+-]?\d+\.\d+', remaining):
            vals.append(float(m.group()))
            if len(vals) >= n:
                break
        return np.array(vals[:n])

    r_pos = find_section(text, ['r(1:jm);', 'r(1:jm)'])
    z_pos = find_section(text, ['z(1:km);', 'z(1:km)'])
    # psi section: could be 'psi(1:jm,1:km)' or '((psi(j,k)-psib,...'
    psi_pos = find_section(text, ['psi(1:jm,1:km)', '((psi(j,k)-psib,j=1,jm),k=1,km)'])

    r = read_floats_from(r_pos, jm)
    z = read_floats_from(z_pos, km)
    psi_data = read_floats_from(psi_pos, jm * km).reshape(km, jm)

    # The .equ file stores (psi - psib), so actual psi = data + psib
    psi = psi_data + psib

    psi_axis = psi[km//4:3*km//4, jm//4:3*jm//4].min()
    return r, z, psi, psi_axis, psib


def read_flux_surface(path):
    R, Z = [], []
    section = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s == "Points":
                section = "points"
                continue
            if s == "Lines":
                break
            if section == "points":
                cols = s.split()
                if len(cols) >= 3:
                    R.append(float(cols[1]))
                    Z.append(float(cols[2]))
    return np.array(R), np.array(Z)


def read_wall_2d(path):
    """Read outer wall from SPARTA 2D surface or mesh.extra-derived file."""
    R, Z = [], []
    section = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s == "Points":
                section = "points"
                continue
            if s == "Lines" or s == "Triangles":
                break
            if section == "points":
                cols = s.split()
                if len(cols) >= 3:
                    R.append(float(cols[1]))
                    Z.append(float(cols[2]))
    return np.array(R), np.array(Z)


def psi_norm_interp(r_grid, z_grid, psi_norm, R_pts, Z_pts):
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((z_grid, r_grid), psi_norm,
                                     bounds_error=False, fill_value=None)
    return interp(np.column_stack([Z_pts, R_pts]))


if __name__ == "__main__":
    from pathlib import Path

    equ_path = sys.argv[1] if len(sys.argv) > 1 else "input/data/g174310.03500_153.X4.equ"
    surf_path = sys.argv[2] if len(sys.argv) > 2 else "input/geometry/flux_surface_core.surf"

    r_grid, z_grid, psi, psi_axis, psib = read_equ(equ_path)

    # Normalized psi
    dpsi = psib - psi_axis
    R2d, Z2d = np.meshgrid(r_grid, z_grid)
    psi_norm = (psi - psi_axis) / dpsi

    # Flux surface
    R_fs, Z_fs = read_flux_surface(surf_path)
    psi_n_fs = psi_norm_interp(r_grid, z_grid, psi_norm, R_fs, Z_fs)

    # Try to read wall
    wall_R, wall_Z = None, None
    for wf in ["d3d_refined_wall_rz.surf", "d3d_wedge_wall_rz.surf"]:
        wpath = Path(surf_path).parent / wf
        if wpath.exists():
            wall_R, wall_Z = read_wall_2d(str(wpath))
            break

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    # Psi filled contours
    levels = np.linspace(0, 1.2, 25)
    cs = ax.contourf(R2d, Z2d, psi_norm, levels=levels, cmap='RdYlBu_r', alpha=0.7)
    plt.colorbar(cs, ax=ax, label=r'$\psi_N$')

    # Psi contour lines
    cl = ax.contour(R2d, Z2d, psi_norm,
                    levels=[0.2, 0.4, 0.6, 0.8, 0.926, 1.0],
                    colors='gray', linewidths=0.8, linestyles='--')
    ax.clabel(cl, inline=True, fontsize=8, fmt='%.2f')

    # Separatrix (psi_norm = 1)
    ax.contour(R2d, Z2d, psi_norm, levels=[1.0], colors='black', linewidths=2)

    # Core flux surface
    R_closed = np.append(R_fs, R_fs[0])
    Z_closed = np.append(Z_fs, Z_fs[0])
    mean_pn = np.mean(psi_n_fs)
    ax.plot(R_closed, Z_closed, 'r-', linewidth=2.5,
            label=f'Core boundary (mean $\\psi_N$={mean_pn:.3f})')
    ax.plot(R_fs, Z_fs, 'r.', markersize=4)

    # Outer wall
    if wall_R is not None:
        wR_closed = np.append(wall_R, wall_R[0])
        wZ_closed = np.append(wall_Z, wall_Z[0])
        ax.plot(wR_closed, wZ_closed, 'k-', linewidth=2, label='Outer wall')

    ax.set_xlabel('R [m]', fontsize=14)
    ax.set_ylabel('Z [m]', fontsize=14)
    ax.set_title('DIII-D #174310: $\\psi_N$ with core boundary', fontsize=14)
    ax.set_aspect('equal')
    ax.set_xlim([0.9, 2.5])
    ax.set_ylim([-1.5, 1.5])
    ax.legend(loc='upper right', fontsize=10)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig('psi_norm_with_surface.png', dpi=150)
    print("Saved: psi_norm_with_surface.png")
    plt.show()
