#!/usr/bin/env python3
"""Check what psi_norm corresponds to a given flux surface polygon.

Reads a SOLPS .equ equilibrium file and a SPARTA 2D surface file,
evaluates psi_norm at each point on the surface.
"""

import numpy as np
import re
import sys


def read_equ(path):
    """Read SOLPS .equ equilibrium file."""
    with open(path, 'r') as f:
        text = f.read()

    # Parse scalars
    jm = int(re.search(r'jm\s*=\s*(\d+)', text).group(1))
    km = int(re.search(r'km\s*=\s*(\d+)', text).group(1))
    psib = float(re.search(r'psib\s*=\s*([^\s;]+)', text).group(1))

    # Parse arrays: find r(1:jm), z(1:km), psi(1:jm,1:km)
    def extract_array(pattern, n):
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"Could not find {pattern}")
        start = m.end()
        # Read n floating point values after the marker
        vals = []
        pos = start
        while len(vals) < n:
            chunk = re.findall(r'[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d+\.\d*|\d+', text[pos:pos+500])
            if not chunk:
                pos += 100
                continue
            vals.extend(float(x) for x in chunk)
            pos += 500
        return np.array(vals[:n])

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
    psi_pos = find_section(text, ['psi(1:jm,1:km)', '((psi(j,k)-psib,j=1,jm),k=1,km)'])

    r = read_floats_from(r_pos, jm)
    z = read_floats_from(z_pos, km)
    psi_data = read_floats_from(psi_pos, jm * km).reshape(km, jm)

    # The .equ file stores (psi - psib), so actual psi = data + psib
    psi = psi_data + psib

    # Find psi at magnetic axis (minimum in central region)
    psi_axis = psi[km//4:3*km//4, jm//4:3*jm//4].min()

    return r, z, psi, psi_axis, psib, jm, km


def read_flux_surface(path):
    """Read R,Z points from SPARTA 2D surface file."""
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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: check_psi_norm.py <equ_file> <flux_surface.surf>")
        sys.exit(1)

    equ_path = sys.argv[1]
    surf_path = sys.argv[2]

    r_grid, z_grid, psi, psi_axis, psib, jm, km = read_equ(equ_path)

    print(f"Equilibrium: {equ_path}")
    print(f"  jm={jm}, km={km}")
    print(f"  psi_axis (min) = {psi_axis:.6e}")
    print(f"  psib (boundary) = {psib:.6e}")
    print(f"  R range: [{r_grid[0]:.4f}, {r_grid[-1]:.4f}]")
    print(f"  Z range: [{z_grid[0]:.4f}, {z_grid[-1]:.4f}]")
    print()

    R_pts, Z_pts = read_flux_surface(surf_path)

    # Interpolate psi at surface points
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((z_grid, r_grid), psi,
                                     bounds_error=False, fill_value=None)
    psi_vals = interp(np.column_stack([Z_pts, R_pts]))

    # Normalized psi: 0 at axis, 1 at boundary
    dpsi = psib - psi_axis
    if abs(dpsi) < 1e-30:
        print("ERROR: psib == psi_axis, cannot normalize")
        sys.exit(1)
    psi_n = (psi_vals - psi_axis) / dpsi

    print(f"Flux surface: {surf_path} ({len(R_pts)} points)")
    print(f"  psi_norm: min={psi_n.min():.6f}  max={psi_n.max():.6f}  "
          f"mean={psi_n.mean():.6f}  std={psi_n.std():.6f}")
    print()
    print(f"  --> Use  psi_norm {psi_n.mean():.4f}  as your threshold")
    print(f"      fix fcore reflect/psi 1 geqdsk <file> psi_norm {psi_n.mean():.4f}")
    print()
    print("Per-point values:")
    for i, (r, z, pn) in enumerate(zip(R_pts, Z_pts, psi_n)):
        print(f"  {i+1:3d}  R={r:.6f}  Z={z:.6f}  psi_norm={pn:.6f}")
