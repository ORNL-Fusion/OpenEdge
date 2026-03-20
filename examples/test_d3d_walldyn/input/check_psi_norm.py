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

    # Better approach: split after markers
    sections = {}
    for marker in ['r(1:jm)', 'z(1:km)', 'psi(1:jm,1:km)']:
        idx = text.find(marker)
        if idx < 0:
            # Try alternate format
            alt = marker.replace('jm', str(jm)).replace('km', str(km))
            idx = text.find(alt)
        if idx >= 0:
            sections[marker] = idx + len(marker)

    def read_floats_from(pos, n):
        vals = []
        remaining = text[pos:]
        # Find all numbers
        for m in re.finditer(r'[+-]?\d+\.\d*[eE][+-]?\d+|[+-]?\d+\.\d+', remaining):
            vals.append(float(m.group()))
            if len(vals) >= n:
                break
        return np.array(vals[:n])

    r = read_floats_from(sections.get('r(1:jm)', 0), jm)
    z = read_floats_from(sections.get('z(1:km)', 0), km)
    psi = read_floats_from(sections.get('psi(1:jm,1:km)', 0), jm * km)
    psi = psi.reshape(km, jm)

    # Find psi at magnetic axis (minimum of |psi|)
    psi_axis = psi[psi.shape[0]//4:3*psi.shape[0]//4,
                   psi.shape[1]//4:3*psi.shape[1]//4].min()

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
