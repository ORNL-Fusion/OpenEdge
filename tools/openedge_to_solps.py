#!/usr/bin/env python3
"""
openedge_to_solps.py
--------------------
Convert OpenEdge droplet evaporation output (mass_loss.txt) to a SOLPS-ITER
particle source file (source2d.00001).

USAGE
-----
    python openedge_to_solps.py

Edit the USER SETTINGS block below before running.

PIPELINE
--------
  1. Read OpenEdge grid dump (mass_loss.txt)
       → time-mean Li atoms lost per SPARTA cell per step [atoms/cell/step]
  2. Convert to volumetric source rate [atoms/m³/s]
       S(R,Z) = dn_per_step / (dt * vol_OE_3D(R))
       where vol_OE_3D = 2π * R * dR_OE * dZ_OE  (toroidal cell volume)
  3. Read SOLPS geometry from b2fgmtry (no quixote needed)
       → cell-centre (R,Z), cell volume vol(iy,ix)  shape (ny+2, nx+2)
  4. Interpolate S onto SOLPS grid  (linear inside hull, nearest outside)
  5. Write source2d.NNNNN
       Layout (matches SOLPS Fortran reader):
         for each species is = 0 .. ns-1:        [outer]
           for each poloidal row iy = 0 .. ny+1: [one line per row]
             nx+2 values = S_SOLPS[iy,ix] * vol_SOLPS[iy,ix]  [atoms/s]
       Only Li0 (species index 13, 0-based) is nonzero; all others = 0.

BUGS FIXED VS. PREVIOUS SCRIPTS (create_source.py / get_source.py)
-------------------------------------------------------------------
  Bug 1  WRONG WRITE LOOP — previous code wrote by radial column:
             for k in range(nx): write arr[:,k]  (ny values/line)
         Correct is by poloidal row:
             for iy in range(ny+2): write arr[iy,:]  (nx+2 values/line)
         Both total to the same number of values (111 112) so SOLPS
         would read them without error, but the (R,Z) → cell mapping
         is completely wrong (transposed + scrambled).
  Bug 2  WRONG SOURCE QUANTITY — previous code used number density [m⁻³]
         from an OpenEdge diagnostic (column 9 of droplet_vapor_diag).
         SOLPS expects an atom SOURCE RATE [atoms/m³/s].  The missing
         factor is 1/dt, and the quantity should come from the per-cell
         mass-change accumulator in fix_evaporation (mass_loss.txt).
  Bug 3  TOROIDAL GEOMETRY MISSING — previous code used the raw SPARTA
         2-D cell area (dR*dZ) as the 3-D volume.  For a tokamak the
         correct 3-D volume of a cell at major radius R is 2π*R*dR*dZ.
         This factor can exceed 10× over the domain and is R-dependent.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import os, sys

# ═══════════════════════════════════════════════════════════════
#  USER SETTINGS
# ═══════════════════════════════════════════════════════════════

# Path to OpenEdge mass_loss.txt (output of 'dump d_ml grid …' in SPARTA)
OE_MASS_LOSS = (
    "/Users/42d/OpenedgeGPU/examples/test_droplet/mass_loss.txt"
)

# SOLPS case folder (must contain b2fgmtry)
SOLPS_DIR = (
    "/Users/42d/ORNL Dropbox/Abdou Diaw/addLi/"
    "fnacore=6.00e22_pheat=90.00MW_cont_dt=1e-6_te_up"
)

# Output file name (written into SOLPS_DIR)
OUT_FILE = "source2d.00001"

# SPARTA timestep [s]  (variable dt in in.droplet_emission)
DT = 1.0e-5

# SPARTA simulation box extent and grid resolution
#   create_box  2.0 6.0  -4.0 3.8  -0.05 0.05
#   create_grid 500 500  1
OE_R_MIN, OE_R_MAX = 2.0, 6.0
OE_Z_MIN, OE_Z_MAX = -4.0, 3.8
OE_NR,    OE_NZ    = 500, 500
OE_Z_THICKNESS     = 0.1          # z-box depth [m]  (0.05 - (-0.05))

# SOLPS grid dimensions (from b2fstate: nx=170, ny=36, ns=17)
NX, NY, NS = 170, 36, 17

# Li0 species index (0-based) — confirmed from b2mn.prt species table
LI0_IDX = 13   # D0=0..D+1=1, Ne0=2..Ne+10=12, Li0=13, Li+1..Li+3=14-16

# ═══════════════════════════════════════════════════════════════
#  STEP 1 — Read OpenEdge mass_loss.txt
# ═══════════════════════════════════════════════════════════════

def read_mass_loss(path):
    """
    Parse the SPARTA grid dump written by:
        dump d_ml grid all ${Ndump} mass_loss.txt id xc yc \\
            f_fml_I[1] f_fml_I[2] f_fml_O[1] f_fml_O[2]

    Returns arrays (R, Z, dn_inner, dn_outer) in SI units.
    Columns (1-based SPARTA numbering):
        1  id
        2  xc   = R [m]
        3  yc   = Z [m]
        4  f_fml_I[1] = time-mean dm_kg/step (inner Li)
        5  f_fml_I[2] = time-mean dn_atoms/step (inner Li)
        6  f_fml_O[1] = time-mean dm_kg/step (outer Li)
        7  f_fml_O[2] = time-mean dn_atoms/step (outer Li)
    """
    # SPARTA dump files have a header block before the data; skip '#' lines
    data = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("ITEM"):
                continue
            # also skip lines that are not numeric (SPARTA header tokens)
            try:
                vals = [float(v) for v in line.split()]
                if len(vals) >= 7:
                    data.append(vals[:7])
            except ValueError:
                continue
    data = np.array(data)
    if data.size == 0:
        raise RuntimeError(f"No data rows found in {path}")

    R        = data[:, 1]   # xc [m]
    Z        = data[:, 2]   # yc [m]
    dn_inner = data[:, 4]   # atoms/cell/step (inner Li)
    dn_outer = data[:, 6]   # atoms/cell/step (outer Li)

    print(f"[OE] Read {len(R)} cells from {path}")
    print(f"     R: [{R.min():.3f}, {R.max():.3f}] m  "
          f"Z: [{Z.min():.3f}, {Z.max():.3f}] m")
    print(f"     dn_inner max={dn_inner.max():.3e}  dn_outer max={dn_outer.max():.3e}")
    return R, Z, dn_inner, dn_outer


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — Compute Li source rate density [atoms/m³/s]
# ═══════════════════════════════════════════════════════════════

def compute_source_rate(R_oe, Z_oe, dn_inner, dn_outer, dt,
                        dR_oe, dZ_oe):
    """
    S [atoms/m³/s] = dn_total_per_step / (dt * vol_3D)

    vol_3D uses the TOROIDAL cell volume:
        vol_3D = 2π * R * dR_oe * dZ_oe   [m³]

    This correctly accounts for the R-dependent toroidal circumference.
    The SPARTA simulation represents the full torus (one poloidal plane,
    toroidal symmetry assumed).
    """
    dn_total = dn_inner + dn_outer          # atoms/cell/step
    vol_3D   = 2.0 * np.pi * R_oe * dR_oe * dZ_oe   # [m³]
    S        = dn_total / (dt * vol_3D)    # atoms/m³/s
    S        = np.where(np.isfinite(S) & (S >= 0.0), S, 0.0)

    total_rate = float((dn_total / dt).sum())
    print(f"[OE] Total Li atom rate = {total_rate:.3e} atoms/s")
    print(f"     Peak source density = {S.max():.3e} atoms/m³/s")
    return S


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — Read SOLPS geometry from b2fgmtry
# ═══════════════════════════════════════════════════════════════

def parse_b2fgmtry(path, nx, ny):
    """
    Read the plain-text b2fgmtry file (SOLPS-ITER geometry).

    Returns:
        R_s  (ny+2, nx+2)  cell-centre major radius [m]
        Z_s  (ny+2, nx+2)  cell-centre height [m]
        vol  (ny+2, nx+2)  cell volume [m³]

    b2fgmtry stores arrays in Fortran column-major order.
    Relevant datasets and their Fortran shapes:
        crx   real  4*(nx+2)*(ny+2)  corner R  → Fortran(4, -1:nx, -1:ny)
        cry   real  4*(nx+2)*(ny+2)  corner Z  → Fortran(4, -1:nx, -1:ny)
        vol   real  (nx+2)*(ny+2)    volumes   → Fortran(-1:nx, -1:ny)
    """
    nxg, nyg = nx + 2, ny + 2        # with guard cells: 172 × 38

    def read_dataset(lines, start, n):
        """Read n floats starting at line `start`."""
        vals = []
        i = start
        while len(vals) < n and i < len(lines):
            vals.extend(float(t) for t in lines[i].split())
            i += 1
        return np.array(vals[:n], dtype=float), i

    with open(path) as fh:
        lines = fh.readlines()

    datasets = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("*cf:"):
            parts = line.split()
            # *cf:  type  count  name
            try:
                count = int(parts[2])
                name  = parts[3]
            except (IndexError, ValueError):
                i += 1
                continue
            vals, i = read_dataset(lines, i + 1, count)
            datasets[name] = vals
        else:
            i += 1

    if "crx" not in datasets:
        raise KeyError("crx not found in b2fgmtry")
    if "cry" not in datasets:
        raise KeyError("cry not found in b2fgmtry")
    if "vol" not in datasets:
        raise KeyError("vol not found in b2fgmtry")

    # crx Fortran storage: crx(1:4, -1:nx, -1:ny)
    # In column-major: fastest index = first = corner (4)
    # Python reshape with order='F' or equivalently reshape then transpose.
    crx = datasets["crx"].reshape(4, nxg, nyg, order="F")  # (4, nx+2, ny+2)
    cry = datasets["cry"].reshape(4, nxg, nyg, order="F")

    # cell centres = mean of 4 corners; shape → (ny+2, nx+2)
    R_s = crx.mean(axis=0).T    # (ny+2, nx+2)
    Z_s = cry.mean(axis=0).T

    # vol Fortran storage: vol(-1:nx, -1:ny) → (nx+2, ny+2)
    vol = datasets["vol"].reshape(nxg, nyg, order="F").T   # (ny+2, nx+2)

    print(f"[SOLPS] b2fgmtry loaded")
    print(f"        R: [{R_s.min():.3f}, {R_s.max():.3f}] m  "
          f"Z: [{Z_s.min():.3f}, {Z_s.max():.3f}] m")
    print(f"        vol: [{vol.min():.3e}, {vol.max():.3e}] m³")
    return R_s, Z_s, vol


# ═══════════════════════════════════════════════════════════════
#  STEP 4 — Interpolate onto SOLPS grid
# ═══════════════════════════════════════════════════════════════

def interpolate_to_solps(R_oe, Z_oe, S_oe, R_s, Z_s):
    """
    Scattered-data interpolation: linear inside the OpenEdge hull,
    nearest-neighbour outside (avoids NaN extrapolation).
    Result is clamped to ≥ 0.
    """
    pts  = np.column_stack([R_oe, Z_oe])
    grid = np.column_stack([R_s.ravel(), Z_s.ravel()])

    S_lin = griddata(pts, S_oe, grid, method="linear")

    # fill NaN outside hull with nearest-neighbour
    if np.isnan(S_lin).any():
        tree = cKDTree(pts)
        _, nn = tree.query(grid[np.isnan(S_lin)], k=1)
        S_lin[np.isnan(S_lin)] = S_oe[nn]

    S_s = S_lin.reshape(R_s.shape)
    S_s = np.where(np.isfinite(S_s) & (S_s >= 0.0), S_s, 0.0)

    print(f"[interp] Peak S on SOLPS grid = {S_s.max():.3e} atoms/m³/s")
    print(f"         Cells with S > 0 : {(S_s > 0).sum()}")
    return S_s


# ═══════════════════════════════════════════════════════════════
#  STEP 5 — Write source2d file
# ═══════════════════════════════════════════════════════════════

def write_source2d(outpath, S_s, vol, ns, li0_idx):
    """
    Write the SOLPS source2d file.

    File layout (confirmed from Jeremy's MATLAB write_source and SOLPS Fortran reader):
        for each species is = 0 .. ns-1:            [outer block]
          for each poloidal row iy = 0 .. ny+1:     [one line per row]
            nx+2 values: factor * S_s[iy,ix] * vol[iy,ix]   [atoms/s]

    Only Li0 (species li0_idx, 0-based) is nonzero.
    All other species → line of zeros.

    The Fortran reader:
        Do j = 0, ns-1
          Do i = -1, ny
            Read(99,*) profile3d(i,j)   ← reads nx+2 values from one line
          End Do
        End Do
    """
    nyg, nxg = vol.shape     # (ny+2, nx+2) = (38, 172)
    src = (S_s * vol).clip(0.0)   # [atoms/s], shape (ny+2, nx+2)

    zero_line = " ".join(["0.0e+00"] * nxg) + "\n"

    with open(outpath, "w") as fh:
        for ispec in range(ns):
            if ispec == li0_idx:
                for iy in range(nyg):
                    fh.write(" ".join(f"{v:.8e}" for v in src[iy, :]) + "\n")
            else:
                for _ in range(nyg):
                    fh.write(zero_line)

    nlines = ns * nyg
    print(f"[write] {outpath}")
    print(f"        {nlines} lines × {nxg} values  (ns={ns}, ny+2={nyg}, nx+2={nxg})")
    total_src = float(src.sum())
    print(f"        Total Li0 source = {total_src:.3e} atoms/s")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # --- Cell sizes in the OpenEdge SPARTA grid ---
    dR_oe = (OE_R_MAX - OE_R_MIN) / OE_NR   # 0.008 m
    dZ_oe = (OE_Z_MAX - OE_Z_MIN) / OE_NZ   # 0.0156 m
    print(f"[OE] SPARTA cell size: dR={dR_oe:.4f} m  dZ={dZ_oe:.4f} m")

    # 1. Read OpenEdge output
    R_oe, Z_oe, dn_inner, dn_outer = read_mass_loss(OE_MASS_LOSS)

    # 2. Source rate density [atoms/m³/s]
    S_oe = compute_source_rate(R_oe, Z_oe, dn_inner, dn_outer,
                               DT, dR_oe, dZ_oe)

    # 3. SOLPS geometry
    b2fgmtry = os.path.join(SOLPS_DIR, "b2fgmtry")
    R_s, Z_s, vol = parse_b2fgmtry(b2fgmtry, NX, NY)

    # 4. Interpolate
    S_s = interpolate_to_solps(R_oe, Z_oe, S_oe, R_s, Z_s)

    # 5. Write
    outpath = os.path.join(SOLPS_DIR, OUT_FILE)
    write_source2d(outpath, S_s, vol, NS, LI0_IDX)

    # Sanity: check total atom budget
    dn_total_step = (dn_inner + dn_outer).sum()
    rate_oe  = dn_total_step / DT
    rate_sol = float((S_s * vol).sum())
    print(f"\n[sanity] OE total rate  : {rate_oe:.3e} atoms/s")
    print(f"         SOLPS file rate : {rate_sol:.3e} atoms/s")
    print(f"         Ratio (should be ~1): {rate_sol/rate_oe:.3f}"
          if rate_oe > 0 else "")
    print("\nDone.")


if __name__ == "__main__":
    main()
