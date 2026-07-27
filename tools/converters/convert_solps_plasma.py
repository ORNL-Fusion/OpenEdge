#!/usr/bin/env python3
"""
SOLPS-ITER -> OpenEdge converter (no quixote dependency).

Reads SOLPS binary text files (b2fgmtry, b2fstate) directly using the
*cf: header format, following Jeremy Lore / Jae-Sun Park's SOLPS routines.

Current EIRENE-mesh mapping model:
  1. Native EIRENE triangles with a valid fort.35 (ix,iy) host-cell map
     receive a direct copy of that B2.5 cell's plasma data.
  2. Non-native EIRENE triangles are left unmapped.

This follows the stricter B2.5/EIRENE interface behavior: triangles
outside the valid fort.35 host-cell map remain on a vacuum floor and are
not filled from neighboring plasma cells.

Outputs:
  plasma.h5  - regular (R,Z) grid with all plasma fields + multi-ion
               species + B-field (br/bt/bz) embedded. compute
               plasma/fields reads everything from this single file.

Usage:
    python convert_solps_plasma.py /path/to/solps_run \\
        --equ-file equilibrium.equ \\
        --plasma-out plasma.h5 \\
        --nr 300 --nz 300 --plot
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from scipy.interpolate import griddata

# SOLPS-ITER fort.31 reader (python port of SOLPS-routines read_ft31.m).
# Lives in the same directory so the converter can hand it a run_path
# and get back the per-B2-cell plasma SOLPS writes out for EIRENE.
try:
    from read_fort31 import read_fort31, Fort31Data
    from _wall_flux import build_wall_flux_solps, write_wall_flux_h5
except ImportError:
    from .read_fort31 import read_fort31, Fort31Data  # type: ignore
    from ._wall_flux import build_wall_flux_solps, write_wall_flux_h5  # type: ignore


# ==========================================================================
# SOLPS binary text file parsers (replaces quixote dependency)
# Based on: /home/cloud/SOLPS-routines/Jae-Sun/Python/b2fextract.py
#           /home/cloud/SOLPS-routines/Python/SOLPSutils.py
# ==========================================================================

def b2f_read_dims(filename: str | Path) -> Tuple[int, int, int]:
    """Read nx, ny, ns from a b2f* file header."""
    with open(filename, "r") as f:
        for line in f:
            tok = line.split()
            if len(tok) >= 4 and tok[0] == "*cf:" and "nx,ny" in tok[3]:
                vals = next(f).split()
                nx = int(vals[0])
                ny = int(vals[1])
                ns = int(vals[2]) if len(vals) > 2 else 1
                return nx, ny, ns
    raise RuntimeError(f"Could not find nx,ny header in {filename}")


def b2f_extract(variable: str, filename: str | Path, reshape: bool = True) -> np.ndarray:
    """
    Extract a variable from a b2f* text file (b2fstate, b2fgmtry, etc.).

    The file uses *cf: headers:
        *cf: {dtype} {nentry} {varname}
    followed by whitespace-separated values.

    Returns:
        If reshape=True: array with shape (nx+2, ny+2, ns) in Fortran order.
        If reshape=False: flat 1D array.
    """
    nx, ny, _ = b2f_read_dims(filename)

    with open(filename, "r") as f:
        lines = f.readlines()

    nentry = None
    datastr = []
    is_body = False

    for i, line in enumerate(lines):
        tok = line.split()
        if not is_body:
            if len(tok) >= 4 and tok[0] == "*cf:" and tok[3] == variable:
                nentry = int(tok[2])
                is_body = True
                continue
        else:
            datastr.extend(tok)
            if len(datastr) >= nentry:
                break

    if nentry is None:
        raise KeyError(f"Variable '{variable}' not found in {filename}")

    data = np.array(datastr[:nentry], dtype=np.float64)

    if reshape:
        ns = max(nentry // ((nx + 2) * (ny + 2)), 1)
        data = data.reshape(nx + 2, ny + 2, ns, order="F")
        if ns == 1:
            data = data[:, :, 0]
    return data


def read_b2fgmtry(filename: str | Path) -> dict:
    """
    Read all variables from b2fgmtry into a dict.
    Based on SOLPSutils.read_b2fgmtry (modified from omfit_solps.py).
    """
    with open(filename, "r") as f:
        raw = f.read()
    raw = raw.replace("\n", " ")
    blocks = raw.split("*c")
    converters = {"int": int, "real": float, "char": str}
    result = {}
    for block in blocks[1:]:
        tok = [t for t in block.split(" ") if t]
        if len(tok) > 4:
            dtype_str = tok[1]
            varname = tok[3]
            conv = converters.get(dtype_str, float)
            try:
                result[varname] = np.array(list(map(conv, tok[4:])))
            except (ValueError, TypeError):
                result[varname] = tok[4:]
        elif len(tok) >= 4:
            result[tok[3]] = None
    return result


# ==========================================================================
# Grid and interpolation utilities
# ==========================================================================

def _cell_centers_from_corners(crx: np.ndarray, cry: np.ndarray,
                                nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cell centers from corner coordinates (mean of 4 corners)."""
    # crx has shape (4*(nx+2)*(ny+2),) from b2fgmtry; reshape to (nx+2, ny+2, 4)
    crx4 = crx.reshape(nx + 2, ny + 2, 4, order="F")
    cry4 = cry.reshape(nx + 2, ny + 2, 4, order="F")
    rc = crx4.mean(axis=2)
    zc = cry4.mean(axis=2)
    return rc, zc


def _cell_polygons_from_corners(crx: np.ndarray, cry: np.ndarray,
                                nx: int, ny: int) -> np.ndarray:
    """
    Build SOLPS quadrilateral polygons from cell corners.

    Jeremy's routines reorder corners as [1, 2, 4, 3] to avoid
    self-intersecting quads in patch plots.
    """
    crx4 = crx.reshape(nx + 2, ny + 2, 4, order="F")
    cry4 = cry.reshape(nx + 2, ny + 2, 4, order="F")
    corner_order = [0, 1, 3, 2]
    polys = np.stack((crx4[:, :, corner_order], cry4[:, :, corner_order]), axis=-1)
    return polys.reshape(-1, 4, 2, order="F")


def _regular_grid(rc, zc, nr, nz, rmin, rmax, zmin, zmax):
    rvals = rc[np.isfinite(rc)]
    zvals = zc[np.isfinite(zc)]
    if rmin is None:
        rmin = float(np.min(rvals))
    if rmax is None:
        rmax = float(np.max(rvals))
    if zmin is None:
        zmin = float(np.min(zvals))
    if zmax is None:
        zmax = float(np.max(zvals))
    zabs = max(abs(zmin), abs(zmax))
    zmin, zmax = -zabs, zabs
    r = np.linspace(rmin, rmax, nr)
    z = np.linspace(zmin, zmax, nz)
    rr, zz = np.meshgrid(r, z)
    return r, z, rr, zz


def _interp_regular_to_points(rgrid, zgrid, values_2d, tgt_rz):
    """Fast 2D interpolation from a regular (R, Z) grid to scattered points.

    Equivalent to _interp_field_points when the source happens to be a
    regular grid, but ~100x faster: uses scipy's RegularGridInterpolator
    (cached binary search) instead of griddata's Delaunay triangulation.

    values_2d shape: (len(zgrid), len(rgrid)) — matching np.meshgrid(r, z)
    output. tgt_rz: (N, 2) array of (R, Z) query points. Returns (N,)
    array; out-of-grid points fall back to nearest-neighbor.
    """
    from scipy.interpolate import RegularGridInterpolator
    pts_zr = np.column_stack([tgt_rz[:, 1], tgt_rz[:, 0]])
    interp_lin = RegularGridInterpolator(
        (zgrid, rgrid), values_2d,
        method='linear', bounds_error=False, fill_value=np.nan)
    out = interp_lin(pts_zr)
    nan_mask = np.isnan(out)
    if nan_mask.any():
        interp_nn = RegularGridInterpolator(
            (zgrid, rgrid), values_2d,
            method='nearest', bounds_error=False, fill_value=None)
        out[nan_mask] = interp_nn(pts_zr[nan_mask])
    return out


def _interp_field_points(src_rz, values, tgt_rz):
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    lin = griddata(src_rz, v, tgt_rz, method="linear")
    nn = griddata(src_rz, v, tgt_rz, method="nearest")
    return np.where(np.isfinite(lin), lin, nn)


def _interp_field(src_rz, values, tgt_rz, nz, nr):
    out = _interp_field_points(src_rz, values, tgt_rz)
    return out.reshape(nz, nr)


# ==========================================================================
# Equilibrium B-field readers (unchanged from previous version)
# ==========================================================================

def _read_equilibrium_bfield(equ_file: Path):
    """Read .equ file and reconstruct (Br, Bt, Bz) from psi.

    Returns a dict with:
      r, z, br, bt, bz         — regular-grid fields for interpolation
      equ_r, equ_z, equ_psi    — raw .equ arrays
      btf, rtf, psib           — toroidal-field params + boundary psi
    Everything in equ_* is what gets embedded into plasma.h5 under
    /equilibrium so downstream consumers don't need the .equ file at
    run time.
    """
    if not equ_file.exists():
        raise FileNotFoundError(f"Equilibrium file not found: {equ_file}")

    jm = km = btf = rtf = None
    psib = 0.0
    read_r = read_z = read_psi = False
    r_vals, z_vals, psi_vals = [], [], []

    with equ_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if len(tok) >= 3 and tok[0] == "jm" and tok[1] == "=":
                jm = int(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "km" and tok[1] == "=":
                km = int(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "btf" and tok[1] == "=":
                btf = float(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "rtf" and tok[1] == "=":
                rtf = float(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "psib" and tok[1] == "=":
                psib = float(tok[2]); continue
            if tok[0] == "r(1:jm);":
                read_r, read_z, read_psi = True, False, False; continue
            if tok[0] == "z(1:km);":
                read_r, read_z, read_psi = False, True, False; continue
            if tok[0].startswith("((psi(j,k)-psib,j=1,jm),k=1,km)"):
                read_r, read_z, read_psi = False, False, True; continue
            if read_r:
                try: r_vals.extend(float(x) for x in tok)
                except ValueError: pass
                continue
            if read_z:
                try: z_vals.extend(float(x) for x in tok)
                except ValueError: pass
                continue
            if read_psi:
                try: psi_vals.extend(float(x) for x in tok)
                except ValueError: pass
                continue

    if any(v is None for v in [jm, km, btf, rtf]):
        raise RuntimeError(f"Missing headers in equilibrium file: {equ_file}")

    r = np.asarray(r_vals[:jm], dtype=np.float64)
    z = np.asarray(z_vals[:km], dtype=np.float64)
    psi = np.asarray(psi_vals[:jm * km], dtype=np.float64).reshape((km, jm))

    grad_z, grad_r = np.gradient(psi, z[1] - z[0], r[1] - r[0])
    rr = np.meshgrid(r, z)[0]
    safe_r = np.where(np.abs(rr) > 1e-12, rr, 1e-12)
    br = -grad_z / safe_r
    bz = grad_r / safe_r
    bt = (btf * rtf) / safe_r
    return {
        "r": r, "z": z, "br": br, "bt": bt, "bz": bz,
        "equ_r": r.copy(), "equ_z": z.copy(), "equ_psi": psi,
        "btf": btf, "rtf": rtf, "psib": psib,
    }


def _read_geqdsk_bfield(gfile: Path):
    """Build (Br, Bt, Bz) from GEQDSK via freeqdsk."""
    if not gfile.exists():
        raise FileNotFoundError(f"GEQDSK file not found: {gfile}")
    try:
        from freeqdsk import geqdsk
    except Exception as exc:
        raise RuntimeError("freeqdsk package required for GEQDSK support") from exc

    with gfile.open("r", encoding="utf-8", errors="ignore") as f:
        data = geqdsk.read(f)

    nx, ny = int(data["nx"]), int(data["ny"])
    rmin = float(data["rleft"])
    rmax = rmin + float(data["rdim"])
    zmin = float(data["zmid"]) - float(data["zdim"]) / 2.0
    zmax = float(data["zmid"]) + float(data["zdim"]) / 2.0
    rs = np.linspace(rmin, rmax, nx)
    zs = np.linspace(zmin, zmax, ny)
    r2d, z2d = np.meshgrid(rs, zs)

    # freeqdsk's GEQDSKFile supports [] but (in newer versions) not .get
    def _lookup(d, *keys, default=None):
        for k in keys:
            try:
                return d[k]
            except (KeyError, TypeError, IndexError):
                continue
        return default

    psi_arr = _lookup(data, "psi", "psirz")
    if psi_arr is None:
        raise RuntimeError("GEQDSK does not contain psi/psirz array")
    flux2d = np.array(psi_arr, dtype=np.float64).reshape((ny, nx))

    dpsidR = np.zeros_like(flux2d)
    dpsidZ = np.zeros_like(flux2d)
    dpsidR[:, 1:-1] = (flux2d[:, 2:] - flux2d[:, :-2]) / (r2d[:, 2:] - r2d[:, :-2])
    dpsidR[:, 0] = (flux2d[:, 1] - flux2d[:, 0]) / (r2d[:, 1] - r2d[:, 0])
    dpsidR[:, -1] = (flux2d[:, -1] - flux2d[:, -2]) / (r2d[:, -1] - r2d[:, -2])
    dpsidZ[1:-1, :] = (flux2d[2:, :] - flux2d[:-2, :]) / (z2d[2:, :] - z2d[:-2, :])
    dpsidZ[0, :] = (flux2d[1, :] - flux2d[0, :]) / (z2d[1, :] - z2d[0, :])
    dpsidZ[-1, :] = (flux2d[-1, :] - flux2d[-2, :]) / (z2d[-1, :] - z2d[-2, :])

    safe_r = np.where(np.abs(r2d) > 1e-12, r2d, 1e-12)
    # EFIT convention (B_p = ∇φ × ∇ψ):
    #     B_R = -(1/R) ∂ψ/∂Z   and   B_Z = +(1/R) ∂ψ/∂R
    # Matches the DG .equ reader above and what the C++ runtime assumes
    # (geqdsk_reader.cpp:254-258, compute_plasma_fields.cpp:2008-2009).
    # Earlier versions of this reader had the opposite signs, which would
    # flip the poloidal drift direction in the Boris pusher. If you hit a
    # case where gyromotion looks reversed, double-check this block first.
    br = -dpsidZ / safe_r
    bz =  dpsidR / safe_r
    bt = (float(data["bcentr"]) * float(data["rcentr"])) / safe_r
    psib = float(_lookup(data, "sibdry", "ssibry", default=0.0))
    return {
        "r": rs, "z": zs, "br": br, "bt": bt, "bz": bz,
        "equ_r": rs.copy(), "equ_z": zs.copy(), "equ_psi": flux2d,
        "btf": float(data["bcentr"]),
        "rtf": float(data["rcentr"]),
        "psib": psib,
        # Magnetic-axis location/value from the GEQDSK. EFIT psi maps can
        # have vacuum-region extrema at the grid corners, so downstream
        # axis detection must not rely on the global min/max of psi.
        "psi_axis": float(_lookup(data, "simagx", "simag", default=0.0)),
        "rmagx": float(_lookup(data, "rmagx", "rmaxis", default=0.0)),
        "zmagx": float(_lookup(data, "zmagx", "zmaxis", default=0.0)),
    }


# ==========================================================================
# Species metadata
# ==========================================================================

def _build_species_metadata(zamax, zamin, zn, am, ns):
    """
    Build per-species metadata from b2fstate arrays.

    SOLPS species convention:
      Each physical element contributes (Zmax - Zmin + 1) species entries.
      e.g., D: Zmin=0 (neutral), Zmax=1 (D+) -> 2 entries
            C: Zmin=0, Zmax=6 -> 7 entries (C0 through C6+)

    Returns:
        names: list of str
        masses_amu: (ns,) float
        charge_states: (ns,) int
        is_neutral: (ns,) bool
    """
    names = []
    masses_amu = np.zeros(ns, dtype=np.float64)
    charge_states = np.zeros(ns, dtype=np.int32)
    is_neutral = np.zeros(ns, dtype=bool)

    # Element symbols by atomic number. SOLPS labels all hydrogen isotopes
    # generically as zn=1; we disambiguate H / D / T below using the mass.
    element_sym = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
        8: "O", 10: "Ne", 18: "Ar", 36: "Kr", 54: "Xe", 74: "W",
    }

    def _hydrogen_isotope_symbol(mass_amu: float) -> str:
        """Map atomic mass of a zn=1 species to H / D / T."""
        if 1.8 <= mass_amu <= 2.3:
            return "D"
        if 2.8 <= mass_amu <= 3.3:
            return "T"
        return "H"

    for i in range(ns):
        z_atom = int(round(zn[i])) if i < len(zn) else 0
        z_min = int(round(zamin[i])) if i < len(zamin) else 0
        z_max = int(round(zamax[i])) if i < len(zamax) else 0
        mass = float(am[i]) if i < len(am) else 0.0
        sym = element_sym.get(z_atom, f"Z{z_atom}")
        if z_atom == 1:
            sym = _hydrogen_isotope_symbol(mass)

        # Determine charge state for this species slot
        # Within an element block, charge goes from Zmin to Zmax
        charge = z_min  # will be refined below
        charge_states[i] = int(round(zamax[i])) if i < len(zamax) else 0
        masses_amu[i] = mass

        if z_min == z_max:
            charge = z_min
        else:
            charge = z_min  # placeholder, refined in block detection

        is_neutral[i] = (charge == 0)

        if charge == 0:
            names.append(f"{sym}0")
        elif charge == 1:
            names.append(f"{sym}+")
        else:
            names.append(f"{sym}{charge}+")

        charge_states[i] = charge

    # Refine: detect element blocks by scanning zamax/zamin transitions
    # Each element block has constant zn and a range zamin..zamax
    idx = 0
    while idx < ns:
        z_atom = int(round(zn[idx]))
        z_min_elem = int(round(zamin[idx]))
        z_max_elem = int(round(zamax[idx]))
        mass_elem = float(am[idx]) if idx < len(am) else 0.0
        sym = element_sym.get(z_atom, f"Z{z_atom}")
        if z_atom == 1:
            sym = _hydrogen_isotope_symbol(mass_elem)
        block_size = z_max_elem - z_min_elem + 1

        for j in range(block_size):
            if idx + j >= ns:
                break
            q = z_min_elem + j
            charge_states[idx + j] = q
            masses_amu[idx + j] = float(am[idx]) if idx < len(am) else 0.0
            is_neutral[idx + j] = (q == 0)
            if q == 0:
                names[idx + j] = f"{sym}0"
            elif q == 1:
                names[idx + j] = f"{sym}+"
            else:
                names[idx + j] = f"{sym}{q}+"

        idx += block_size

    return names, masses_amu, charge_states, is_neutral


# ==========================================================================
# Wall geometry from mesh.extra
# ==========================================================================

# --------------------------------------------------------------------------
# Wall polygon + per-segment B2-cell mapping from mesh.extra + b2fgmtry
# --------------------------------------------------------------------------
# This is the SOLPS-native path to a wall geometry — does NOT require the
# EIRENE triangulation (fort.33/34/35). Works with only SOLPS inputs and
# leaves OpenEdge self-contained vs EIRENE.

def _parse_mesh_extra_to_polygon(mesh_extra: Path, tol: float = 1e-8):
    """Read mesh.extra (R1 Z1 R2 Z2 per line) and walk the segments into
    a single continuous closed polygon (list of (R, Z) points). Returns
    the cleaned point list with the closing vertex duplicated at the end.
    """
    d = np.loadtxt(mesh_extra); d = np.atleast_2d(d)
    if d.shape[1] < 4:
        raise RuntimeError(f"mesh.extra requires >=4 columns, got {d.shape}")
    seg_a = d[:, 0:2].astype(np.float64)
    seg_b = d[:, 2:4].astype(np.float64)
    nseg = seg_a.shape[0]
    used = np.zeros(nseg, dtype=bool); used[0] = True
    pts = [seg_a[0], seg_b[0]]
    for _ in range(nseg - 1):
        cur = pts[-1]
        best_i, best_side, best_d = -1, 0, 1e99
        for i in range(nseg):
            if used[i]: continue
            d1 = float(np.hypot(cur[0]-seg_a[i,0], cur[1]-seg_a[i,1]))
            d2 = float(np.hypot(cur[0]-seg_b[i,0], cur[1]-seg_b[i,1]))
            if d1 < best_d: best_d, best_i, best_side = d1, i, 1
            if d2 < best_d: best_d, best_i, best_side = d2, i, 2
        if best_i < 0 or best_d > max(tol, 1e-5): break
        used[best_i] = True
        pts.append(seg_b[best_i] if best_side == 1 else seg_a[best_i])
    if len(pts) < 3:
        pts = list(seg_a) + [seg_b[-1]]
    clean = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0]-clean[-1][0], p[1]-clean[-1][1]) > tol:
            clean.append(p)
    if np.hypot(clean[0][0]-clean[-1][0], clean[0][1]-clean[-1][1]) > tol:
        clean.append(clean[0])
    return np.asarray(clean)


def _ordered_loops_from_edges(edges):
    """Turn an undirected boundary-edge list into ordered closed loops."""
    from collections import defaultdict

    adj = defaultdict(list)
    for a, b in edges:
        a = int(a); b = int(b)
        adj[a].append(b)
        adj[b].append(a)

    loops = []
    used = set()
    for a0, b0 in edges:
        key0 = tuple(sorted((int(a0), int(b0))))
        if key0 in used:
            continue
        loop = [int(a0), int(b0)]
        used.add(key0)
        prev = int(a0)
        cur = int(b0)
        while True:
            nxt = None
            for cand in adj[cur]:
                k = tuple(sorted((cur, int(cand))))
                if k in used:
                    continue
                nxt = int(cand)
                used.add(k)
                break
            if nxt is None:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
            if cur == loop[0]:
                break
            if cur == prev:
                break
        if len(loop) >= 3 and loop[0] == loop[-1]:
            loops.append(loop)
    return loops


def _wall_from_mesh_extra(mesh_extra_path: Path, nx, ny, crx4, cry4,
                          mesh_wall_face_area):
    """Build a watertight wall from mesh.extra and assign each segment to
    the nearest B2 outer-boundary cell (by midpoint distance). Returns
    (points_rz, segments, seg_cells, seg_areas) where segments are
    (p1_idx, p2_idx) 0-based indices into points_rz.
    """
    poly = _parse_mesh_extra_to_polygon(mesh_extra_path)  # (Npts+1, 2) closed
    pts = poly[:-1]                                       # drop duplicate close
    n_pts = pts.shape[0]
    segs = [(i, (i+1) % n_pts) for i in range(n_pts)]

    # B2 outer-face centroids (only cells with face_area > 0)
    face_r, face_z, face_cell = [], [], []
    for iy_c in range(ny + 2):
        for ix_c in range(nx + 2):
            c = iy_c * (nx + 2) + ix_c
            if mesh_wall_face_area[c] <= 0.0:
                continue
            # choose the outer-face edge of this cell
            if iy_c == 1:      ca, cb = 0, 1
            elif iy_c == ny:   ca, cb = 2, 3
            elif ix_c == nx:   ca, cb = 1, 3
            else:              continue   # ix=1 core side: skipped
            rm = 0.5*(crx4[ix_c,iy_c,ca] + crx4[ix_c,iy_c,cb])
            zm = 0.5*(cry4[ix_c,iy_c,ca] + cry4[ix_c,iy_c,cb])
            face_r.append(rm); face_z.append(zm); face_cell.append(c)
    face_r = np.asarray(face_r); face_z = np.asarray(face_z)
    face_cell = np.asarray(face_cell, dtype=np.int32)

    # For each wall segment, find nearest B2 face; and for each B2 face,
    # find nearest wall segment — aggregate face area onto the segment
    # the face chose. This conserves total flux budget.
    from scipy.spatial import cKDTree
    seg_rmid = 0.5 * (pts[np.array([s[0] for s in segs])][:,0]
                    + pts[np.array([s[1] for s in segs])][:,0])
    seg_zmid = 0.5 * (pts[np.array([s[0] for s in segs])][:,1]
                    + pts[np.array([s[1] for s in segs])][:,1])
    tree_seg = cKDTree(np.column_stack([seg_rmid, seg_zmid]))
    _, seg_for_face = tree_seg.query(np.column_stack([face_r, face_z]))

    n_seg = len(segs)
    seg_cells = np.full(n_seg, -1, dtype=np.int32)
    seg_areas = np.zeros(n_seg, dtype=np.float64)
    dom_area = np.zeros(n_seg, dtype=np.float64)
    for k in range(len(face_cell)):
        s = int(seg_for_face[k]); c = int(face_cell[k])
        a = float(mesh_wall_face_area[c])
        seg_areas[s] += a
        if a > dom_area[s]:
            dom_area[s] = a
            seg_cells[s] = c

    print(f"wall from mesh.extra: {n_seg} segments, "
          f"{(seg_cells >= 0).sum()} mapped to B2 cells, "
          f"captured area = {seg_areas.sum():.2f} m^2 "
          f"(of {mesh_wall_face_area.sum():.2f} m^2 total)")
    return pts, segs, seg_cells, seg_areas


def _layout_polyline(rs, zs, geometry: str):
    """Return (x, y) arrays ordered for SPARTA wall.surf in the requested
    layout, with vertex order forced CCW so SPARTA normals point inward
    and the deck does not need `invert`.

    geometry='cart' (default): x=R, y=Z. Pairs with `boundary o o p`.
    geometry='axi'           : x=Z, y=R. Pairs with `boundary o ao p`.
    """
    r = np.asarray(rs, dtype=np.float64).reshape(-1)
    z = np.asarray(zs, dtype=np.float64).reshape(-1)
    if geometry == "axi":
        x = z; y = r
    elif geometry == "cart":
        x = r; y = z
    else:
        raise ValueError(f"geometry must be 'cart' or 'axi', got {geometry!r}")
    signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if signed_area < 0.0:
        x = x[::-1]
        y = y[::-1]
    return x, y


def _write_sparta_wall_from_mesh_extra(mesh_extra: Path, wall_out: Path,
                                       tol: float = 1e-8,
                                       geometry: str = "cart"):
    if not mesh_extra.exists():
        raise FileNotFoundError(f"mesh.extra not found: {mesh_extra}")

    d = np.loadtxt(mesh_extra)
    d = np.atleast_2d(d)
    if d.shape[1] < 4:
        raise RuntimeError(f"mesh.extra requires >=4 columns, got shape {d.shape}")

    seg_a = d[:, 0:2].astype(np.float64)
    seg_b = d[:, 2:4].astype(np.float64)
    nseg = seg_a.shape[0]
    used = np.zeros(nseg, dtype=bool)
    pts = [seg_a[0], seg_b[0]]
    used[0] = True

    for _ in range(nseg - 1):
        cur = pts[-1]
        best_i, best_side, best_d = -1, 0, 1e99
        for i in range(nseg):
            if used[i]:
                continue
            d1 = float(np.hypot(cur[0] - seg_a[i, 0], cur[1] - seg_a[i, 1]))
            d2 = float(np.hypot(cur[0] - seg_b[i, 0], cur[1] - seg_b[i, 1]))
            if d1 < best_d:
                best_d, best_i, best_side = d1, i, 1
            if d2 < best_d:
                best_d, best_i, best_side = d2, i, 2
        if best_i < 0 or best_d > max(tol, 1e-5):
            break
        used[best_i] = True
        pts.append(seg_b[best_i] if best_side == 1 else seg_a[best_i])

    if len(pts) < 3:
        pts = list(seg_a) + [seg_b[-1]]

    clean = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > tol:
            clean.append(p)
    if np.hypot(clean[0][0] - clean[-1][0], clean[0][1] - clean[-1][1]) > tol:
        clean.append(clean[0])

    r = np.array([p[0] for p in clean[:-1]], dtype=np.float64)
    z = np.array([p[1] for p in clean[:-1]], dtype=np.float64)
    n = int(r.size)
    if n < 3:
        raise RuntimeError("mesh.extra did not produce a valid closed wall polygon")

    x, y = _layout_polyline(r, z, geometry)
    wall_out.parent.mkdir(parents=True, exist_ok=True)
    with wall_out.open("w", encoding="utf-8") as f:
        f.write("surface geometry\n\n")
        f.write(f"{n} points\n{n} lines\n\nPoints\n\n")
        for i in range(n):
            f.write(f"{i+1} {x[i]:.12g} {y[i]:.12g}\n")
        f.write("\nLines\n\n")
        for i in range(n):
            f.write(f"{i+1} {i+1} {(i+1) % n + 1}\n")


# ==========================================================================
# Eirene mesh readers (fort.33, fort.34, fort.35)
# Following Jeremy Lore's SOLPS-routines conventions
# ==========================================================================

def read_eirene_mesh(run_path: Path):
    """
    Read Eirene triangular mesh from fort.33/34/35.

    Returns:
        vtx_r, vtx_z: (nvtx,) vertex coordinates in metres
        tri: (ntri, 3) triangle connectivity (0-based vertex indices)
        b2_ix, b2_iy: (ntri,) B2.5 cell indices *0-based into (nx+2, ny+2)
            plasma arrays*, or -1 if this triangle is not a B2.5 cell.
            The raw fort.35 stores them 1-based in range [1..nx+2],
            [1..ny+2], including both ghost rings; we subtract 1 here so
            the rest of the converter can use `iy*(nx+2)+ix` directly
            against Fortran-flattened (nx+2, ny+2) arrays.
    """
    ft33 = run_path / "fort.33"
    ft34 = run_path / "fort.34"
    ft35 = run_path / "fort.35"

    for f in [ft33, ft34, ft35]:
        if not f.exists():
            raise FileNotFoundError(f"Eirene mesh file not found: {f}")

    # fort.33: node coordinates (cm)
    with ft33.open("r") as f:
        nnodes = int(f.readline().strip())
        vals = []
        for line in f:
            vals.extend(float(x) for x in line.split())
    vtx_r = np.array(vals[:nnodes]) / 100.0  # cm -> m
    vtx_z = np.array(vals[nnodes:2 * nnodes]) / 100.0

    # fort.34: triangle connectivity (1-based -> 0-based)
    with ft34.open("r") as f:
        ntri = int(f.readline().strip())
        tri = np.zeros((ntri, 3), dtype=np.int32)
        for i, line in enumerate(f):
            parts = line.split()
            if len(parts) >= 4:
                tri[i] = [int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1]

    # fort.35: B2.5 cell mapping (last 2 columns = ix, iy, 1-based).
    # We convert to 0-based for direct indexing into (nx+2, ny+2) arrays;
    # -1 (and any non-positive) means "this triangle is not a B2.5 cell".
    b2_ix = np.full(ntri, -1, dtype=np.int32)
    b2_iy = np.full(ntri, -1, dtype=np.int32)
    with ft35.open("r") as f:
        _ = int(f.readline().strip())
        for i, line in enumerate(f):
            parts = line.split()
            if len(parts) >= 12:
                ix_val = int(parts[10])
                iy_val = int(parts[11])
                if ix_val > 0 and iy_val > 0:
                    b2_ix[i] = ix_val - 1
                    b2_iy[i] = iy_val - 1

    print(f"Eirene mesh: {nnodes} vertices, {ntri} triangles")
    has_b2 = (b2_ix >= 0) & (b2_iy >= 0)
    print(f"  B2.5 mapped: {has_b2.sum()}, vacuum/PFR: {(~has_b2).sum()}")

    return vtx_r, vtx_z, tri, b2_ix, b2_iy


# ==========================================================================
# Main conversion
# ==========================================================================

def convert_solps_to_openedge(
    run_path: Path,
    plasma_out: Path,
    nr: int = 300,
    nz: int = 300,
    rmin: float | None = None,
    rmax: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    gfile: Path | None = None,
    equ_file: Path | None = None,
    plot: bool = False,
    plot_prefix: Path | None = None,
    wall_out: Path | None = None,
    wall_in: Path | None = None,
    mesh_extra: Path | None = None,
    b2fgmtry_path: Path | None = None,
    b2fstate_path: Path | None = None,
    wall_source: str = "auto",
    geometry: str = "cart",
    heatflux: str = "none",
) -> None:
    """
    Convert SOLPS run directory to OpenEdge plasma.h5 (B-field embedded).

    Reads b2fgmtry (geometry) and b2fstate (plasma state) directly —
    no quixote dependency.
    """
    b2fgmtry_file = b2fgmtry_path if b2fgmtry_path else run_path / "b2fgmtry"
    b2fstate_file = b2fstate_path if b2fstate_path else run_path / "b2fstate"

    for f in [b2fgmtry_file, b2fstate_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required SOLPS file not found: {f}")

    # -- Geometry --
    nx, ny, ns = b2f_read_dims(b2fstate_file)
    print(f"SOLPS grid: nx={nx}, ny={ny}, ns={ns}")

    gmtry = read_b2fgmtry(str(b2fgmtry_file))
    crx = gmtry["crx"]
    cry = gmtry["cry"]
    rc, zc = _cell_centers_from_corners(crx, cry, nx, ny)
    cell_polys = _cell_polygons_from_corners(crx, cry, nx, ny)

    # Regular-grid build (was used for an earlier regular-grid plasma.h5
    # output mode that's been retired). Mesh-format output doesn't need it.
    # The CLI still accepts --nr/--nz for plot-mode parity but the values
    # only affect the diagnostic print below.

    # -- Electron fields --
    # b2fstate stores te/ti in eV (some SOLPS versions store in Joules — check)
    ne = b2f_extract("ne", b2fstate_file)  # (nx+2, ny+2)
    te = b2f_extract("te", b2fstate_file)  # (nx+2, ny+2)
    ti = b2f_extract("ti", b2fstate_file)  # (nx+2, ny+2)

    # Electric potential po [V] from balance.nc (not b2fstate). SOLPS
    # solves a current-continuity equation and writes po — OpenEdge
    # reads this directly so we never have to reconstruct an ambipolar
    # approximation from electron pressure balance.
    po_data = None
    try:
        import netCDF4 as _nc
        balance_path = run_path / "balance.nc"
        if balance_path.exists():
            _bds = _nc.Dataset(str(balance_path))
            if "po" in _bds.variables:
                po_raw = np.asarray(_bds["po"][:], dtype=np.float64)
                # balance.nc is (ny+2, nx+2); b2fstate is (nx+2, ny+2).
                # Transpose to match the rest of the converter.
                if po_raw.shape == (ny + 2, nx + 2):
                    po_data = po_raw.T
                elif po_raw.shape == (nx + 2, ny + 2):
                    po_data = po_raw
                else:
                    print(f"WARNING: unexpected po shape {po_raw.shape}; skipping E-field")
            _bds.close()
    except Exception as _e:
        print(f"WARNING: could not read po from balance.nc: {_e}")
    if po_data is None:
        print("WARNING: no electric potential po found — E = 0 in plasma.h5")
        po_data = np.zeros_like(ne)

    # SOLPS stores temperatures in Joules; convert to eV
    eV = 1.602176634e-19
    te_eV = te / eV
    ti_eV = ti / eV

    # -- Per-species ion data --
    na = b2f_extract("na", b2fstate_file)  # (nx+2, ny+2, ns)
    ua_raw = b2f_extract("ua", b2fstate_file)  # (nx+2, ny+2) or (nx+2, ny+2, ns)

    # Species metadata
    zamax = b2f_extract("zamax", b2fstate_file, reshape=False)
    zamin = b2f_extract("zamin", b2fstate_file, reshape=False)
    zn = b2f_extract("zn", b2fstate_file, reshape=False)
    am = b2f_extract("am", b2fstate_file, reshape=False)

    species_names, masses_amu, charge_states, is_neutral = _build_species_metadata(
        zamax, zamin, zn, am, ns
    )
    print(f"Species detected ({ns}): {species_names}")
    print(f"Charge states: {charge_states.tolist()}")

    # Filter to charged species only (skip neutrals)
    ion_mask = ~is_neutral
    ion_indices = np.where(ion_mask)[0]
    nion = int(ion_indices.size)
    print(f"Charged species: {nion} (skipping {ns - nion} neutrals)")
    if nion == 0:
        raise RuntimeError("No charged ion species found in b2fstate")

    main_sidx = int(ion_indices[0])
    main_dens_raw = na[:, :, main_sidx] if na.ndim == 3 else na
    if ua_raw.ndim == 3 and ua_raw.shape[2] > main_sidx:
        main_upar_raw = ua_raw[:, :, main_sidx]
    elif ua_raw.ndim == 2:
        main_upar_raw = ua_raw
    else:
        main_upar_raw = np.zeros_like(main_dens_raw)

    # B-field from equilibrium file only — the only trustworthy source.
    # The returned dict also carries the raw equilibrium (r, z, psi,
    # btf, rtf, psib) that we embed into plasma.h5 /equilibrium so
    # downstream consumers don't need the .equ file at run time.
    if gfile is not None:
        equ_dict = _read_geqdsk_bfield(gfile)
    elif equ_file is not None:
        equ_dict = _read_equilibrium_bfield(equ_file)
    else:
        raise RuntimeError(
            "Equilibrium file required for B-field reconstruction.\n"
            "Provide --gfile (GEQDSK) or --equ-file (.equ)."
        )
    # -- Per-ion metadata (names, masses, charges) --
    # Per-species plasma fields are written as flat B2 cell arrays in
    # plasma.h5 /mesh/ions/*. No regular-grid interpolation.
    ion_names = []
    ion_masses = np.zeros(nion, dtype=np.float64)
    ion_charges = np.zeros(nion, dtype=np.int32)

    for k, sidx in enumerate(ion_indices):
        # Isotope disambiguation already happened in _build_species_metadata,
        # so species_names carries the correct H / D / T label.
        ion_names.append(species_names[sidx])
        ion_masses[k] = masses_amu[sidx]
        ion_charges[k] = charge_states[sidx]
    main_k = 0

    # -- Build Eirene mesh with B2.5 plasma data --
    # Read the Eirene triangulation (fort.33/34/35) which covers the full
    # SOLPS domain including near-wall regions.  Each Eirene triangle maps
    # to a B2.5 cell via fort.35, so we assign plasma data from the
    # corresponding B2.5 cell.  Triangles outside the B2.5 domain (vacuum,
    # PFR) get zero plasma values and are naturally skipped by the C++ code.
    eirene_path = run_path  # fort.33/34/35 are in the run directory
    try:
        mesh_vtx_r, mesh_vtx_z, mesh_tri_raw, b2_ix, b2_iy = read_eirene_mesh(eirene_path)
    except FileNotFoundError:
        # Fall back to baserun directory
        if b2fgmtry_path and b2fgmtry_path.parent != run_path:
            mesh_vtx_r, mesh_vtx_z, mesh_tri_raw, b2_ix, b2_iy = read_eirene_mesh(b2fgmtry_path.parent)
        else:
            raise

    mesh_tri = mesh_tri_raw
    ntri_eirene = len(mesh_tri)

    # Map Eirene triangles to B2.5 cell data.
    # B2.5 arrays are (nx+2, ny+2) with Fortran ordering.
    # b2_ix, b2_iy are 0-based B2.5 indices into (nx+2, ny+2) arrays,
    # already converted from fort.35's 1-based encoding by read_eirene_mesh.
    ncell_flat = (nx + 2) * (ny + 2)
    ne_flat = ne.reshape(ncell_flat, order="F")
    te_flat = te_eV.reshape(ncell_flat, order="F")
    ti_flat = ti_eV.reshape(ncell_flat, order="F")
    ni_flat = main_dens_raw.reshape(ncell_flat, order="F")
    upar_flat = main_upar_raw.reshape(ncell_flat, order="F")

    # Per-triangle cell index into the flat B2.5 array
    mesh_cell_idx = np.full(ntri_eirene, -1, dtype=np.int32)
    for t in range(ntri_eirene):
        ix, iy = int(b2_ix[t]), int(b2_iy[t])
        if ix >= 0 and iy >= 0 and ix < nx + 2 and iy < ny + 2:
            mesh_cell_idx[t] = iy * (nx + 2) + ix

    has_cell_native = mesh_cell_idx >= 0
    tri_source_kind = np.zeros(ntri_eirene, dtype=np.int8)
    tri_source_kind[has_cell_native] = 1  # native fort.35 mapping
    wall_gap_mask = np.zeros(ntri_eirene, dtype=bool)
    print(f"Eirene mesh -> B2.5 mapping: {has_cell_native.sum()} triangles "
          f"with plasma, {(~has_cell_native).sum()} vacuum/PFR (native)")

    # Centroid bookkeeping for the projections below.
    centroid_r = mesh_vtx_r[mesh_tri].mean(axis=1)
    centroid_z = mesh_vtx_z[mesh_tri].mean(axis=1)
    max_proj_dist = 0.05   # metres

    mesh_extra_path_local = mesh_extra if mesh_extra is not None else (run_path / "mesh.extra")
    wall_poly = None
    wall_samples = None
    if mesh_extra_path_local.exists():
        try:
            wall_poly = _parse_mesh_extra_to_polygon(mesh_extra_path_local)
            dens_r, dens_z = [], []
            for i in range(len(wall_poly) - 1):
                p0 = wall_poly[i]
                p1 = wall_poly[i + 1]
                seg = float(np.hypot(*(p1 - p0)))
                n_seg = max(2, int(np.ceil(seg / 0.01)) + 1)
                t = np.linspace(0.0, 1.0, n_seg)
                dens_r.extend(p0[0] + t * (p1[0] - p0[0]))
                dens_z.extend(p0[1] + t * (p1[1] - p0[1]))
            wall_samples = np.column_stack([dens_r, dens_z])
        except Exception as e:
            print(f"WARNING: could not parse mesh.extra wall polygon for gap logic: {e}")

    # Boundary edges geometrically close to the physical wall are still
    # identified for diagnostics and downstream wall bookkeeping, but
    # non-native EIRENE triangles are not back-filled with neighboring
    # B2.5 plasma data.
    edge_count = {}
    edge_tri = {}
    tri_neighbors = [[] for _ in range(ntri_eirene)]
    for t in range(ntri_eirene):
        tri_row = mesh_tri[t]
        for a, b in [(tri_row[0], tri_row[1]),
                     (tri_row[1], tri_row[2]),
                     (tri_row[2], tri_row[0])]:
            e = (int(min(a, b)), int(max(a, b)))
            if e in edge_tri:
                other = edge_tri[e]
                tri_neighbors[t].append(other)
                tri_neighbors[other].append(t)
                edge_count[e] = edge_count.get(e, 0) + 1
            else:
                edge_count[e] = 1
                edge_tri[e] = t
    boundary_edges_all = [e for e, n in edge_count.items() if n == 1]

    native_idx = np.where(has_cell_native)[0]
    native_cells = mesh_cell_idx[native_idx]
    native_iy = native_cells // (nx + 2)
    native_ix = native_cells - native_iy * (nx + 2)
    native_is_sheath = ((native_ix == 1) | (native_ix == nx) | (native_iy == ny))
    sheath_src = native_idx[native_is_sheath]
    if sheath_src.size == 0:
        sheath_src = native_idx

    wall_edge_mid_r = np.asarray(
        [0.5 * (mesh_vtx_r[a] + mesh_vtx_r[b]) for a, b in boundary_edges_all],
        dtype=np.float64
    )
    wall_edge_mid_z = np.asarray(
        [0.5 * (mesh_vtx_z[a] + mesh_vtx_z[b]) for a, b in boundary_edges_all],
        dtype=np.float64
    )
    if wall_samples is not None and len(boundary_edges_all) > 0:
        from scipy.spatial import cKDTree
        tree_wall = cKDTree(wall_samples)
        edge_wall_dist, _ = tree_wall.query(np.column_stack([wall_edge_mid_r, wall_edge_mid_z]))
        wall_boundary_mask = edge_wall_dist <= max_proj_dist
    else:
        wall_boundary_mask = np.ones(len(boundary_edges_all), dtype=bool)
    wall_boundary_edges = [e for e, keep in zip(boundary_edges_all, wall_boundary_mask) if keep]
    wall_boundary_loops = _ordered_loops_from_edges(wall_boundary_edges)
    wall_boundary_verts = sorted({v for e in wall_boundary_edges for v in e})
    wall_edge_mid_r = wall_edge_mid_r[wall_boundary_mask]
    wall_edge_mid_z = wall_edge_mid_z[wall_boundary_mask]
    if len(wall_boundary_edges) > 0:
        from scipy.spatial import cKDTree
        tree_src = cKDTree(np.column_stack([centroid_r[sheath_src], centroid_z[sheath_src]]))
        _, src_j = tree_src.query(np.column_stack([wall_edge_mid_r, wall_edge_mid_z]))
        wall_edge_cells = mesh_cell_idx[sheath_src[src_j]].astype(np.int32, copy=False)
    else:
        wall_edge_cells = np.array([], dtype=np.int32)
    print(f"EIRENE wall-facing boundary edges: {len(wall_boundary_edges)} "
          f"across {len(wall_boundary_loops)} loop(s)")

    n_vacuum_native = int((~has_cell_native).sum())
    if n_vacuum_native > 0:
        print(f"Eirene mesh -> B2.5 mapping: {n_vacuum_native} vacuum/PFR triangles "
              "kept at zero plasma (no wall-vacuum extrapolation)")

    print("Triangle provenance: "
          f"native={int((tri_source_kind == 1).sum())}, "
          f"filled-vacuum={int((tri_source_kind == 2).sum())}, "
          f"wall-gap={int(wall_gap_mask.sum())}, "
          f"vacuum={int((mesh_cell_idx < 0).sum())}")

    # ------------------------------------------------------------------
    # Per-cell wall face area for B2 boundary cells.
    # ------------------------------------------------------------------
    # SOLPS corner ordering (inferred from _cell_polygons_from_corners):
    #   c=0 lower-left  (-x, -y), c=1 lower-right (+x, -y),
    #   c=3 upper-right (+x, +y), c=2 upper-left  (-x, +y)
    # SOLPS-ITER convention for this grid (verified empirically by
    # probing (R, Z) and Te at corner cells, 2026-04-21):
    #   ix  = poloidal  (sweeps from one divertor target to the other)
    #   iy  = radial    (iy=1 is CORE-SIDE, iy=ny is OUTER-SOL/WALL)
    #
    # Physical walls:
    #   ix=1   inner divertor target
    #   ix=nx  outer divertor target
    #   iy=ny  outer SOL / main-chamber wall
    # NOT a wall:
    #   iy=1   core-side radial flux surface (hot plasma, virtual boundary)
    #
    # Cell-corner convention (per b2fgmtry):
    #   corner 0 = (ix-, iy-)   "lower-left"
    #   corner 1 = (ix+, iy-)   "lower-right"
    #   corner 2 = (ix-, iy+)   "upper-left"
    #   corner 3 = (ix+, iy+)   "upper-right"
    # Outer faces per cell:
    #   ix=1  (left target, -x side):     edge 0-2
    #   ix=nx (right target, +x side):    edge 1-3
    #   iy=ny (outer-SOL/wall, +y side):  edge 2-3
    #
    # Face area (axisymmetric) = 2*pi * R_mid * poloidal_length.
    # Interior cells get 0. Units: m^2.
    crx4 = crx.reshape(nx + 2, ny + 2, 4, order="F")
    cry4 = cry.reshape(nx + 2, ny + 2, 4, order="F")
    mesh_wall_face_area = np.zeros(ncell_flat, dtype=np.float64)

    def _edge_area(ix_c, iy_c, ca, cb):
        Ra, Rb = crx4[ix_c, iy_c, ca], crx4[ix_c, iy_c, cb]
        Za, Zb = cry4[ix_c, iy_c, ca], cry4[ix_c, iy_c, cb]
        Lp = np.sqrt((Rb - Ra) ** 2 + (Zb - Za) ** 2)
        return 2.0 * np.pi * 0.5 * (Ra + Rb) * Lp

    for ix_c in range(nx + 2):
        for iy_c in range(ny + 2):
            area = 0.0
            if ix_c == 1:     area += _edge_area(ix_c, iy_c, 0, 2)  # inner target
            if ix_c == nx:    area += _edge_area(ix_c, iy_c, 1, 3)  # outer target
            if iy_c == ny:    area += _edge_area(ix_c, iy_c, 2, 3)  # outer SOL/wall
            if area > 0.0:
                c_flat = iy_c * (nx + 2) + ix_c
                mesh_wall_face_area[c_flat] = area
    n_bc = int((mesh_wall_face_area > 0).sum())
    print(f"B2 wall face areas (ix=1 inner target + ix=nx outer target "
          f"+ iy=ny outer wall; core iy=1 excluded): "
          f"{n_bc} boundary cells, total area = "
          f"{mesh_wall_face_area.sum():.3e} m^2")

    # ------------------------------------------------------------------
    # Build per-wall-segment -> B2-cell mapping for the existing
    # (watertight) wall.surf. For each wall.surf segment midpoint, find
    # the B2 boundary cell whose outer-face midpoint is closest — the
    # segment is assigned that cell's flat index (iy*(nx+2)+ix).
    #
    # Boundary face edges, by the corner convention used here:
    #   iy=1  (lower row): edge 0-1
    #   iy=ny (upper row): edge 2-3
    #   ix=1  (inner col): edge 0-2
    #   ix=nx (outer col): edge 1-3
    #
    # Only done if the caller also asked for wall.surf generation
    # (--wall-out). The mapping is written to mesh/wall_surf_cell so the
    # fix at runtime can index it directly by SPARTA surf ID.
    # ------------------------------------------------------------------
    # Wall geometry + per-segment B2-cell mapping.
    # wall_source=
    #   "mesh-extra": walk mesh.extra into closed polygon, match each
    #                 segment to nearest B2 boundary face. SOLPS-native,
    #                 no EIRENE files needed.
    #   "eirene":     use fort.33/34/35 boundary edges. Exact mapping,
    #                 but requires EIRENE output.
    #   "b2":         (not yet implemented) walk B2 outer boundary only.
    #   "auto":       prefer mesh-extra if available, else eirene, else b2.
    # ------------------------------------------------------------------
    mesh_wall_surf_cell = np.array([], dtype=np.int32)
    mesh_wall_surf_area = np.array([], dtype=np.float64)

    # Resolve "auto": pick the first available path.
    mesh_extra_path = mesh_extra if mesh_extra is not None else (run_path / "mesh.extra")
    if not mesh_extra_path.exists() and b2fgmtry_path:
        mesh_extra_path = b2fgmtry_path.parent / "mesh.extra"
    effective_wall_source = wall_source
    if wall_source == "auto":
        if mesh_extra_path.exists():
            effective_wall_source = "mesh-extra"
        else:
            effective_wall_source = "eirene"
    print(f"Wall-source path: {effective_wall_source}")

    # --- mesh-extra path (SOLPS-native, no EIRENE) ----------------------
    if effective_wall_source == "mesh-extra" and wall_out is not None:
        if not mesh_extra_path.exists():
            raise FileNotFoundError(
                f"wall_source=mesh-extra requested but {mesh_extra_path} missing")
        me_pts, me_segs, me_cells, me_areas = _wall_from_mesh_extra(
            mesh_extra_path, nx, ny, crx4, cry4, mesh_wall_face_area)
        me_r = np.array([rv for rv, _ in me_pts], dtype=np.float64)
        me_z = np.array([zv for _, zv in me_pts], dtype=np.float64)
        x_pts, y_pts = _layout_polyline(me_r, me_z, geometry)
        wall_out.parent.mkdir(parents=True, exist_ok=True)
        with wall_out.open("w", encoding="utf-8") as f:
            f.write("surface geometry\n\n")
            f.write(f"{len(me_pts)} points\n{len(me_segs)} lines\n\nPoints\n\n")
            for i in range(len(me_pts)):
                f.write(f"{i+1} {x_pts[i]:.12g} {y_pts[i]:.12g}\n")
            f.write("\nLines\n\n")
            for i, (a, b) in enumerate(me_segs):
                f.write(f"{i+1} {a+1} {b+1}\n")
        layout = "x=Z, y=R" if geometry == "axi" else "x=R, y=Z"
        print(f"Wrote wall.surf from mesh.extra ({geometry}: {layout}): {wall_out}")
        mesh_wall_surf_cell = me_cells
        mesh_wall_surf_area = me_areas

    # --- eirene path (uses fort.33/34/35) -------------------------------
    if effective_wall_source == "eirene" and wall_out is not None:
        # Below is the original EIRENE-triangulation-boundary-edges path.
        # Kept for exact validation against standalone EIRENE; not required
        # to run OpenEdge on its own.
        _eirene_path_active = True
    else:
        _eirene_path_active = False

    # Short-circuit: if not on the EIRENE path, skip the entire
    # triangulation-edge block by jumping past it using early-continue
    # trick via `if not _eirene_path_active: raise StopIteration` then
    # catch. Actually: use a named block-if.
    edge_count = {}
    edge_tri = {}
    for t in range(ntri_eirene):
        tri = mesh_tri[t]
        for a, b in [(0,1), (1,2), (2,0)]:
            e = (int(min(tri[a], tri[b])), int(max(tri[a], tri[b])))
            edge_count[e] = edge_count.get(e, 0) + 1
            edge_tri[e] = t   # last-seen triangle; for boundary edges it's the unique one

    boundary_edges = [e for e, n in edge_count.items() if n == 1]
    print(f"EIRENE boundary edges: {len(boundary_edges)} "
          f"(from {ntri_eirene} triangles)")

    # Include ALL boundary edges (to keep the wall watertight), but
    # ORIENT them into traversal sequences so each vertex is p2 of one
    # segment and p1 of the next. SPARTA's watertight check requires
    # this orientation. Handles multiple disjoint loops (the outer wall
    # and the core-side inner boundary form separate closed loops).
    from collections import defaultdict
    adj = defaultdict(list)   # vertex -> list of (other_vertex, edge_id)
    for k, (a, b) in enumerate(boundary_edges):
        adj[a].append((b, k))
        adj[b].append((a, k))

    wall_edges = []         # ordered (va, vb) with va = prev, vb = next
    wall_edge_cells = []
    used = [False] * len(boundary_edges)
    for start_k in range(len(boundary_edges)):
        if used[start_k]: continue
        a0, b0 = boundary_edges[start_k]
        used[start_k] = True
        t0 = edge_tri[(a0, b0)]
        wall_edges.append((a0, b0))
        wall_edge_cells.append(int(mesh_cell_idx[t0]))
        cur = b0
        while True:
            nxt = None
            for (other, eid) in adj[cur]:
                if not used[eid]:
                    nxt = (other, eid); break
            if nxt is None: break
            other, eid = nxt
            used[eid] = True
            t = edge_tri[boundary_edges[eid]]
            wall_edges.append((cur, other))
            wall_edge_cells.append(int(mesh_cell_idx[t]))
            cur = other
            if cur == a0: break   # closed loop

    n_emitting = sum(1 for c in wall_edge_cells
                     if c >= 0 and mesh_wall_face_area[c] > 0.0)
    print(f"EIRENE wall edges (oriented traversal): "
          f"{len(wall_edges)} total ({n_emitting} on outer wall, rest "
          f"are inner/non-emitting)")

    # Write SPARTA wall.surf from these EIRENE boundary edges. Segment i
    # in wall.surf corresponds to B2 cell wall_edge_cells[i].
    #
    # Dedupe vertices: each EIRENE vertex participating in N boundary
    # edges appears N times otherwise; SPARTA's watertight check demands
    # shared point indices between adjacent segments. Collect unique
    # vertex indices and remap segment endpoints.
    if wall_edges and wall_out is not None and _eirene_path_active:
        # Dedupe by (R,Z) coordinates — EIRENE sometimes has distinct
        # vertex indices at the same physical point (adjacent B2 quads
        # sharing a corner). SPARTA requires points to be geometrically
        # unique. Use a spatial tolerance (1e-8 m) to snap together
        # coincident vertices before writing.
        used_verts = sorted(set(v for e in wall_edges for v in e))
        # Snap to grid with 10 nm tolerance
        TOL = 1e-8
        point_map = {}     # (rx, zy) rounded -> SPARTA point index (1-based)
        vmap = {}          # eirene-vertex-index -> SPARTA point index
        unique_rz = []
        for v in used_verts:
            key = (round(mesh_vtx_r[v] / TOL), round(mesh_vtx_z[v] / TOL))
            if key not in point_map:
                point_map[key] = len(unique_rz) + 1   # 1-based
                unique_rz.append((mesh_vtx_r[v], mesh_vtx_z[v]))
            vmap[v] = point_map[key]

        wall_out.parent.mkdir(parents=True, exist_ok=True)
        with wall_out.open("w", encoding="utf-8") as f:
            f.write("surface geometry\n\n")
            f.write(f"{len(unique_rz)} points\n{len(wall_edges)} lines"
                    f"\n\nPoints\n\n")
            for i, (rv, zv) in enumerate(unique_rz):
                if geometry == "axi":
                    f.write(f"{i+1} {zv:.12g} {rv:.12g}\n")
                else:
                    f.write(f"{i+1} {rv:.12g} {zv:.12g}\n")
            f.write("\nLines\n\n")
            for i, (va, vb) in enumerate(wall_edges):
                f.write(f"{i+1} {vmap[va]} {vmap[vb]}\n")
        layout = "x=Z, y=R" if geometry == "axi" else "x=R, y=Z"
        print(f"Wrote EIRENE-consistent wall ({geometry}: {layout}): {wall_out} "
              f"({len(wall_edges)} segments, {len(unique_rz)} unique pts "
              f"after snap)")
        mesh_wall_surf_cell = np.asarray(wall_edge_cells, dtype=np.int32)
        mesh_wall_surf_area = mesh_wall_face_area[mesh_wall_surf_cell].astype(
            np.float64)
    # else: mesh_wall_surf_cell/area were already set by the mesh-extra
    # path (or remain empty if neither ran).

    # Legacy path below — keep in case wall_out wasn't passed but an
    # existing wall.surf is already present; we still write the mapping.
    # If the user passed --wall-in (a custom SPARTA wall.surf to map
    # against — typically a strip-of-target test surface that doesn't
    # match the auto-generated full-vessel wall.surf), skip the auto
    # paths' mapping and use the user's geometry directly.
    if wall_in is not None:
        wall_path_for_map = wall_in
        mesh_wall_surf_cell = np.array([], dtype=np.int32)
        mesh_wall_surf_area = np.array([], dtype=np.float64)
    else:
        wall_path_for_map = wall_out if (wall_out and wall_out.exists()) else \
                            (plasma_out.parent / "wall.surf")
    if wall_path_for_map.exists() and mesh_wall_surf_cell.size == 0:
        wall_out = wall_path_for_map   # rebind for the block below
        def _edge_mid(ix_c, iy_c, ca, cb):
            Ra, Za = crx4[ix_c, iy_c, ca], cry4[ix_c, iy_c, ca]
            Rb, Zb = crx4[ix_c, iy_c, cb], cry4[ix_c, iy_c, cb]
            return 0.5*(Ra+Rb), 0.5*(Za+Zb)

        face_rc, face_zc, face_cell = [], [], []
        for ix_c in range(nx + 2):
            if 1 <= ix_c <= nx:
                for (iy_c, ca, cb) in [(1, 0, 1), (ny, 2, 3)]:
                    rmid, zmid = _edge_mid(ix_c, iy_c, ca, cb)
                    face_rc.append(rmid); face_zc.append(zmid)
                    face_cell.append(iy_c*(nx+2) + ix_c)
        for iy_c in range(ny + 2):
            if 1 <= iy_c <= ny:
                for (ix_c, ca, cb) in [(1, 0, 2), (nx, 1, 3)]:
                    rmid, zmid = _edge_mid(ix_c, iy_c, ca, cb)
                    face_rc.append(rmid); face_zc.append(zmid)
                    face_cell.append(iy_c*(nx+2) + ix_c)
        face_rc = np.asarray(face_rc); face_zc = np.asarray(face_zc)
        face_cell = np.asarray(face_cell, dtype=np.int32)

        # Parse wall.surf back in to get segment midpoints in the exact
        # order SPARTA will read them.
        import re as _re
        pts, segs = [], []
        mode = None
        with wall_out.open() as _f:
            for L in _f:
                s = L.strip()
                if not s or s.startswith("#"): continue
                if s.lower().startswith("points"): mode = "P"; continue
                if s.lower().startswith("lines"):  mode = "L"; continue
                if not _re.match(r"^\d", s): continue
                parts = s.split()
                if mode == "P": pts.append((float(parts[1]), float(parts[2])))
                if mode == "L": segs.append((int(parts[1]), int(parts[2])))
        if pts and segs:
            P = np.array(pts)
            S = np.array(segs)
            seg_rmid = 0.5 * (P[S[:,0]-1, 0] + P[S[:,1]-1, 0])
            seg_zmid = 0.5 * (P[S[:,0]-1, 1] + P[S[:,1]-1, 1])

            from scipy.spatial import cKDTree
            nseg_wall = len(S)
            face_area = mesh_wall_face_area[face_cell]
            seg_dominant_cell = np.full(nseg_wall, -1, dtype=np.int32)
            seg_total_area = np.zeros(nseg_wall, dtype=np.float64)

            if wall_in is not None:
                # User-supplied wall.surf may be a partial strip that
                # doesn't cover the full B2 boundary. For each segment,
                # find the NEAREST B2 boundary face (segment->face), so
                # only the segments themselves get cell assignments and
                # nothing is forced onto far-away strip points.
                tree = cKDTree(np.column_stack([face_rc, face_zc]))
                _, face_for_seg = tree.query(
                    np.column_stack([seg_rmid, seg_zmid]))
                for s in range(nseg_wall):
                    k = int(face_for_seg[s])
                    seg_dominant_cell[s] = int(face_cell[k])
                    seg_total_area[s] = float(face_area[k])
            else:
                # Default (full wall): each B2 boundary face chooses its
                # closest wall segment. Segments accumulate face area;
                # the dominant cell wins. Conserves the total flux
                # budget when user-wall and B2-boundary cover the same
                # extent.
                tree = cKDTree(np.column_stack([seg_rmid, seg_zmid]))
                _, seg_for_face = tree.query(
                    np.column_stack([face_rc, face_zc]))
                seg_dominant_area = np.zeros(nseg_wall, dtype=np.float64)
                for k in range(len(face_cell)):
                    s = int(seg_for_face[k])
                    c = int(face_cell[k])
                    a = float(face_area[k])
                    seg_total_area[s] += a
                    if a > seg_dominant_area[s]:
                        seg_dominant_area[s] = a
                        seg_dominant_cell[s] = c
            mesh_wall_surf_cell = seg_dominant_cell
            print(f"wall.surf <- B2 face mapping: {len(face_cell)} B2 faces "
                  f"distributed onto {nseg_wall} wall segments "
                  f"({(seg_dominant_cell >= 0).sum()} segments own a cell), "
                  f"captured area = {seg_total_area.sum():.2f} / "
                  f"{mesh_wall_face_area.sum():.2f} m^2 "
                  f"({100.0*seg_total_area.sum()/mesh_wall_face_area.sum():.1f}%)")
            # Per-segment captured face area is also written, so the fix
            # can use it instead of face_area at the dominant-cell's index
            # (which would otherwise under-weight segments that own many
            # B2 faces but got the smallest face's area).
            mesh_wall_surf_area = seg_total_area

    # Per-cell plasma arrays (shared across all triangles referencing the same cell)
    mesh_ne = ne_flat.copy()
    mesh_te = te_flat.copy()
    mesh_ti = ti_flat.copy()
    mesh_ni = ni_flat.copy()
    mesh_upar = upar_flat.copy()

    # Multi-ion per-cell data
    mesh_ions_dens = np.zeros((nion, ncell_flat), dtype=np.float64)
    mesh_ions_temp = np.zeros((nion, ncell_flat), dtype=np.float64)
    mesh_ions_upar = np.zeros((nion, ncell_flat), dtype=np.float64)
    for k, sidx in enumerate(ion_indices):
        n_s = na[:, :, sidx] if na.ndim == 3 else na
        mesh_ions_dens[k] = n_s.reshape(ncell_flat, order="F")
        mesh_ions_temp[k] = ti_eV.reshape(ncell_flat, order="F")
        if ua_raw.ndim == 3 and ua_raw.shape[2] > sidx:
            mesh_ions_upar[k] = ua_raw[:, :, sidx].reshape(ncell_flat, order="F")
        elif ua_raw.ndim == 2:
            mesh_ions_upar[k] = ua_raw.reshape(ncell_flat, order="F")

    # NaN cleanup
    for arr in [mesh_ne, mesh_te, mesh_ti, mesh_ni, mesh_upar,
                mesh_ions_dens, mesh_ions_temp, mesh_ions_upar]:
        arr[~np.isfinite(arr)] = 0.0

    # -- Per-cell gradients (grad Te, grad Ti, grad ne) on the B2 mesh --
    # Use a weighted least-squares fit directly in physical (R, Z) space
    # around each B2 cell center. This is much more robust near the strike
    # point / X-point than differentiating in (ix, iy) and pushing through
    # the local Jacobian, which amplifies mesh distortion and small det(J).
    nxp = nx + 2
    nyp = ny + 2
    # 2D views on the (nxp, nyp) B2 grid. Use C-order here because
    # ne_flat was built with order='F' from a (nxp, nyp) array, so
    # reshape(nxp, nyp, order='F') gives back the original (ix, iy) layout.
    def _grid(flat):
        return flat.reshape(nxp, nyp, order='F')
    Te_g = _grid(te_flat)
    Ti_g = _grid(ti_flat)
    Rc = rc   # (nxp, nyp) cell-center R
    Zc = zc   # (nxp, nyp) cell-center Z

    def _grad_rz_lsq(field_g):
        g_r = np.zeros_like(field_g)
        g_z = np.zeros_like(field_g)
        finite_geom = np.isfinite(Rc) & np.isfinite(Zc)

        def _accumulate(ix0, iy0, radius):
            r0 = Rc[ix0, iy0]
            z0 = Zc[ix0, iy0]
            f0 = field_g[ix0, iy0]
            if not (np.isfinite(r0) and np.isfinite(z0) and np.isfinite(f0)):
                return None
            a11 = a12 = a22 = 0.0
            b1 = b2 = 0.0
            npt = 0
            ix_lo = max(0, ix0 - radius)
            ix_hi = min(nxp, ix0 + radius + 1)
            iy_lo = max(0, iy0 - radius)
            iy_hi = min(nyp, iy0 + radius + 1)
            for ix1 in range(ix_lo, ix_hi):
                for iy1 in range(iy_lo, iy_hi):
                    if ix1 == ix0 and iy1 == iy0:
                        continue
                    if not finite_geom[ix1, iy1]:
                        continue
                    f1 = field_g[ix1, iy1]
                    if not np.isfinite(f1):
                        continue
                    dR = Rc[ix1, iy1] - r0
                    dZ = Zc[ix1, iy1] - z0
                    ds2 = dR * dR + dZ * dZ
                    if ds2 <= 1e-20:
                        continue
                    # Inverse-distance weighting favors the local stencil
                    # while remaining robust on irregular cells.
                    w = 1.0 / ds2
                    df = f1 - f0
                    a11 += w * dR * dR
                    a12 += w * dR * dZ
                    a22 += w * dZ * dZ
                    b1 += w * dR * df
                    b2 += w * dZ * df
                    npt += 1
            det = a11 * a22 - a12 * a12
            if npt < 3 or abs(det) <= 1e-24:
                return None
            return ((a22 * b1 - a12 * b2) / det,
                    (-a12 * b1 + a11 * b2) / det)

        for ix0 in range(nxp):
            for iy0 in range(nyp):
                sol = _accumulate(ix0, iy0, radius=1)
                if sol is None:
                    sol = _accumulate(ix0, iy0, radius=2)
                if sol is None:
                    continue
                g_r[ix0, iy0], g_z[ix0, iy0] = sol
        return g_r, g_z

    grad_te_r_g, grad_te_z_g = _grad_rz_lsq(Te_g)
    grad_ti_r_g, grad_ti_z_g = _grad_rz_lsq(Ti_g)

    # Electric field E = -grad(phi) on the B2 mesh (same least-squares path).
    # E_toroidal = 0 in axisymmetric SOLPS. No pressure-balance
    # fallback — consumers read E directly.
    po_g = _grid(po_data.reshape(ncell_flat, order='F'))
    grad_po_r_g, grad_po_z_g = _grad_rz_lsq(po_g)
    e_r_g = -grad_po_r_g
    e_z_g = -grad_po_z_g

    mesh_grad_te_r = grad_te_r_g.reshape(ncell_flat, order='F')
    mesh_grad_te_z = grad_te_z_g.reshape(ncell_flat, order='F')
    mesh_grad_ti_r = grad_ti_r_g.reshape(ncell_flat, order='F')
    mesh_grad_ti_z = grad_ti_z_g.reshape(ncell_flat, order='F')
    mesh_e_r = e_r_g.reshape(ncell_flat, order='F')
    mesh_e_z = e_z_g.reshape(ncell_flat, order='F')
    mesh_e_t = np.zeros_like(mesh_e_r)
    for arr in [mesh_grad_te_r, mesh_grad_te_z,
                mesh_grad_ti_r, mesh_grad_ti_z,
                mesh_e_r, mesh_e_z, mesh_e_t]:
        arr[~np.isfinite(arr)] = 0.0

    # -- Optional: read SOLPS-ITER fort.31 (EIRENE plasma input file) --
    # fort.31 is the per-B2-cell plasma SOLPS-ITER writes for EIRENE
    # coupling. Same grid as b2fstate, plus derived quantities B2 knows
    # about (|B|, B_pol, B_tor, potential, cell volumes, fluxes, Z_eff).
    # When present, we cross-check against our b2fstate-derived fields
    # and write the full fort.31 content into plasma.h5 as a /fort31/
    # group. Downstream consumers can then consume SOLPS-ITER-canonical
    # values instead of recomputing them from b2fstate+balance.nc.
    fort31_path = run_path / "fort.31"
    ft31 = None
    if fort31_path.exists():
        try:
            ft31 = read_fort31(fort31_path, nx=nx, ny=ny, ns_plasma=nion)
            print(f"fort.31 parsed OK (truncated={ft31.truncated}): "
                  f"{fort31_path}")
            # Cross-check ne / Te / Ti against our b2fstate-derived values
            # on the matching (nx+2, ny+2) grid. Flat-indexed comparison.
            def _rms_rel(a_ref, b_check):
                a_ref_f = a_ref.reshape(-1, order="F")
                b_check_f = b_check.reshape(-1, order="F")
                good = np.isfinite(a_ref_f) & np.isfinite(b_check_f) & (np.abs(a_ref_f) > 0)
                if not good.any():
                    return float("nan")
                return float(np.sqrt(np.mean(((b_check_f[good] - a_ref_f[good])
                                                / a_ref_f[good]) ** 2)))
            # fort.31 dnib[:,:,0] vs b2fstate na[:,:,main_sidx]
            na_ref = na if na.ndim == 2 else na[:, :, main_sidx]
            d_ne = _rms_rel(na_ref, ft31.dnib[:, :, 0])
            d_te = _rms_rel(te_eV, ft31.teb)
            d_ti = _rms_rel(ti_eV, ft31.tib)
            print(f"  fort.31 vs b2fstate RMS rel diff:  ne={d_ne:.2e}  "
                  f"Te={d_te:.2e}  Ti={d_ti:.2e}")
        except Exception as exc:
            print(f"WARNING: fort.31 parse failed ({exc}); skipping.")
            ft31 = None
    else:
        print(f"fort.31 not found at {fort31_path} — skipping.")

    # Evaluate the equilibrium magnetic field directly on the EIRENE mesh
    # vertices. Unlike plasma quantities, B should not inherit the
    # Sample B-field on the EIRENE mesh vertices directly. Plasma remains
    # piecewise-constant per mapped B2.5 cell, while non-native triangles
    # stay unmapped.
    # Equilibrium B-field is on a regular (R, Z) grid (jm x km), so we can
    # use RegularGridInterpolator instead of scipy.griddata. This drops ~20s
    # from the converter (was 86% of total runtime in profiling).
    mesh_vtx_pts = np.column_stack([mesh_vtx_r, mesh_vtx_z])
    equ_r = np.asarray(equ_dict["r"], dtype=np.float64)
    equ_z = np.asarray(equ_dict["z"], dtype=np.float64)
    mesh_vtx_br = _interp_regular_to_points(
        equ_r, equ_z, np.asarray(equ_dict["br"], dtype=np.float64), mesh_vtx_pts)
    mesh_vtx_bt = _interp_regular_to_points(
        equ_r, equ_z, np.asarray(equ_dict["bt"], dtype=np.float64), mesh_vtx_pts)
    mesh_vtx_bz = _interp_regular_to_points(
        equ_r, equ_z, np.asarray(equ_dict["bz"], dtype=np.float64), mesh_vtx_pts)
    mesh_vtx_bpol = np.sqrt(mesh_vtx_br**2 + mesh_vtx_bz**2)
    mesh_vtx_bmag = np.sqrt(mesh_vtx_bpol**2 + mesh_vtx_bt**2)

    b2_pts = np.column_stack([
        rc.reshape(ncell_flat, order="F"),
        zc.reshape(ncell_flat, order="F"),
    ])
    b2_br = _interp_regular_to_points(
        equ_r, equ_z, np.asarray(equ_dict["br"], dtype=np.float64), b2_pts)
    b2_bt = _interp_regular_to_points(
        equ_r, equ_z, np.asarray(equ_dict["bt"], dtype=np.float64), b2_pts)
    b2_bz = _interp_regular_to_points(
        equ_r, equ_z, np.asarray(equ_dict["bz"], dtype=np.float64), b2_pts)
    b2_bpol = np.sqrt(b2_br**2 + b2_bz**2)
    b2_bmag = np.sqrt(b2_bpol**2 + b2_bt**2)

    # ------------------------------------------------------------------
    # Per-species wall-face ion flux scatter for /wall_flux/ group.
    # ------------------------------------------------------------------
    # Plasma-source-agnostic wall-flux representation. Each B2 wall-
    # boundary face contributes one (R, Z) sample with per-species
    # gamma (m^-2 s^-1, +ve into wall), Te, Ti, area, and outward
    # normal. OpenEdge runtime queries this scatter at any wall-segment
    # midpoint via fix background's KDTree, so plasma.h5 stays
    # decoupled from the SPARTA wall.surf geometry.
    #
    # SOLPS staggered convention (verified vs b2plot b2ptrgl.F /
    # b2pwldw.F):
    #   inner target  ix=1:  -fnixb[0,    iy,  s] / area_face
    #   outer target  ix=nx: +fnixb[nx,   iy,  s] / area_face
    #   outer SOL     iy=ny: +fniyb[ix,   ny,  s] / area_face
    # The guard-face entries fnixb[nx+1, ...], fniyb[..., ny+1] are
    # zero in fort.31; the actual wall flux is one index earlier
    # (right face of last fluid cell). Matches what b2plot wlld
    # consumes when extracting per-segment loads.
    # ------------------------------------------------------------------
    wall_flux_data = None
    if ft31 is not None and ft31.fnixb is not None and ft31.fniyb is not None:
        wall_flux_data = build_wall_flux_solps(
            ft31, nx, ny, nion, crx4, cry4)
        # B-field at each wall-flux sample, sampled from the equilibrium
        # psi map. Consumers (compute surface/physical/sputter) IDW-
        # smooth this with K=3 and combine with the SPARTA segment's
        # own normal to get a smooth incidence angle. The per-triangle
        # B-field used at runtime otherwise gives cell-to-cell jitter
        # that propagates into Y(E,theta) near the sputter threshold.
        wf_pts = np.column_stack([wall_flux_data["r"], wall_flux_data["z"]])
        wall_flux_data["b_r"] = _interp_regular_to_points(
            equ_r, equ_z, np.asarray(equ_dict["br"], dtype=np.float64), wf_pts)
        wall_flux_data["b_z"] = _interp_regular_to_points(
            equ_r, equ_z, np.asarray(equ_dict["bz"], dtype=np.float64), wf_pts)
        wall_flux_data["b_t"] = _interp_regular_to_points(
            equ_r, equ_z, np.asarray(equ_dict["bt"], dtype=np.float64), wf_pts)
        gpeak = float(np.abs(wall_flux_data["gamma_i"]).max())
        print(f"/wall_flux/ built from fort.31 staggered fluxes: "
              f"N={wall_flux_data['r'].size}, "
              f"nspec={wall_flux_data['gamma_i'].shape[0]}, "
              f"peak |gamma_i| = {gpeak:.3e} m^-2 s^-1")
    else:
        print("fort.31 absent; /wall_flux/ will not be written")

    # -- Write plasma.h5 --
    with h5py.File(plasma_out, "w") as f:
        # Embedded equilibrium: raw .equ / GEQDSK-derived psi map on its
        # native jm×km grid, plus btf/rtf/psib. compute plasma/fields and
        # fix background use this to evaluate B at any (R, Z) and
        # psi_norm for the core-sink boundary; no separate .equ file is
        # needed at run time.
        f.create_dataset("equilibrium/r",    data=np.asarray(equ_dict["equ_r"], dtype=np.float64))
        f.create_dataset("equilibrium/z",    data=np.asarray(equ_dict["equ_z"], dtype=np.float64))
        f.create_dataset("equilibrium/psi",  data=np.asarray(equ_dict["equ_psi"], dtype=np.float64))
        f.create_dataset("equilibrium/btf",  data=float(equ_dict["btf"]))
        f.create_dataset("equilibrium/rtf",  data=float(equ_dict["rtf"]))
        f.create_dataset("equilibrium/psib", data=float(equ_dict["psib"]))
        for key in ("psi_axis", "rmagx", "zmagx"):
            if key in equ_dict:
                f.create_dataset(f"equilibrium/{key}", data=float(equ_dict[key]))

        # Multi-ion species metadata. Per-species plasma fields live on
        # the EIRENE triangulation under /mesh/ions/.
        sdt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("ion_species/names", data=np.array(ion_names, dtype=object), dtype=sdt)
        ion_elements = [re.sub(r'[+\-\d]+$', '', n) for n in ion_names]
        f.create_dataset("ion_species/elements", data=np.array(ion_elements, dtype=object), dtype=sdt)
        f.create_dataset("ion_species/spec_index", data=ion_indices.astype(np.int32))
        f.create_dataset("ion_species/main_ion_spec_index", data=np.array([int(ion_indices[main_k])], dtype=np.int32))
        f.create_dataset("ion_species/mass_amu", data=ion_masses)
        f.create_dataset("ion_species/charge_state_z", data=ion_charges)

        # Native B2.5 cell geometry and per-cell fields. These preserve the
        # SOLPS quadrilateral mesh and are the right choice for patch-style
        # visualization near the separatrix and X-point.
        f.create_dataset("b2/nx", data=np.array(nx, dtype=np.int32))
        f.create_dataset("b2/ny", data=np.array(ny, dtype=np.int32))
        f.create_dataset("b2/cell_polygons", data=cell_polys)
        f.create_dataset("b2/r_center", data=rc.reshape(ncell_flat, order="F"))
        f.create_dataset("b2/z_center", data=zc.reshape(ncell_flat, order="F"))
        f.create_dataset("b2/dens_e", data=mesh_ne)
        f.create_dataset("b2/temp_e", data=mesh_te)
        f.create_dataset("b2/dens_i", data=mesh_ni)
        f.create_dataset("b2/temp_i", data=mesh_ti)
        f.create_dataset("b2/parr_flow", data=mesh_upar)
        f.create_dataset("b2/grad_te_r", data=mesh_grad_te_r)
        f.create_dataset("b2/grad_te_z", data=mesh_grad_te_z)
        f.create_dataset("b2/grad_ti_r", data=mesh_grad_ti_r)
        f.create_dataset("b2/grad_ti_z", data=mesh_grad_ti_z)
        f.create_dataset("b2/e_r", data=mesh_e_r)
        f.create_dataset("b2/e_z", data=mesh_e_z)
        f.create_dataset("b2/e_t", data=mesh_e_t)
        f.create_dataset("b2/b_r", data=b2_br)
        f.create_dataset("b2/b_z", data=b2_bz)
        f.create_dataset("b2/b_t", data=b2_bt)
        f.create_dataset("b2/b_mag", data=b2_bmag)

        # SOLPS mesh triangulation for direct point-in-cell interpolation
        f.create_dataset("mesh/vtx_r", data=mesh_vtx_r)
        f.create_dataset("mesh/vtx_z", data=mesh_vtx_z)
        f.create_dataset("mesh/vtx_br", data=mesh_vtx_br)
        f.create_dataset("mesh/vtx_bz", data=mesh_vtx_bz)
        f.create_dataset("mesh/vtx_bt", data=mesh_vtx_bt)
        f.create_dataset("mesh/vtx_bmag", data=mesh_vtx_bmag)
        f.create_dataset("mesh/vtx_bpol", data=mesh_vtx_bpol)
        f.create_dataset("mesh/vtx_btor", data=mesh_vtx_bt)
        f.create_dataset("mesh/triangles", data=mesh_tri)
        f.create_dataset("mesh/cell_index", data=mesh_cell_idx)
        f.create_dataset("mesh/tri_source_kind", data=tri_source_kind)
        f.create_dataset("mesh/wall_gap_mask", data=wall_gap_mask)
        f.create_dataset("mesh/dens_e", data=mesh_ne)
        f.create_dataset("mesh/temp_e", data=mesh_te)
        f.create_dataset("mesh/dens_i", data=mesh_ni)
        f.create_dataset("mesh/temp_i", data=mesh_ti)
        f.create_dataset("mesh/parr_flow", data=mesh_upar)

        # Optional: plasma heat flux INTO plasma.h5 for fix droplet/evaporate.
        # Reuse the SOLPS-face-flux computation; |q| -> q_par, q_perp=0 (the
        # droplet model uses |q|=sqrt(q_par^2+q_perp^2)). Padded to the full
        # (nx+2)(ny+2) Fortran-flat layout so it aligns with mesh/temp_e.
        if heatflux != "none":
            from convert_solps_heatflux import compute_heatflux_cell_center
            _rc, _zc, _qmag = compute_heatflux_cell_center(
                run_path, method=heatflux, b2fgmtry_path=b2fgmtry_path)
            _qfull = np.zeros((nx + 2, ny + 2), dtype=np.float64)
            _qfull[1:-1, 1:-1] = np.nan_to_num(_qmag, nan=0.0,
                                               posinf=0.0, neginf=0.0)
            mesh_q_par = _qfull.reshape(ncell_flat, order="F")
            mesh_q_perp = np.zeros_like(mesh_q_par)
            f.create_dataset("mesh/q_par",  data=mesh_q_par)
            f.create_dataset("mesh/q_perp", data=mesh_q_perp)
            f.create_dataset("b2/q_par",  data=mesh_q_par)
            f.create_dataset("b2/q_perp", data=mesh_q_perp)
            print(f"mesh/q_par written (heatflux={heatflux}): peak "
                  f"{mesh_q_par.max():.3e} W/m^2, nonzero "
                  f"{int(np.count_nonzero(mesh_q_par))}/{mesh_q_par.size}")
        f.create_dataset("mesh/grad_te_r", data=mesh_grad_te_r)
        f.create_dataset("mesh/grad_te_z", data=mesh_grad_te_z)
        f.create_dataset("mesh/grad_ti_r", data=mesh_grad_ti_r)
        f.create_dataset("mesh/grad_ti_z", data=mesh_grad_ti_z)
        f.create_dataset("mesh/e_r", data=mesh_e_r)
        f.create_dataset("mesh/e_z", data=mesh_e_z)
        f.create_dataset("mesh/e_t", data=mesh_e_t)
        f.create_dataset("mesh/ions/dens", data=mesh_ions_dens)
        f.create_dataset("mesh/ions/temp", data=mesh_ions_temp)
        f.create_dataset("mesh/ions/parr_flow", data=mesh_ions_upar)

        # Per-cell wall face area (m^2, toroidally integrated). Nonzero for
        # B2 boundary cells only. Downstream consumers (emit/surf/recycle)
        # multiply by ne*cs*sin(alpha_B) to get the Bohm wall flux that
        # drives wall recycling at each boundary cell.
        f.create_dataset("mesh/wall_face_area", data=mesh_wall_face_area)

        # Plasma-source-agnostic wall-flux scatter. See _wall_flux.py
        # for the schema. Consumers (e.g. compute surface/physical/
        # sputter) query this via fix background's KDTree at any
        # SPARTA segment (R, Z) — no per-segment cell mapping needed.
        if wall_flux_data is not None:
            write_wall_flux_h5(f, wall_flux_data,
                               source="SOLPS-ITER",
                               extraction="fort31",
                               species_names=ion_names)

        # Per-wall-segment topological mapping to B2 boundary cells:
        #   mesh_wall_surf_cell[iseg]: flat index of the B2 boundary cell
        #       that dominantly owns SPARTA wall segment iseg (the B2 cell
        #       contributing the largest face-area into this segment).
        #   mesh_wall_surf_area[iseg]: summed B2 boundary face area from
        #       all faces that chose this segment. Conserves the Bohm
        #       flux budget across a coarser SPARTA wall.
        f.create_dataset("mesh/wall_surf_cell", data=mesh_wall_surf_cell)
        f.create_dataset("mesh/wall_surf_area", data=mesh_wall_surf_area)

        # ---- Per-B2-cell fields sourced from fort.31 (SOLPS-ITER
        # canonical). Stored length = (nx+2)*(ny+2), same convention as
        # the existing /mesh/{dens_e, temp_e, ...} datasets so consumers
        # get triangle-level values via /mesh/cell_index in a uniform
        # way. Native triangles follow the EIRENE pattern:
        # triangle -> fort.35 B2 cell -> fort.31 piecewise-constant
        # plasma. Non-native triangles remain unmapped
        # (mesh/cell_index == -1) and therefore stay on a vacuum floor.
        if ft31 is not None:
            if ft31.pob is not None:
                f.create_dataset("mesh/pob",
                                 data=ft31.pob.reshape(-1, order="F"))
            if ft31.volb is not None:
                f.create_dataset("mesh/volb",
                                 data=ft31.volb.reshape(-1, order="F"))
            if ft31.rrb is not None:
                f.create_dataset("mesh/rrb",
                                 data=ft31.rrb.reshape(-1, order="F"))
            if ft31.zib is not None:
                f.create_dataset("mesh/zeff",
                                 data=ft31.zib[:, :, 0].reshape(-1, order="F"))

        # ---- /fort31 group: SOLPS-ITER's per-B2-cell plasma (for EIRENE)
        # Written verbatim from fort.31 so downstream consumers can pull
        # SOLPS-ITER-canonical values (|B|, B_pol, B_tor, potential, cell
        # volumes, fluxes, Z_eff) instead of recomputing them.
        # Grid shape (nx+2, ny+2) mirrors b2fstate exactly.
        if ft31 is not None:
            f.create_group("fort31")
            f.attrs  # no-op to silence linters
            f31g = f["fort31"]
            f31g.attrs["source_file"] = ft31.file
            f31g.attrs["nx"] = ft31.nx
            f31g.attrs["ny"] = ft31.ny
            f31g.attrs["ns_plasma"] = ft31.ns_plasma
            f31g.attrs["truncated"] = bool(ft31.truncated)
            # Required (always-present) blocks
            for name, arr in [
                ("dnib", ft31.dnib), ("uub", ft31.uub), ("vvb", ft31.vvb),
                ("wwb", ft31.wwb), ("upb", ft31.upb),
                ("fnixb", ft31.fnixb), ("fniyb", ft31.fniyb),
                ("uudiab", ft31.uudiab), ("vvdiab", ft31.vvdiab),
                ("teb", ft31.teb), ("tib", ft31.tib), ("prb", ft31.prb),
                ("rrb", ft31.rrb),
                ("feixb", ft31.feixb), ("feiyb", ft31.feiyb),
                ("feexb", ft31.feexb), ("feeyb", ft31.feeyb),
            ]:
                f31g.create_dataset(name, data=arr)
            # Optional blocks (absent in truncated fort.31 builds)
            for name, arr in [
                ("pob", ft31.pob), ("volb", ft31.volb),
                ("bfeldb", ft31.bfeldb), ("bpolb", ft31.bpolb),
                ("bradb", ft31.bradb), ("btorb", ft31.btorb),
                ("delta_sheathxb", ft31.delta_sheathxb),
                ("delta_sheathyb", ft31.delta_sheathyb),
                ("zib", ft31.zib),
            ]:
                if arr is not None:
                    f31g.create_dataset(name, data=arr)

    # B-field is embedded in plasma.h5 above (br/bt/bz datasets), so no
    # separate bfield.h5 is needed. compute plasma/fields reads B from
    # plasma.h5 directly.

    # -- Optional wall geometry --
    # Preferred path: wall.surf built from EIRENE triangulation boundary
    # edges (already written above when mesh_wall_surf_cell was populated).
    # Fallback: legacy simplified walk of mesh.extra. Only used when the
    # EIRENE-based path couldn't produce edges (e.g. no fort.33/34/35).
    if wall_out is not None and mesh_wall_surf_cell.size == 0:
        mesh_path = mesh_extra if mesh_extra is not None else (run_path / "mesh.extra")
        _write_sparta_wall_from_mesh_extra(mesh_path, wall_out, geometry=geometry)
        print(f"Wrote wall (legacy mesh.extra walk): {wall_out}")

    print(f"Wrote plasma: {plasma_out}")
    print(f"Mesh extent: R=[{rc.min():.4f},{rc.max():.4f}], "
          f"Z=[{zc.min():.4f},{zc.max():.4f}]  "
          f"(B2 cell centers; mesh-format output)")
    print(f"Ion species ({nion}): {ion_names}")
    print(f"Charge states: {ion_charges.tolist()}")

    # -- Optional plots --
    if plot:
        # B-field is no longer interpolated onto B2 cell centroids here
        # (converter writes only the equilibrium psi/btf/rtf, not a
        # per-cell br). Pass zeros for the br panel — diagnostic plots
        # should be rebuilt to read from /equilibrium/ directly.
        plot_plasma_h5_b2(
            cell_polys,
            rc.reshape(ncell_flat, order="F"),
            zc.reshape(ncell_flat, order="F"),
            nx,
            ny,
            ne,
            te_eV,
            main_dens_raw,
            ti_eV,
            main_upar_raw,
            np.zeros_like(ne),
            plot_prefix or Path("convert_solps_plasma"),
        )

def _best_ordered_b2_polys(polys, r_center, z_center, nx, ny):
    """Choose polygon ordering that best matches stored B2 cell centers."""
    polys = np.asarray(polys, dtype=float)
    centers = np.column_stack([np.asarray(r_center, dtype=float), np.asarray(z_center, dtype=float)])

    candidates = [polys]
    ncell = (nx + 2) * (ny + 2)
    if polys.shape[0] == ncell:
        candidates.append(polys.reshape(nx + 2, ny + 2, 4, 2, order='C').reshape(-1, 4, 2, order='F'))
        candidates.append(polys.reshape(nx + 2, ny + 2, 4, 2, order='F').reshape(-1, 4, 2, order='C'))

    best = candidates[0]
    best_score = np.inf
    for cand in candidates:
        cent = np.nanmean(cand, axis=1)
        score = float(np.nanmedian(np.hypot(cent[:, 0] - centers[:, 0], cent[:, 1] - centers[:, 1])))
        if score < best_score:
            best = cand
            best_score = score
    return best


def _poly_panel(ax, polys, values, title, cmap, *, log=False, symmetric=False):
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    from matplotlib.collections import PolyCollection

    vals = np.asarray(values, dtype=float).reshape(-1)
    valid_poly = np.all(np.isfinite(polys), axis=(1, 2))
    vals = vals[valid_poly]
    poly_use = polys[valid_poly]

    if log:
        vals = np.where(vals > 0.0, vals, np.nan)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            finite = np.array([1.0])
        vmin = float(np.nanpercentile(finite, 1))
        vmax = float(np.nanpercentile(finite, 99))
        if not np.isfinite(vmin) or vmin <= 0.0:
            vmin = max(float(np.nanmin(finite)), 1e-30)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin * 10.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif symmetric:
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            finite = np.array([0.0])
        vlim = float(np.nanpercentile(np.abs(finite), 99))
        if not np.isfinite(vlim) or vlim <= 0.0:
            vlim = 1.0
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    else:
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            finite = np.array([0.0])
        vmin = float(np.nanpercentile(finite, 1))
        vmax = float(np.nanpercentile(finite, 99))
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)

    coll = PolyCollection(poly_use, array=vals, cmap=cmap, norm=norm,
                          edgecolors='none', linewidths=0.0, antialiased=False)
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(r'$R$ (m)')
    ax.set_ylabel(r'$Z$ (m)')
    return coll


def plot_plasma_h5_b2(cell_polys, r_center, z_center, nx, ny, ne, te, ni, ti, upar, br, prefix):
    import matplotlib.pyplot as plt

    polys = _best_ordered_b2_polys(cell_polys, r_center, z_center, nx, ny)

    panels = [
        ('ne', ne, True,  r'$n_e$ (m$^{-3}$)', 'viridis', False),
        ('Te', te, True,  r'$T_e$ (eV)', 'inferno', False),
        ('ni', ni, True,  r'$n_i$ (m$^{-3}$)', 'viridis', False),
        ('Ti', ti, True,  r'$T_i$ (eV)', 'inferno', False),
        ('upar', upar, False, r'$u_\parallel$ (m s$^{-1}$)', 'RdBu_r', True),
        ('Br', br, False, r'$B_r$ (T)', 'RdBu_r', True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for ax, (_, val, log, label, cmap, symmetric) in zip(axes.flat, panels):
        coll = _poly_panel(ax, polys, val, label, cmap, log=log, symmetric=symmetric)
        cb = fig.colorbar(coll, ax=ax, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(f"{prefix}_plasma.png", dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote B2 plot: {prefix}_plasma.png")


# ==========================================================================
# CLI
# ==========================================================================

def _build_parser():
    p = argparse.ArgumentParser(
        description="Convert SOLPS run to OpenEdge plasma.h5 (no quixote, B-field embedded)"
    )
    p.add_argument("run_path", type=Path, help="SOLPS run directory (contains b2fgmtry, b2fstate)")
    p.add_argument("--plasma-out", type=Path, default=Path("plasma.h5"))
    p.add_argument("--nr", type=int, default=300)
    p.add_argument("--nz", type=int, default=300)
    p.add_argument("--rmin", type=float, default=None)
    p.add_argument("--rmax", type=float, default=None)
    p.add_argument("--zmin", type=float, default=None)
    p.add_argument("--zmax", type=float, default=None)
    p.add_argument("--gfile", type=Path, default=None, help="GEQDSK file for B-field (preferred)")
    p.add_argument("--equ-file", type=Path, default=None, help=".equ file for B-field")
    p.add_argument("--plot", action="store_true",
                   help="Write the default native-B2 diagnostic plot (plot_plasma_h5_b2 style).")
    p.add_argument("--plot-prefix", type=Path, default=None)
    p.add_argument("--wall-out", type=Path, default=None, help="Output SPARTA wall file from mesh.extra")
    p.add_argument("--wall-in", type=Path, default=None,
                   help="Input SPARTA wall.surf to map against (e.g., a custom "
                   "test strip). Builds mesh/wall_surf_cell from this geometry "
                   "instead of auto-generating one. Does not overwrite the file.")
    p.add_argument("--mesh-extra", type=Path, default=None)
    p.add_argument("--b2fgmtry", type=Path, default=None, help="Path to b2fgmtry (if not in run_path)")
    p.add_argument("--b2fstate", type=Path, default=None,
                   help="Path to b2fstate (defaults to run_path/b2fstate; use "
                        "b2fstati when the run didn't write a converged b2fstate)")
    p.add_argument("--wall-source", type=str, default="auto",
                   choices=["auto", "mesh-extra", "eirene", "b2"],
                   help=("Which wall-construction path to use. 'auto' picks "
                         "mesh-extra if available, else eirene, else b2. "
                         "'mesh-extra' + 'b2' are SOLPS-native and do NOT "
                         "depend on EIRENE fort.33/34/35 files."))
    p.add_argument("--geometry", choices=["cart", "axi"], default="cart",
                   help="SPARTA wall.surf slot layout. cart (default): "
                        "x=R, y=Z; pairs with 'boundary o o p'. axi: "
                        "x=Z, y=R; pairs with 'boundary o ao p'.")
    p.add_argument("--heatflux", default="none",
                   choices=["none", "total", "jeremy_no_jv", "jeremy_total",
                            "vector_mag"],
                   help="Also write plasma heat flux INTO plasma.h5 as "
                        "mesh/q_par (=|q| from SOLPS face fluxes) + mesh/q_perp=0, "
                        "for fix droplet/evaporate. 'total' = |fht/sxprl| parallel "
                        "heat flux (recommended). Default none.")
    p.add_argument("--core-out", type=Path, default=None,
                   help="If given, also write a SPARTA surface file tracing "
                        "the psi_norm=<--psi-norm-core> contour around the "
                        "magnetic axis. Use as the core-absorb boundary "
                        "(read_surf + surf_collide vanish).")
    p.add_argument("--psi-norm-core", type=float, default=0.90,
                   help="Normalized psi level for --core-out. Default 0.90 "
                        "keeps the contour safely inside the wall near the "
                        "X-point; 0.95 often dips into the private-flux "
                        "region and crosses the divertor wall.")
    p.add_argument("--core-preview", type=Path, default=None,
                   help="Optional PNG preview of the core contour.")
    return p


def main():
    args = _build_parser().parse_args()
    convert_solps_to_openedge(
        run_path=args.run_path,
        plasma_out=args.plasma_out,
        nr=args.nr, nz=args.nz,
        rmin=args.rmin, rmax=args.rmax,
        zmin=args.zmin, zmax=args.zmax,
        gfile=args.gfile,
        equ_file=args.equ_file,
        plot=args.plot,
        plot_prefix=args.plot_prefix,
        wall_out=args.wall_out,
        wall_in=args.wall_in,
        mesh_extra=args.mesh_extra,
        b2fgmtry_path=args.b2fgmtry,
        b2fstate_path=args.b2fstate,
        wall_source=args.wall_source,
        geometry=args.geometry,
        heatflux=args.heatflux,
    )

    # Optional: trace the psi_norm=<level> contour from the equilibrium
    # embedded in plasma.h5 and write it as a SPARTA surface, matching the
    # convention used by convert_s3x_plasma.py.
    if args.core_out:
        import sys as _sys, os as _os
        _sys.path.insert(0,
            _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
        from extract_psi_contour import write_core_surf_from_plasma_h5
        write_core_surf_from_plasma_h5(
            args.plasma_out,
            args.core_out,
            psi_norm=args.psi_norm_core,
            preview=args.core_preview,
            verbose=True,
        )


if __name__ == "__main__":
    main()
