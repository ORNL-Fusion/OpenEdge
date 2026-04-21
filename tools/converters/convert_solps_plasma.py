#!/usr/bin/env python3
"""
SOLPS-ITER -> OpenEdge converter (no quixote dependency).

Reads SOLPS binary text files (b2fgmtry, b2fstate) directly using the
*cf: header format, following Jeremy Lore / Jae-Sun Park's SOLPS routines.

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
import sys
import warnings
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from scipy.interpolate import griddata


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
    return polys.reshape(-1, 4, 2)


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

    psi_arr = data.get("psi", data.get("psirz", None))
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
    br = dpsidZ / safe_r
    bz = -dpsidR / safe_r
    bt = (float(data["bcentr"]) * float(data["rcentr"])) / safe_r
    psib = float(data.get("sibdry", data.get("ssibry", 0.0)))
    return {
        "r": rs, "z": zs, "br": br, "bt": bt, "bz": bz,
        "equ_r": rs.copy(), "equ_z": zs.copy(), "equ_psi": flux2d,
        "btf": float(data["bcentr"]),
        "rtf": float(data["rcentr"]),
        "psib": psib,
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

    # Element symbols by atomic number
    element_sym = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
        8: "O", 10: "Ne", 18: "Ar", 36: "Kr", 54: "Xe", 74: "W",
    }

    for i in range(ns):
        z_atom = int(round(zn[i])) if i < len(zn) else 0
        z_min = int(round(zamin[i])) if i < len(zamin) else 0
        z_max = int(round(zamax[i])) if i < len(zamax) else 0
        mass = float(am[i]) if i < len(am) else 0.0
        sym = element_sym.get(z_atom, f"Z{z_atom}")

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
        sym = element_sym.get(z_atom, f"Z{z_atom}")
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


def _write_sparta_wall_from_mesh_extra(mesh_extra: Path, wall_out: Path, tol: float = 1e-8):
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

    wall_out.parent.mkdir(parents=True, exist_ok=True)
    with wall_out.open("w", encoding="utf-8") as f:
        f.write("surface geometry\n\n")
        f.write(f"{n} points\n{n} lines\n\nPoints\n\n")
        for i in range(n):
            f.write(f"{i+1} {r[i]:.12g} {z[i]:.12g}\n")
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
        b2_ix, b2_iy: (ntri,) B2.5 cell indices (1-based), or -1 if not in B2.5
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

    # fort.35: B2.5 cell mapping (last 2 columns = ix, iy)
    b2_ix = np.full(ntri, -1, dtype=np.int32)
    b2_iy = np.full(ntri, -1, dtype=np.int32)
    with ft35.open("r") as f:
        _ = int(f.readline().strip())
        for i, line in enumerate(f):
            parts = line.split()
            if len(parts) >= 12:
                ix_val = int(parts[10])
                iy_val = int(parts[11])
                if ix_val >= 0 and iy_val >= 0:
                    b2_ix[i] = ix_val
                    b2_iy[i] = iy_val

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
    mesh_extra: Path | None = None,
    b2fgmtry_path: Path | None = None,
    wall_source: str = "auto",
) -> None:
    """
    Convert SOLPS run directory to OpenEdge plasma.h5 (B-field embedded).

    Reads b2fgmtry (geometry) and b2fstate (plasma state) directly —
    no quixote dependency.
    """
    b2fgmtry_file = b2fgmtry_path if b2fgmtry_path else run_path / "b2fgmtry"
    b2fstate_file = run_path / "b2fstate"

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

    r, z, rr, zz = _regular_grid(rc, zc, nr, nz, rmin, rmax, zmin, zmax)
    src_pts = np.column_stack((rc.reshape(-1), zc.reshape(-1)))
    tgt_pts = np.column_stack((rr.reshape(-1), zz.reshape(-1)))

    # -- Electron fields --
    # b2fstate stores te/ti in eV (some SOLPS versions store in Joules — check)
    ne = b2f_extract("ne", b2fstate_file)  # (nx+2, ny+2)
    te = b2f_extract("te", b2fstate_file)  # (nx+2, ny+2)
    ti = b2f_extract("ti", b2fstate_file)  # (nx+2, ny+2)

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
        # SOLPS labels all hydrogen isotopes generically as "H"; the
        # actual isotope is set by atomic mass in b2fparam. Promote to
        # "D" (mass ≈ 2) or "T" (mass ≈ 3) so downstream sees the
        # correct isotope name.
        raw_name = species_names[sidx]
        m = masses_amu[sidx]
        if raw_name.startswith("H") and 1.8 < m < 2.3:
            fixed_name = "D" + raw_name[1:]
        elif raw_name.startswith("H") and 2.8 < m < 3.3:
            fixed_name = "T" + raw_name[1:]
        else:
            fixed_name = raw_name
        ion_names.append(fixed_name)
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
    # b2_ix, b2_iy are 1-based B2.5 indices.
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
    print(f"Eirene mesh -> B2.5 mapping: {has_cell_native.sum()} triangles "
          f"with plasma, {(~has_cell_native).sum()} vacuum/PFR (native)")

    # Centroid bookkeeping for the projections below.
    centroid_r = mesh_vtx_r[mesh_tri].mean(axis=1)
    centroid_z = mesh_vtx_z[mesh_tri].mean(axis=1)

    # Project sheath-edge plasma outward onto vacuum/PFR triangles.
    # Sheath-edge cells: outermost ring of B2 REAL cells (ix=1/nx, iy=1/ny),
    # which physically drive EIRENE's recycling strata. Restricting the
    # projection target to this ring (vs. any nearest B2 cell) puts the
    # SOLPS boundary-layer plasma at the wall, not the volume-averaged SOL.
    if (~has_cell_native).any() and has_cell_native.any():
        from scipy.spatial import cKDTree
        native_idx = np.where(has_cell_native)[0]
        c_flat = mesh_cell_idx[native_idx]
        iy = c_flat // (nx + 2)
        ix = c_flat - iy * (nx + 2)
        is_sheath = ((ix == 1) | (ix == nx) | (iy == 1) | (iy == ny))
        sheath_src = native_idx[is_sheath]
        if sheath_src.size == 0:
            sheath_src = native_idx   # fallback: any native cell

        dst_idx = np.where(~has_cell_native)[0]
        tree = cKDTree(np.column_stack([centroid_r[sheath_src],
                                        centroid_z[sheath_src]]))
        _, j = tree.query(np.column_stack([centroid_r[dst_idx],
                                           centroid_z[dst_idx]]))
        mesh_cell_idx[dst_idx] = mesh_cell_idx[sheath_src[j]]
        print(f"Eirene mesh -> B2.5 mapping: {len(dst_idx)} vacuum/PFR "
              f"triangles projected to nearest sheath-edge B2 cell "
              f"({sheath_src.size} sheath-edge cells available)")

    has_cell = mesh_cell_idx >= 0

    # ------------------------------------------------------------------
    # Extend the EIRENE triangulation out to the mesh.extra wall polygon.
    # ------------------------------------------------------------------
    # fort.33/34/35 triangulation typically stops a few cm shy of the
    # real wall (EIRENE's "neighbour polygon"). Without an explicit
    # extension, particles in the ~5 cm gap fall back to a regular-grid
    # interpolation that is zero outside the convex hull — ionization
    # silently switches off there. Extend by combining the EIRENE
    # outer-boundary vertices with the wall polygon and re-triangulating,
    # then assign each new triangle to the nearest B2 sheath cell so the
    # SOLPS sheath plasma is propagated all the way to the wall. This is
    # the same trick SOLPS-ITER's coupled mode does internally (vacuum
    # triangles with nearest-sheath plasma assignment).
    mesh_extra_path_local = mesh_extra if mesh_extra is not None else (run_path / "mesh.extra")
    if mesh_extra_path_local.exists() and has_cell.any():
        try:
            from scipy.spatial import Delaunay, cKDTree
            from matplotlib.path import Path as MplPath

            wall_poly = _parse_mesh_extra_to_polygon(mesh_extra_path_local)
            wall_r_poly = wall_poly[:-1, 0]
            wall_z_poly = wall_poly[:-1, 1]

            # Densify the wall polygon a bit so Delaunay produces
            # reasonably-sized triangles near the wall.
            dens_r, dens_z = [], []
            npts = len(wall_r_poly)
            for i in range(npts):
                j = (i + 1) % npts
                dens_r.append(wall_r_poly[i]); dens_z.append(wall_z_poly[i])
                dr = wall_r_poly[j] - wall_r_poly[i]
                dz = wall_z_poly[j] - wall_z_poly[i]
                seg = float(np.hypot(dr, dz))
                # target ~2 cm spacing between wall samples
                n_ins = max(1, int(seg / 0.02)) - 1
                for k in range(n_ins):
                    t = (k + 1) / (n_ins + 1)
                    dens_r.append(wall_r_poly[i] + t * dr)
                    dens_z.append(wall_z_poly[i] + t * dz)
            wall_r_dense = np.asarray(dens_r, dtype=np.float64)
            wall_z_dense = np.asarray(dens_z, dtype=np.float64)

            # Find boundary vertices of the existing triangulation
            # (edges that appear in exactly one triangle).
            edge_count = {}
            for tri_row in mesh_tri:
                for a, b in [(tri_row[0], tri_row[1]),
                             (tri_row[1], tri_row[2]),
                             (tri_row[2], tri_row[0])]:
                    e = (int(min(a, b)), int(max(a, b)))
                    edge_count[e] = edge_count.get(e, 0) + 1
            boundary_verts = sorted({v for (a, b), c in edge_count.items()
                                       if c == 1 for v in (a, b)})
            bv_r = mesh_vtx_r[boundary_verts]
            bv_z = mesh_vtx_z[boundary_verts]

            # Delaunay on (EIRENE boundary vertices + densified wall).
            n_boundary = len(boundary_verts)
            combined_r = np.concatenate([bv_r, wall_r_dense])
            combined_z = np.concatenate([bv_z, wall_z_dense])
            pts2d = np.column_stack([combined_r, combined_z])
            dtri = Delaunay(pts2d)

            # Keep only triangles whose centroid is inside the wall
            # polygon (filters out Delaunay's big outer triangles) AND
            # outside the existing mesh (only near-wall triangles).
            wall_path = MplPath(wall_poly)
            centroids = pts2d[dtri.simplices].mean(axis=1)
            inside_wall = wall_path.contains_points(centroids)

            # "Outside existing mesh" check: use the EIRENE boundary as
            # a polygon. Nearest-boundary-point distance is a decent
            # proxy — centroids far enough from the existing mesh
            # belong in the new annulus region.
            tree_boundary = cKDTree(np.column_stack([bv_r, bv_z]))
            d_to_bdry, _ = tree_boundary.query(centroids)
            min_gap = 0.005   # 5 mm from EIRENE boundary -> new region

            # Additionally reject triangles that straddle the existing
            # mesh (all 3 vertices from EIRENE boundary): Delaunay can
            # fill the inner void if boundaries are concave.
            new_mask = (inside_wall) & (d_to_bdry > min_gap)
            # require at least one wall vertex in each kept triangle
            uses_wall = np.any(dtri.simplices >= n_boundary, axis=1)
            new_mask &= uses_wall

            new_simplices = dtri.simplices[new_mask]

            if len(new_simplices) > 0:
                # Remap Delaunay indices back to mesh_vtx_r/z + wall
                # arrays, after appending the new vertices.
                n_mesh_old = len(mesh_vtx_r)
                mesh_vtx_r = np.concatenate([mesh_vtx_r, wall_r_dense])
                mesh_vtx_z = np.concatenate([mesh_vtx_z, wall_z_dense])

                remap = np.empty(len(pts2d), dtype=np.int64)
                # first n_boundary entries are EIRENE boundary vertices
                remap[:n_boundary] = np.asarray(boundary_verts, dtype=np.int64)
                # remaining entries are the new wall-polygon vertices,
                # which we just appended starting at n_mesh_old
                remap[n_boundary:] = n_mesh_old + np.arange(len(wall_r_dense))
                new_tri_remapped = remap[new_simplices]

                # Map each new triangle to nearest B2 sheath cell.
                native_idx = np.where(has_cell_native)[0]
                c_flat = mesh_cell_idx[native_idx]
                iy_ = c_flat // (nx + 2)
                ix_ = c_flat - iy_ * (nx + 2)
                is_sheath_ = ((ix_ == 1) | (ix_ == nx) | (iy_ == 1) | (iy_ == ny))
                sheath_src_ = native_idx[is_sheath_]
                if sheath_src_.size == 0:
                    sheath_src_ = native_idx
                tree_sh = cKDTree(np.column_stack([centroid_r[sheath_src_],
                                                    centroid_z[sheath_src_]]))
                new_centroids_r = mesh_vtx_r[new_tri_remapped].mean(axis=1)
                new_centroids_z = mesh_vtx_z[new_tri_remapped].mean(axis=1)
                _, j_s = tree_sh.query(np.column_stack([new_centroids_r,
                                                        new_centroids_z]))
                new_cell_idx = mesh_cell_idx[sheath_src_[j_s]]

                mesh_tri = np.vstack([mesh_tri, new_tri_remapped.astype(mesh_tri.dtype)])
                mesh_cell_idx = np.concatenate([mesh_cell_idx,
                                                 new_cell_idx.astype(mesh_cell_idx.dtype)])
                centroid_r = np.concatenate([centroid_r, new_centroids_r])
                centroid_z = np.concatenate([centroid_z, new_centroids_z])
                has_cell = mesh_cell_idx >= 0
                ntri_eirene = len(mesh_tri)
                print(f"Mesh extension: added {len(new_simplices)} triangles "
                      f"to close the EIRENE -> wall gap "
                      f"({n_boundary} boundary + {len(wall_r_dense)} wall vertices)")
            else:
                print("Mesh extension: no triangles needed "
                      "(EIRENE boundary already reaches wall).")
        except Exception as e:
            print(f"WARNING: mesh extension to wall failed: {e}")

    # ------------------------------------------------------------------
    # Per-cell wall face area for B2 boundary cells.
    # ------------------------------------------------------------------
    # SOLPS corner ordering (inferred from _cell_polygons_from_corners):
    #   c=0 lower-left  (-x, -y), c=1 lower-right (+x, -y),
    #   c=3 upper-right (+x, +y), c=2 upper-left  (-x, +y)
    # External face for each boundary row of real cells:
    #   iy=1  (lower target): face = edge 0-1   (-y side)
    #   iy=ny (upper target): face = edge 2-3   (+y side)
    #   ix=1  (inner radial): face = edge 0-2   (-x side)
    #   ix=nx (outer radial): face = edge 1-3   (+x side)
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

    # Face areas are populated only for OUTER-wall boundary cells:
    # ix=nx (main chamber), iy=1 and iy=ny (divertor targets). The
    # ix=1 radial boundary is the core-side flux surface, not a wall,
    # so it gets face_area=0 and no recycling emission.
    for ix_c in range(nx + 2):
        for iy_c in range(ny + 2):
            area = 0.0
            if iy_c == 1:     area += _edge_area(ix_c, iy_c, 0, 1)
            if iy_c == ny:    area += _edge_area(ix_c, iy_c, 2, 3)
            if ix_c == nx:    area += _edge_area(ix_c, iy_c, 1, 3)
            if area > 0.0:
                c_flat = iy_c * (nx + 2) + ix_c
                mesh_wall_face_area[c_flat] = area
    n_bc = int((mesh_wall_face_area > 0).sum())
    print(f"B2 wall face areas (outer walls only, skipping ix=1 core): "
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
        # SOLPS is an axisymmetric code. Write wall.surf in SPARTA's
        # native axisymmetric layout: column 1 = x = Z (axial), column 2
        # = y = R (radial). Pairs with `boundary o ao p`, `create_box ...
        # 0 R_max ...`, where SPARTA computes true cylindrical cell
        # volumes and 2*pi*R*L surface areas internally.
        wall_out.parent.mkdir(parents=True, exist_ok=True)
        with wall_out.open("w", encoding="utf-8") as f:
            f.write("surface geometry\n\n")
            f.write(f"{len(me_pts)} points\n{len(me_segs)} lines\n\nPoints\n\n")
            for i, (rv, zv) in enumerate(me_pts):
                f.write(f"{i+1} {zv:.12g} {rv:.12g}\n")
            f.write("\nLines\n\n")
            for i, (a, b) in enumerate(me_segs):
                f.write(f"{i+1} {a+1} {b+1}\n")
        print(f"Wrote wall.surf from mesh.extra (axi: x=Z, y=R): {wall_out}")
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
                f.write(f"{i+1} {zv:.12g} {rv:.12g}\n")
            f.write("\nLines\n\n")
            for i, (va, vb) in enumerate(wall_edges):
                f.write(f"{i+1} {vmap[va]} {vmap[vb]}\n")
        print(f"Wrote EIRENE-consistent wall (axi: x=Z, y=R): {wall_out} "
              f"({len(wall_edges)} segments, {len(unique_rz)} unique pts "
              f"after snap)")
        mesh_wall_surf_cell = np.asarray(wall_edge_cells, dtype=np.int32)
        mesh_wall_surf_area = mesh_wall_face_area[mesh_wall_surf_cell].astype(
            np.float64)
    # else: mesh_wall_surf_cell/area were already set by the mesh-extra
    # path (or remain empty if neither ran).

    # Legacy path below — keep in case wall_out wasn't passed but an
    # existing wall.surf is already present; we still write the mapping.
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

            # Each B2 boundary face chooses its closest wall segment.
            # Wall segments accumulate flux by summing the face areas of
            # every B2 face that picked them. This conserves the total
            # Bohm-flux budget from the B2 boundary onto the SPARTA wall.
            from scipy.spatial import cKDTree
            tree = cKDTree(np.column_stack([seg_rmid, seg_zmid]))
            _, seg_for_face = tree.query(np.column_stack([face_rc, face_zc]))

            # Aggregate face area onto each wall segment, and assign the
            # DOMINANT cell (largest-area contributor) as the segment's
            # owner in mesh_wall_surf_cell[i].
            nseg_wall = len(S)
            seg_dominant_cell = np.full(nseg_wall, -1, dtype=np.int32)
            seg_dominant_area = np.zeros(nseg_wall, dtype=np.float64)
            seg_total_area = np.zeros(nseg_wall, dtype=np.float64)
            face_area = mesh_wall_face_area[face_cell]
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
    # Compute central differences in structured (ix, iy) index space, then
    # convert to (R, Z) via the cell-center Jacobian J = [[∂R/∂ix, ∂R/∂iy],
    # [∂Z/∂ix, ∂Z/∂iy]]. fix_plasma_data reads these flat arrays directly
    # via mesh_cell_at() + mesh_grad_*_r/z[cell] — no regular-grid
    # interpolation or runtime FD needed.
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

    # Central-difference operator (clamped at boundaries via one-sided)
    def _grad2d(field):
        dix = np.zeros_like(field)
        diy = np.zeros_like(field)
        dix[1:-1, :] = 0.5 * (field[2:, :] - field[:-2, :])
        dix[0, :]    = field[1, :]  - field[0, :]
        dix[-1, :]   = field[-1, :] - field[-2, :]
        diy[:, 1:-1] = 0.5 * (field[:, 2:] - field[:, :-2])
        diy[:, 0]    = field[:, 1]  - field[:, 0]
        diy[:, -1]   = field[:, -1] - field[:, -2]
        return dix, diy

    dR_dix, dR_diy = _grad2d(Rc)
    dZ_dix, dZ_diy = _grad2d(Zc)
    det_J = dR_dix * dZ_diy - dR_diy * dZ_dix
    # Avoid divide-by-zero at degenerate cells (ix=0/nxp-1 guard rows).
    det_J_safe = np.where(np.abs(det_J) > 1e-30, det_J, 1.0)

    def _grad_rz(field_g):
        dfdi, dfdj = _grad2d(field_g)
        # [d/dR, d/dZ] = J^{-1} [d/dix, d/diy]
        # J^{-1} = (1/det) [[dZ_diy, -dR_diy], [-dZ_dix, dR_dix]]
        g_r = ( dZ_diy * dfdi - dR_diy * dfdj) / det_J_safe
        g_z = (-dZ_dix * dfdi + dR_dix * dfdj) / det_J_safe
        g_r[np.abs(det_J) <= 1e-30] = 0.0
        g_z[np.abs(det_J) <= 1e-30] = 0.0
        return g_r, g_z

    grad_te_r_g, grad_te_z_g = _grad_rz(Te_g)
    grad_ti_r_g, grad_ti_z_g = _grad_rz(Ti_g)

    mesh_grad_te_r = grad_te_r_g.reshape(ncell_flat, order='F')
    mesh_grad_te_z = grad_te_z_g.reshape(ncell_flat, order='F')
    mesh_grad_ti_r = grad_ti_r_g.reshape(ncell_flat, order='F')
    mesh_grad_ti_z = grad_ti_z_g.reshape(ncell_flat, order='F')
    for arr in [mesh_grad_te_r, mesh_grad_te_z,
                mesh_grad_ti_r, mesh_grad_ti_z]:
        arr[~np.isfinite(arr)] = 0.0

    # -- Write plasma.h5 --
    with h5py.File(plasma_out, "w") as f:
        # Embedded equilibrium: raw .equ / GEQDSK-derived psi map on its
        # native jm×km grid, plus btf/rtf/psib. compute plasma/fields and
        # fix plasma/data use this to evaluate B at any (R, Z) and
        # psi_norm for the core-sink boundary; no separate .equ file is
        # needed at run time.
        f.create_dataset("equilibrium/r",    data=np.asarray(equ_dict["equ_r"], dtype=np.float64))
        f.create_dataset("equilibrium/z",    data=np.asarray(equ_dict["equ_z"], dtype=np.float64))
        f.create_dataset("equilibrium/psi",  data=np.asarray(equ_dict["equ_psi"], dtype=np.float64))
        f.create_dataset("equilibrium/btf",  data=float(equ_dict["btf"]))
        f.create_dataset("equilibrium/rtf",  data=float(equ_dict["rtf"]))
        f.create_dataset("equilibrium/psib", data=float(equ_dict["psib"]))

        # Multi-ion species metadata. Per-species plasma fields live on
        # the EIRENE triangulation under /mesh/ions/.
        sdt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("ion_species/names", data=np.array(ion_names, dtype=object), dtype=sdt)
        f.create_dataset("ion_species/spec_index", data=ion_indices.astype(np.int32))
        f.create_dataset("ion_species/main_ion_spec_index", data=np.array([int(ion_indices[main_k])], dtype=np.int32))
        f.create_dataset("ion_species/mass_amu", data=ion_masses)
        f.create_dataset("ion_species/charge_state_z", data=ion_charges)

        # SOLPS mesh triangulation for direct point-in-cell interpolation
        f.create_dataset("mesh/vtx_r", data=mesh_vtx_r)
        f.create_dataset("mesh/vtx_z", data=mesh_vtx_z)
        f.create_dataset("mesh/triangles", data=mesh_tri)
        f.create_dataset("mesh/cell_index", data=mesh_cell_idx)
        f.create_dataset("mesh/dens_e", data=mesh_ne)
        f.create_dataset("mesh/temp_e", data=mesh_te)
        f.create_dataset("mesh/dens_i", data=mesh_ni)
        f.create_dataset("mesh/temp_i", data=mesh_ti)
        f.create_dataset("mesh/parr_flow", data=mesh_upar)
        f.create_dataset("mesh/grad_te_r", data=mesh_grad_te_r)
        f.create_dataset("mesh/grad_te_z", data=mesh_grad_te_z)
        f.create_dataset("mesh/grad_ti_r", data=mesh_grad_ti_r)
        f.create_dataset("mesh/grad_ti_z", data=mesh_grad_ti_z)
        f.create_dataset("mesh/ions/dens", data=mesh_ions_dens)
        f.create_dataset("mesh/ions/temp", data=mesh_ions_temp)
        f.create_dataset("mesh/ions/parr_flow", data=mesh_ions_upar)

        # Per-cell wall face area (m^2, toroidally integrated). Nonzero for
        # B2 boundary cells only. Downstream consumers (emit/surf/recycle)
        # multiply by ne*cs*sin(alpha_B) to get the Bohm wall flux that
        # drives wall recycling at each boundary cell.
        f.create_dataset("mesh/wall_face_area", data=mesh_wall_face_area)

        # Per-wall-segment topological mapping to B2 boundary cells:
        #   mesh_wall_surf_cell[iseg]: flat index of the B2 boundary cell
        #       that dominantly owns SPARTA wall segment iseg (the B2 cell
        #       contributing the largest face-area into this segment).
        #   mesh_wall_surf_area[iseg]: summed B2 boundary face area from
        #       all faces that chose this segment. Conserves the Bohm
        #       flux budget across a coarser SPARTA wall.
        f.create_dataset("mesh/wall_surf_cell", data=mesh_wall_surf_cell)
        f.create_dataset("mesh/wall_surf_area", data=mesh_wall_surf_area)

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
        _write_sparta_wall_from_mesh_extra(mesh_path, wall_out)
        print(f"Wrote wall (legacy mesh.extra walk): {wall_out}")

    print(f"Wrote plasma: {plasma_out}")
    print(f"Grid: nr={nr}, nz={nz}, R=[{r[0]:.4f},{r[-1]:.4f}], Z=[{z[0]:.4f},{z[-1]:.4f}]")
    print(f"Ion species ({nion}): {ion_names}")
    print(f"Charge states: {ion_charges.tolist()}")

    # -- Optional plots --
    if plot:
        # B-field is no longer interpolated onto B2 cell centroids here
        # (converter writes only the equilibrium psi/btf/rtf, not a
        # per-cell br). Pass zeros for the br panel — diagnostic plots
        # should be rebuilt to read from /equilibrium/ directly.
        _save_plots(
            cell_polys,
            ne,
            te_eV,
            main_dens_raw,
            ti_eV,
            main_upar_raw,
            np.zeros_like(ne),
            plot_prefix or Path("convert_solps_plasma"),
        )

def _save_plots(cell_polys, ne, te, ni, ti, upar, br, prefix):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.collections import PolyCollection

    valid_poly = np.all(np.isfinite(cell_polys), axis=(1, 2))

    def _flatten(arr):
        vals = np.asarray(arr, dtype=np.float64).reshape(-1)
        return vals[valid_poly]

    polys = cell_polys[valid_poly]

    def _add_poly_panel(ax, values, title, cmap, *, log10=False, symmetric=False):
        vals = _flatten(values)
        mask = np.isfinite(vals)
        if log10:
            mask &= vals > 0.0
            vals = np.where(mask, np.log10(vals), np.nan)
        else:
            vals = np.where(mask, vals, np.nan)

        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            finite = np.array([0.0], dtype=np.float64)

        if symmetric:
            vmax = float(np.nanmax(np.abs(finite)))
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0
            norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        else:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        coll = PolyCollection(
            polys,
            array=vals,
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            linewidths=0.0,
            antialiased=False,
        )
        ax.add_collection(coll)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        return coll

    panels = [
        (ne, "log10(ne) [m^-3]", "inferno", True, False),
        (te, "Te [eV]", "magma", False, False),
        (ni, "log10(ni) [m^-3]", "inferno", True, False),
        (ti, "Ti [eV]", "magma", False, False),
        (upar, "u_par [m/s]", "RdBu_r", False, True),
        (br, "Br [T]", "RdBu_r", False, True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (arr, title, cmap, log10, symmetric) in zip(axes.flat, panels):
        coll = _add_poly_panel(ax, arr, title, cmap, log10=log10, symmetric=symmetric)
        fig.colorbar(coll, ax=ax, shrink=0.9)
    fig.savefig(f"{prefix}_plasma.png", dpi=180)
    plt.close(fig)
    print(f"Wrote plot: {prefix}_plasma.png")


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
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-prefix", type=Path, default=None)
    p.add_argument("--wall-out", type=Path, default=None, help="Output SPARTA wall file from mesh.extra")
    p.add_argument("--mesh-extra", type=Path, default=None)
    p.add_argument("--b2fgmtry", type=Path, default=None, help="Path to b2fgmtry (if not in run_path)")
    p.add_argument("--wall-source", type=str, default="auto",
                   choices=["auto", "mesh-extra", "eirene", "b2"],
                   help=("Which wall-construction path to use. 'auto' picks "
                         "mesh-extra if available, else eirene, else b2. "
                         "'mesh-extra' + 'b2' are SOLPS-native and do NOT "
                         "depend on EIRENE fort.33/34/35 files."))
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
        mesh_extra=args.mesh_extra,
        b2fgmtry_path=args.b2fgmtry,
        wall_source=args.wall_source,
    )


if __name__ == "__main__":
    main()
