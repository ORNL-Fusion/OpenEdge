#!/usr/bin/env python3
"""
SOLPS-ITER -> OpenEdge converter (no quixote dependency).

Reads SOLPS binary text files (b2fgmtry, b2fstate) directly using the
*cf: header format, following Jeremy Lore / Jae-Sun Park's SOLPS routines.

Outputs:
  plasma.h5  - regular (R,Z) grid with all plasma fields + multi-ion species
  bfield.h5  - regular (R,Z) grid with Br, Bt, Bz

Usage:
    python convert_solps_plasma.py /path/to/solps_run \\
        --equ-file equilibrium.equ \\
        --plasma-out plasma.h5 --bfield-out bfield.h5 \\
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
    """Read .equ file and reconstruct (Br, Bt, Bz) from psi."""
    if not equ_file.exists():
        raise FileNotFoundError(f"Equilibrium file not found: {equ_file}")

    jm = km = btf = rtf = None
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
    return r, z, br, bt, bz


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
    return rs, zs, br, bt, bz


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
# Main conversion
# ==========================================================================

def convert_solps_to_openedge(
    run_path: Path,
    plasma_out: Path,
    bfield_out: Path,
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
) -> None:
    """
    Convert SOLPS run directory to OpenEdge plasma.h5 + bfield.h5.

    Reads b2fgmtry (geometry) and b2fstate (plasma state) directly —
    no quixote dependency.
    """
    b2fgmtry_file = run_path / "b2fgmtry"
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

    ne_grid = _interp_field(src_pts, ne, tgt_pts, nz, nr)
    te_grid = _interp_field(src_pts, te_eV, tgt_pts, nz, nr)

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
    if gfile is not None:
        rg, zg, br_eq, bt_eq, bz_eq = _read_geqdsk_bfield(gfile)
    elif equ_file is not None:
        rg, zg, br_eq, bt_eq, bz_eq = _read_equilibrium_bfield(equ_file)
    else:
        raise RuntimeError(
            "Equilibrium file required for B-field reconstruction.\n"
            "Provide --gfile (GEQDSK) or --equ-file (.equ)."
        )

    rr_eq, zz_eq = np.meshgrid(rg, zg)
    b_pts = np.column_stack((rr_eq.reshape(-1), zz_eq.reshape(-1)))
    br_grid = _interp_field(b_pts, br_eq, tgt_pts, nz, nr)
    bt_grid = _interp_field(b_pts, bt_eq, tgt_pts, nz, nr)
    bz_grid = _interp_field(b_pts, bz_eq, tgt_pts, nz, nr)
    br_cells = _interp_field_points(b_pts, br_eq, src_pts).reshape(nx + 2, ny + 2)

    bmag_grid = np.sqrt(br_grid**2 + bt_grid**2 + bz_grid**2)
    eps = 1e-12
    bhat_r = np.where(bmag_grid > eps, br_grid / bmag_grid, 0.0)
    bhat_t = np.where(bmag_grid > eps, bt_grid / bmag_grid, 0.0)
    bhat_z = np.where(bmag_grid > eps, bz_grid / bmag_grid, 0.0)

    # -- Interpolate per-species ion fields --
    dens_i_all = np.zeros((nion, nz, nr), dtype=np.float64)
    temp_i_all = np.zeros((nion, nz, nr), dtype=np.float64)
    flow_i_par_all = np.zeros((nion, nz, nr), dtype=np.float64)
    flow_i_r_all = np.zeros((nion, nz, nr), dtype=np.float64)
    flow_i_t_all = np.zeros((nion, nz, nr), dtype=np.float64)
    flow_i_z_all = np.zeros((nion, nz, nr), dtype=np.float64)

    ion_names = []
    ion_masses = np.zeros(nion, dtype=np.float64)
    ion_charges = np.zeros(nion, dtype=np.int32)

    for k, sidx in enumerate(ion_indices):
        n_s = na[:, :, sidx] if na.ndim == 3 else na
        # ua may be scalar (single species) or per-species
        if ua_raw.ndim == 3 and ua_raw.shape[2] > sidx:
            u_s = ua_raw[:, :, sidx]
        elif ua_raw.ndim == 2:
            u_s = ua_raw
        else:
            u_s = np.zeros_like(n_s)

        n_grid = _interp_field(src_pts, n_s, tgt_pts, nz, nr)
        # ti is shared across all ion species in SOLPS
        t_grid = _interp_field(src_pts, ti_eV, tgt_pts, nz, nr)
        u_grid = _interp_field(src_pts, u_s, tgt_pts, nz, nr)

        dens_i_all[k] = n_grid
        temp_i_all[k] = t_grid
        flow_i_par_all[k] = u_grid
        flow_i_r_all[k] = u_grid * bhat_r
        flow_i_t_all[k] = u_grid * bhat_t
        flow_i_z_all[k] = u_grid * bhat_z

        ion_names.append(species_names[sidx])
        ion_masses[k] = masses_amu[sidx]
        ion_charges[k] = charge_states[sidx]

    # -- Main ion (first charged species, typically D+) for legacy fields --
    main_k = 0
    dens_i_grid = dens_i_all[main_k]
    temp_i_grid = temp_i_all[main_k]

    # -- Temperature gradients --
    dz = z[1] - z[0]
    dr = r[1] - r[0]
    gte_z, gte_r = np.gradient(te_grid, dz, dr)
    gti_z, gti_r = np.gradient(temp_i_grid, dz, dr)

    # -- NaN cleanup --
    for arr in [ne_grid, te_grid, dens_i_grid, temp_i_grid,
                br_grid, bt_grid, bz_grid, gte_r, gte_z, gti_r, gti_z]:
        arr[~np.isfinite(arr)] = 0.0

    for arr3 in [dens_i_all, temp_i_all, flow_i_par_all,
                 flow_i_r_all, flow_i_t_all, flow_i_z_all]:
        arr3[~np.isfinite(arr3)] = 0.0

    # -- Write plasma.h5 --
    with h5py.File(plasma_out, "w") as f:
        f.create_dataset("r", data=r)
        f.create_dataset("z", data=z)
        f.create_dataset("dens_e", data=ne_grid)
        f.create_dataset("temp_e", data=te_grid)
        f.create_dataset("dens_i", data=dens_i_grid)
        f.create_dataset("temp_i", data=temp_i_grid)
        f.create_dataset("parr_flow", data=flow_i_par_all[main_k])
        f.create_dataset("parr_flow_r", data=flow_i_r_all[main_k])
        f.create_dataset("parr_flow_t", data=flow_i_t_all[main_k])
        f.create_dataset("parr_flow_z", data=flow_i_z_all[main_k])
        f.create_dataset("grad_te_r", data=gte_r)
        f.create_dataset("grad_te_t", data=np.zeros_like(gte_r))
        f.create_dataset("grad_te_z", data=gte_z)
        f.create_dataset("grad_ti_r", data=gti_r)
        f.create_dataset("grad_ti_t", data=np.zeros_like(gti_r))
        f.create_dataset("grad_ti_z", data=gti_z)

        # Multi-ion species extension
        sdt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("ion_species/names", data=np.array(ion_names, dtype=object), dtype=sdt)
        f.create_dataset("ion_species/spec_index", data=ion_indices.astype(np.int32))
        f.create_dataset("ion_species/main_ion_spec_index", data=np.array([int(ion_indices[main_k])], dtype=np.int32))
        f.create_dataset("ion_species/mass_amu", data=ion_masses)
        f.create_dataset("ion_species/charge_state_z", data=ion_charges)
        f.create_dataset("ions/dens", data=dens_i_all)
        f.create_dataset("ions/temp", data=temp_i_all)
        f.create_dataset("ions/parr_flow", data=flow_i_par_all)
        f.create_dataset("ions/parr_flow_r", data=flow_i_r_all)
        f.create_dataset("ions/parr_flow_t", data=flow_i_t_all)
        f.create_dataset("ions/parr_flow_z", data=flow_i_z_all)

        # Legacy compatibility
        f.create_dataset("n_e/dens", data=ne_grid)
        f.create_dataset("n_e/temp", data=te_grid)
        f.create_dataset("n_i/dens", data=dens_i_grid)
        f.create_dataset("n_i/temp", data=temp_i_grid)
        f.create_dataset("n_i/parr_flow", data=flow_i_par_all[main_k])

    # -- Write bfield.h5 --
    with h5py.File(bfield_out, "w") as f:
        f.create_dataset("r", data=r)
        f.create_dataset("z", data=z)
        f.create_dataset("br", data=np.nan_to_num(br_grid))
        f.create_dataset("bt", data=np.nan_to_num(bt_grid))
        f.create_dataset("bz", data=np.nan_to_num(bz_grid))

    # -- Optional wall geometry --
    if wall_out is not None:
        mesh_path = mesh_extra if mesh_extra is not None else (run_path / "mesh.extra")
        _write_sparta_wall_from_mesh_extra(mesh_path, wall_out)
        print(f"Wrote wall: {wall_out}")

    print(f"Wrote plasma: {plasma_out}")
    print(f"Wrote bfield: {bfield_out}")
    print(f"Grid: nr={nr}, nz={nz}, R=[{r[0]:.4f},{r[-1]:.4f}], Z=[{z[0]:.4f},{z[-1]:.4f}]")
    print(f"Ion species ({nion}): {ion_names}")
    print(f"Charge states: {ion_charges.tolist()}")

    # -- Optional plots --
    if plot:
        _save_plots(
            cell_polys,
            ne,
            te_eV,
            main_dens_raw,
            ti_eV,
            main_upar_raw,
            br_cells,
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
        description="Convert SOLPS run to OpenEdge plasma.h5 + bfield.h5 (no quixote)"
    )
    p.add_argument("run_path", type=Path, help="SOLPS run directory (contains b2fgmtry, b2fstate)")
    p.add_argument("--plasma-out", type=Path, default=Path("plasma.h5"))
    p.add_argument("--bfield-out", type=Path, default=Path("bfield.h5"))
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
    return p


def main():
    args = _build_parser().parse_args()
    convert_solps_to_openedge(
        run_path=args.run_path,
        plasma_out=args.plasma_out,
        bfield_out=args.bfield_out,
        nr=args.nr, nz=args.nz,
        rmin=args.rmin, rmax=args.rmax,
        zmin=args.zmin, zmax=args.zmax,
        gfile=args.gfile,
        equ_file=args.equ_file,
        plot=args.plot,
        plot_prefix=args.plot_prefix,
        wall_out=args.wall_out,
        mesh_extra=args.mesh_extra,
    )


if __name__ == "__main__":
    main()
