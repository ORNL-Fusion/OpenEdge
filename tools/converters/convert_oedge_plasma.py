#!/usr/bin/env python3
"""OEDGE/DIVIMP NetCDF -> OpenEdge plasma.h5 (mesh-native).

Reads an OEDGE `.nc` background file plus an equilibrium `.equ`, builds the
native OEDGE cell polygons (via KORPG/NVERTP/RVERTP/ZVERTP) as an
unstructured triangular mesh, and writes plasma.h5 with the same /mesh/*
schema as `convert_solps_plasma.py` and `convert_s3x_plasma.py`.

Output groups:
    /equilibrium/{r, z, psi, btf, rtf, psib}
    /ion_species/{names, spec_index, main_ion_spec_index, mass_amu, charge_state_z}
    /mesh/{vtx_r, vtx_z, triangles, cell_index,
           dens_e, temp_e, dens_i, temp_i, parr_flow,
           grad_te_r, grad_te_z, grad_ti_r, grad_ti_z,
           e_r, e_z, e_t, pob,
           vtx_br, vtx_bz, vtx_bt, vtx_bmag, vtx_bpol, vtx_btor,
           tri_source_kind, wall_gap_mask,
           ions/{dens, temp, parr_flow}}

Usage:
    python convert_oedge_plasma.py d3d-204953-bkg-v25.nc \\
        --equ-file 204953_3000.x16.equ \\
        --plasma-out plasma.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata


# ======================================================================
# Equilibrium (.equ) file reader — same format as SOLPS/s3x converters
# ======================================================================

def read_equilibrium(equ_file: Path):
    """Read .equ and reconstruct (Br, Bt, Bz) from psi on its native grid."""
    jm = km = btf = rtf = None
    psib = 0.0
    read_r = read_z = read_psi = False
    r_vals, z_vals, psi_vals = [], [], []

    with open(equ_file, "r", encoding="utf-8", errors="ignore") as f:
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
            if tok[0].startswith("((psi(j,k)"):
                read_r, read_z, read_psi = False, False, True; continue
            if read_r:
                try: r_vals.extend(float(x) for x in tok)
                except ValueError: read_r = False
                continue
            if read_z:
                try: z_vals.extend(float(x) for x in tok)
                except ValueError: read_z = False
                continue
            if read_psi:
                try: psi_vals.extend(float(x) for x in tok)
                except ValueError: read_psi = False
                continue

    if any(v is None for v in [jm, km, btf, rtf]):
        raise RuntimeError(f"Missing headers (jm/km/btf/rtf) in {equ_file}")

    r_eq = np.asarray(r_vals[:jm], dtype=np.float64)
    z_eq = np.asarray(z_vals[:km], dtype=np.float64)
    psi = np.asarray(psi_vals[: jm * km], dtype=np.float64).reshape((km, jm))

    # Reconstruct B from psi: Br = -(1/R) dpsi/dZ, Bz = (1/R) dpsi/dR,
    # Bt = btf*rtf/R.
    dz = z_eq[1] - z_eq[0]
    dr = r_eq[1] - r_eq[0]
    grad_z, grad_r = np.gradient(psi, dz, dr)
    rr = np.meshgrid(r_eq, z_eq)[0]
    safe_r = np.where(np.abs(rr) > 1e-12, rr, 1e-12)
    br = -grad_z / safe_r
    bz =  grad_r / safe_r
    bt = (btf * rtf) / safe_r

    return {
        "r": r_eq, "z": z_eq, "br": br, "bt": bt, "bz": bz,
        "equ_r": r_eq.copy(), "equ_z": z_eq.copy(), "equ_psi": psi,
        "btf": btf, "rtf": rtf, "psib": psib,
    }


# ======================================================================
# OEDGE NetCDF reader
# ======================================================================

def _detect_korpg_offset(korpg: np.ndarray, nvertp: np.ndarray) -> int:
    """OEDGE sometimes indexes polygons from 0 and sometimes from 1.
    Match what plot_oedge_native.py does."""
    max_id = int(korpg.max())
    if max_id == len(nvertp):
        return 1
    if max_id == len(nvertp) - 1:
        return 0
    return 1 if len(nvertp) > 1 else 0


def read_oedge_nc(nc_file: Path) -> dict:
    """Read OEDGE .nc and return both ring/knot-indexed fields AND native
    polygon geometry (KORPG/NVERTP/RVERTP/ZVERTP) for mesh building."""
    ds = nc.Dataset(str(nc_file), "r")

    # Grid metadata
    nrs    = int(ds.variables["NRS"][:])
    nks    = np.asarray(ds.variables["NKS"][:])
    irsep  = int(ds.variables["IRSEP"][:])
    irwall = int(ds.variables["IRWALL"][:])
    irtrap = int(ds.variables["IRTRAP"][:])

    # Per-cell fields on (ring, knot) layout.
    rs    = np.asarray(ds.variables["RS"][:])
    zs    = np.asarray(ds.variables["ZS"][:])
    ktebs = np.asarray(ds.variables["KTEBS"][:])
    ktibs = np.asarray(ds.variables["KTIBS"][:])
    knbs  = np.asarray(ds.variables["KNBS"][:])
    kvhs  = np.asarray(ds.variables["KVHS"][:])
    kes   = np.asarray(ds.variables["KES"][:])
    tegs  = np.asarray(ds.variables["TEGS"][:])
    tigs  = np.asarray(ds.variables["TIGS"][:])

    # Parallel flow is sometimes pre-scaled by QTIM; undo if present.
    qtim = None
    if "QTIM" in ds.variables:
        qtim = float(ds.variables["QTIM"][:])
        if qtim > 0.0:
            kvhs = kvhs / qtim

    # Native polygon geometry.
    korpg  = np.asarray(ds.variables["KORPG"][:])
    nvertp = np.asarray(ds.variables["NVERTP"][:])
    rvertp = np.asarray(ds.variables["RVERTP"][:])
    zvertp = np.asarray(ds.variables["ZVERTP"][:])
    offset = _detect_korpg_offset(korpg, nvertp)

    # Scalar metadata
    crmb = float(ds.variables["CRMB"][:]) if "CRMB" in ds.variables else 2.0
    r0   = float(ds.variables["R0"][:])   if "R0"   in ds.variables else 0.0
    z0   = float(ds.variables["Z0"][:])   if "Z0"   in ds.variables else 0.0

    ds.close()

    # Validity mask on (ring, knot). Same filter plot_oedge_native.py uses:
    # polygon id > 0 and at least 3 vertices.
    valid = np.zeros(korpg.shape, dtype=bool)
    for ir in range(korpg.shape[0]):
        nk = int(nks[ir]) if ir < len(nks) else korpg.shape[1]
        for ik in range(min(nk, korpg.shape[1])):
            pid_raw = int(korpg[ir, ik])
            if pid_raw <= 0:
                continue
            pid = pid_raw - offset
            if pid < 0 or pid >= len(nvertp):
                continue
            if int(nvertp[pid]) < 3:
                continue
            # Skip degenerate centroids at origin (OEDGE sometimes pads).
            if not (np.isfinite(rs[ir, ik]) and rs[ir, ik] > 0.1):
                continue
            valid[ir, ik] = True

    return dict(
        # Ring/knot grid (full 2D arrays; filtered via `valid` downstream)
        rs=rs, zs=zs, valid=valid, nks=nks, nrs=nrs,
        # Polygon geometry
        korpg=korpg, nvertp=nvertp, rvertp=rvertp, zvertp=zvertp,
        korpg_offset=offset,
        # Per-cell plasma (eV, eV, m^-3, m/s, V/m, eV/m, eV/m)
        ktebs=ktebs, ktibs=ktibs, knbs=knbs, kvhs=kvhs, kes=kes,
        tegs=tegs, tigs=tigs,
        # Metadata
        crmb=crmb, r0=r0, z0=z0,
        irsep=irsep, irwall=irwall, irtrap=irtrap,
    )


# ======================================================================
# Mesh builder — polygons -> deduplicated triangular mesh
# ======================================================================

def build_oedge_mesh(oedge: dict, vertex_round_digits: int = 9):
    """Fan-triangulate OEDGE cell polygons into a single triangular mesh
    with deduplicated vertices.

    Returns
    -------
    vtx_r, vtx_z     : (nvtx,) float64   deduplicated vertex coords
    triangles        : (ntri, 3) int32   vertex indices per triangle
    cell_index       : (ntri,)  int32    per-triangle -> cell ordinal (0..ncell-1)
    cell_ir, cell_ik : (ncell,) int32    per-cell -> OEDGE (ring, knot) for plasma lookup
    """
    korpg   = oedge["korpg"]
    nvertp  = oedge["nvertp"]
    rvertp  = oedge["rvertp"]
    zvertp  = oedge["zvertp"]
    valid   = oedge["valid"]
    offset  = oedge["korpg_offset"]

    vtx_map: dict[tuple[float, float], int] = {}
    vtx_r_list: list[float] = []
    vtx_z_list: list[float] = []
    tri_list: list[tuple[int, int, int]] = []
    cell_index_list: list[int] = []
    cell_ir_list: list[int] = []
    cell_ik_list: list[int] = []

    def _get_vid(r: float, z: float) -> int:
        key = (round(float(r), vertex_round_digits),
               round(float(z), vertex_round_digits))
        vid = vtx_map.get(key)
        if vid is None:
            vid = len(vtx_r_list)
            vtx_map[key] = vid
            vtx_r_list.append(key[0])
            vtx_z_list.append(key[1])
        return vid

    cell_ord = 0
    nr, nk = korpg.shape
    for ir in range(nr):
        for ik in range(nk):
            if not valid[ir, ik]:
                continue
            pid = int(korpg[ir, ik]) - offset
            nv = int(nvertp[pid])
            poly_r = rvertp[pid, :nv]
            poly_z = zvertp[pid, :nv]
            if not (np.isfinite(poly_r).all() and np.isfinite(poly_z).all()):
                continue
            vids = [_get_vid(poly_r[i], poly_z[i]) for i in range(nv)]
            # Fan triangulation: (v0, vi, vi+1) for i=1..nv-2.
            added_tri = 0
            for i in range(1, nv - 1):
                a, b, c = vids[0], vids[i], vids[i + 1]
                if a == b or b == c or a == c:
                    continue
                tri_list.append((a, b, c))
                cell_index_list.append(cell_ord)
                added_tri += 1
            if added_tri == 0:
                continue
            cell_ir_list.append(ir)
            cell_ik_list.append(ik)
            cell_ord += 1

    vtx_r = np.asarray(vtx_r_list, dtype=np.float64)
    vtx_z = np.asarray(vtx_z_list, dtype=np.float64)
    triangles = np.asarray(tri_list, dtype=np.int32)
    cell_index = np.asarray(cell_index_list, dtype=np.int32)
    cell_ir = np.asarray(cell_ir_list, dtype=np.int32)
    cell_ik = np.asarray(cell_ik_list, dtype=np.int32)
    return vtx_r, vtx_z, triangles, cell_index, cell_ir, cell_ik


# ======================================================================
# Per-triangle mesh-native LSQ gradient (same pattern as s3x/SOLPS)
# ======================================================================

def mesh_lsq_gradient(vtx_r, vtx_z, triangles, field_tri):
    """One-ring vertex-neighbor LSQ fit of (grad_R, grad_Z) per triangle."""
    ntri = triangles.shape[0]
    nvtx = vtx_r.size
    tri_R = vtx_r[triangles].mean(axis=1)
    tri_Z = vtx_z[triangles].mean(axis=1)

    vtx_tris: list[list[int]] = [[] for _ in range(nvtx)]
    for t in range(ntri):
        for k in range(3):
            vtx_tris[triangles[t, k]].append(t)

    gR = np.zeros(ntri, dtype=np.float64)
    gZ = np.zeros(ntri, dtype=np.float64)
    f_tri = np.asarray(field_tri, dtype=np.float64)
    for t in range(ntri):
        f0 = f_tri[t]
        r0 = tri_R[t]; z0 = tri_Z[t]
        if not (np.isfinite(f0) and np.isfinite(r0) and np.isfinite(z0)):
            continue
        stencil = set()
        for k in range(3):
            stencil.update(vtx_tris[triangles[t, k]])
        stencil.discard(t)
        a11 = a12 = a22 = 0.0
        b1 = b2 = 0.0
        npt = 0
        for s in stencil:
            f1 = f_tri[s]
            if not np.isfinite(f1):
                continue
            dR = tri_R[s] - r0
            dZ = tri_Z[s] - z0
            ds2 = dR * dR + dZ * dZ
            if ds2 <= 1e-20:
                continue
            w = 1.0 / ds2
            df = f1 - f0
            a11 += w * dR * dR
            a12 += w * dR * dZ
            a22 += w * dZ * dZ
            b1  += w * dR * df
            b2  += w * dZ * df
            npt += 1
        det = a11 * a22 - a12 * a12
        if npt < 3 or abs(det) <= 1e-24:
            continue
        gR[t] = ( a22 * b1 - a12 * b2) / det
        gZ[t] = (-a12 * b1 + a11 * b2) / det
    gR[~np.isfinite(gR)] = 0.0
    gZ[~np.isfinite(gZ)] = 0.0
    return gR, gZ


# ======================================================================
# Main conversion
# ======================================================================

def convert_oedge_to_openedge(
    nc_file: Path,
    equ_file: Path,
    plasma_out: Path = Path("plasma.h5"),
):
    print(f"Reading OEDGE file: {nc_file}")
    oedge = read_oedge_nc(nc_file)
    print(f"  ring/knot grid: NRS={oedge['nrs']}, valid cells={int(oedge['valid'].sum())}, "
          f"IRSEP={oedge['irsep']}, IRWALL={oedge['irwall']}, IRTRAP={oedge['irtrap']}")

    print(f"Reading equilibrium: {equ_file}")
    equ_dict = read_equilibrium(equ_file)
    print(f"  equilibrium grid: {len(equ_dict['r'])}x{len(equ_dict['z'])}, "
          f"R=[{equ_dict['r'][0]:.3f}, {equ_dict['r'][-1]:.3f}], "
          f"Z=[{equ_dict['z'][0]:.3f}, {equ_dict['z'][-1]:.3f}]")

    # ---- Build mesh from OEDGE polygons ----
    print("Building OEDGE native polygon mesh ...")
    vtx_r, vtx_z, tri, cell_index, cell_ir, cell_ik = build_oedge_mesh(oedge)
    ntri = tri.shape[0]
    ncell = cell_ir.size
    print(f"  mesh: {vtx_r.size} vertices, {ntri} triangles, {ncell} cells "
          f"(typically ~2 triangles/cell for quad polygons)")

    # ---- Gather per-cell plasma (length = ncell, NOT ntri). ----
    # Downstream consumers (fix background, compute pmi/surf/data) expect
    # mesh/dens_e, mesh/temp_e, mesh/ions/*, etc. to be ncell-long, and
    # mesh/cell_index[tri] maps each triangle to its owning cell. This
    # matches the SOLPS / s3x converter convention.
    def per_cell(field_rk):
        v = field_rk[cell_ir, cell_ik].astype(np.float64, copy=False)
        return np.nan_to_num(v, nan=0.0)

    mesh_dens_e    = per_cell(oedge["knbs"])
    mesh_temp_e    = per_cell(oedge["ktebs"])
    mesh_temp_i    = per_cell(oedge["ktibs"])
    mesh_parr_flow = per_cell(oedge["kvhs"])
    mesh_e_par     = per_cell(oedge["kes"])
    # OEDGE is single-ion deuterium: ni = ne under quasi-neutrality.
    mesh_dens_i    = mesh_dens_e.copy()

    # ---- Evaluate equilibrium B directly onto mesh vertices ----
    # Same approach as SOLPS: scipy.griddata linear with nearest fallback.
    rr_eq, zz_eq = np.meshgrid(equ_dict["r"], equ_dict["z"])
    eq_pts = np.column_stack([rr_eq.reshape(-1), zz_eq.reshape(-1)])
    vtx_pts = np.column_stack([vtx_r, vtx_z])

    def _interp(values_2d):
        v = np.asarray(values_2d, dtype=np.float64).reshape(-1)
        lin = griddata(eq_pts, v, vtx_pts, method="linear")
        nn  = griddata(eq_pts, v, vtx_pts, method="nearest")
        return np.where(np.isfinite(lin), lin, nn)

    mesh_vtx_br = _interp(equ_dict["br"])
    mesh_vtx_bz = _interp(equ_dict["bz"])
    mesh_vtx_bt = _interp(equ_dict["bt"])
    mesh_vtx_bpol = np.sqrt(mesh_vtx_br ** 2 + mesh_vtx_bz ** 2)
    mesh_vtx_bmag = np.sqrt(mesh_vtx_bpol ** 2 + mesh_vtx_bt ** 2)
    print(f"Per-vertex B from equilibrium: "
          f"|B| range=[{mesh_vtx_bmag.min():.3f}, {mesh_vtx_bmag.max():.3f}] T")

    # ---- Cylindrical-component decomposition of parallel quantities ----
    # Need per-cell B-hat for E-field decomposition. Compute per-triangle
    # B (vertex-mean), then average the (potentially 2) triangles of each
    # cell back to per-cell via cell_index.
    tri_br = mesh_vtx_br[tri].mean(axis=1)
    tri_bz = mesh_vtx_bz[tri].mean(axis=1)
    tri_bt = mesh_vtx_bt[tri].mean(axis=1)
    cell_br = np.zeros(ncell); cell_bz = np.zeros(ncell); cell_bt = np.zeros(ncell)
    cell_count = np.zeros(ncell, dtype=np.int32)
    np.add.at(cell_br,    cell_index, tri_br)
    np.add.at(cell_bz,    cell_index, tri_bz)
    np.add.at(cell_bt,    cell_index, tri_bt)
    np.add.at(cell_count, cell_index, 1)
    nz_mask = cell_count > 0
    cell_br[nz_mask] /= cell_count[nz_mask]
    cell_bz[nz_mask] /= cell_count[nz_mask]
    cell_bt[nz_mask] /= cell_count[nz_mask]
    cell_bmag = np.sqrt(cell_br**2 + cell_bz**2 + cell_bt**2)
    safe = np.where(cell_bmag > 1e-12, cell_bmag, 1e-12)
    bhat_r_c = cell_br / safe
    bhat_t_c = cell_bt / safe
    bhat_z_c = cell_bz / safe

    # Per-species ion arrays (shape (nion, ncell)) -- single ion for OEDGE.
    mesh_ions_dens = mesh_dens_i[np.newaxis, :]
    mesh_ions_temp = mesh_temp_i[np.newaxis, :]
    mesh_ions_upar = mesh_parr_flow[np.newaxis, :]

    # ---- Temperature gradients on the mesh via per-triangle LSQ, then
    # cell-average (matches the per-cell layout of dens_e/temp_e).
    print("Computing mesh-native LSQ gradients ...")
    tri_grad_te_r, tri_grad_te_z = mesh_lsq_gradient(
        vtx_r, vtx_z, tri, mesh_temp_e[cell_index])
    tri_grad_ti_r, tri_grad_ti_z = mesh_lsq_gradient(
        vtx_r, vtx_z, tri, mesh_temp_i[cell_index])

    def tri_to_cell(tri_arr):
        out = np.zeros(ncell)
        np.add.at(out, cell_index, tri_arr)
        out[nz_mask] /= cell_count[nz_mask]
        return out

    mesh_grad_te_r = tri_to_cell(tri_grad_te_r)
    mesh_grad_te_z = tri_to_cell(tri_grad_te_z)
    mesh_grad_ti_r = tri_to_cell(tri_grad_ti_r)
    mesh_grad_ti_z = tri_to_cell(tri_grad_ti_z)

    # ---- E-field from KES (parallel E) projected into cylindrical (per-cell).
    mesh_e_r = mesh_e_par * bhat_r_c
    mesh_e_t = mesh_e_par * bhat_t_c
    mesh_e_z = mesh_e_par * bhat_z_c

    # pob (potential) — not carried through OEDGE standard output; zero stub.
    mesh_pob = np.zeros(ncell, dtype=np.float64)

    # Provenance stubs (OEDGE polygons extend to the wall, no vacuum fill).
    mesh_tri_source_kind = np.ones(ntri, dtype=np.int32)
    mesh_wall_gap_mask   = np.zeros(ntri, dtype=np.int32)

    # ---- Write plasma.h5 ----
    print(f"Writing {plasma_out} ...")
    with h5py.File(str(plasma_out), "w") as f:
        # Embedded equilibrium on its native .equ grid.
        f.create_dataset("equilibrium/r",    data=np.asarray(equ_dict["equ_r"], dtype=np.float64))
        f.create_dataset("equilibrium/z",    data=np.asarray(equ_dict["equ_z"], dtype=np.float64))
        f.create_dataset("equilibrium/psi",  data=np.asarray(equ_dict["equ_psi"], dtype=np.float64))
        f.create_dataset("equilibrium/btf",  data=float(equ_dict["btf"]))
        f.create_dataset("equilibrium/rtf",  data=float(equ_dict["rtf"]))
        f.create_dataset("equilibrium/psib", data=float(equ_dict["psib"]))

        # Single-ion metadata (deuterium).
        sdt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("ion_species/names",
                         data=np.array(["D+"], dtype=object), dtype=sdt)
        f.create_dataset("ion_species/elements",
                         data=np.array(["D"], dtype=object), dtype=sdt)
        f.create_dataset("ion_species/spec_index",
                         data=np.array([0], dtype=np.int32))
        f.create_dataset("ion_species/main_ion_spec_index",
                         data=np.array([0], dtype=np.int32))
        f.create_dataset("ion_species/mass_amu",
                         data=np.array([oedge["crmb"]], dtype=np.float64))
        f.create_dataset("ion_species/charge_state_z",
                         data=np.array([1], dtype=np.int32))

        # Mesh.
        f.create_dataset("mesh/vtx_r",      data=vtx_r)
        f.create_dataset("mesh/vtx_z",      data=vtx_z)
        f.create_dataset("mesh/triangles",  data=tri)
        f.create_dataset("mesh/cell_index", data=cell_index)

        f.create_dataset("mesh/dens_e",    data=mesh_dens_e)
        f.create_dataset("mesh/temp_e",    data=mesh_temp_e)
        f.create_dataset("mesh/dens_i",    data=mesh_dens_i)
        f.create_dataset("mesh/temp_i",    data=mesh_temp_i)
        f.create_dataset("mesh/parr_flow", data=mesh_parr_flow)
        f.create_dataset("mesh/pob",       data=mesh_pob)

        f.create_dataset("mesh/grad_te_r", data=mesh_grad_te_r)
        f.create_dataset("mesh/grad_te_z", data=mesh_grad_te_z)
        f.create_dataset("mesh/grad_ti_r", data=mesh_grad_ti_r)
        f.create_dataset("mesh/grad_ti_z", data=mesh_grad_ti_z)

        f.create_dataset("mesh/e_r", data=mesh_e_r)
        f.create_dataset("mesh/e_z", data=mesh_e_z)
        f.create_dataset("mesh/e_t", data=mesh_e_t)

        f.create_dataset("mesh/vtx_br",   data=mesh_vtx_br)
        f.create_dataset("mesh/vtx_bz",   data=mesh_vtx_bz)
        f.create_dataset("mesh/vtx_bt",   data=mesh_vtx_bt)
        f.create_dataset("mesh/vtx_bmag", data=mesh_vtx_bmag)
        f.create_dataset("mesh/vtx_bpol", data=mesh_vtx_bpol)
        f.create_dataset("mesh/vtx_btor", data=mesh_vtx_bt)

        f.create_dataset("mesh/tri_source_kind", data=mesh_tri_source_kind)
        f.create_dataset("mesh/wall_gap_mask",   data=mesh_wall_gap_mask)

        f.create_dataset("mesh/ions/dens",      data=mesh_ions_dens)
        f.create_dataset("mesh/ions/temp",      data=mesh_ions_temp)
        f.create_dataset("mesh/ions/parr_flow", data=mesh_ions_upar)

        f.attrs["source"] = f"OEDGE: {Path(nc_file).name}"
        f.attrs["r0"] = oedge["r0"]
        f.attrs["z0"] = oedge["z0"]

    print("Done.")


# ======================================================================
# CLI
# ======================================================================

def _build_parser():
    p = argparse.ArgumentParser(
        description="Convert OEDGE/DIVIMP .nc to OpenEdge plasma.h5 "
                    "(mesh-native, matches SOLPS/s3x /mesh/* schema)."
    )
    p.add_argument("nc_file", type=Path,
                   help="OEDGE NetCDF background file (.nc)")
    p.add_argument("--equ-file", type=Path, required=True,
                   help="Equilibrium .equ file for B-field reconstruction.")
    p.add_argument("--plasma-out", type=Path, default=Path("plasma.h5"))
    return p


def main():
    args = _build_parser().parse_args()
    convert_oedge_to_openedge(
        nc_file=args.nc_file,
        equ_file=args.equ_file,
        plasma_out=args.plasma_out,
    )


if __name__ == "__main__":
    main()
