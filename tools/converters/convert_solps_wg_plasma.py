#!/usr/bin/env python3
# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# SPDX-License-Identifier: GPL-2.0
"""
Wide-grid (GOAT/unstructured) SOLPS-ITER run -> OpenEdge plasma.h5.

Counterpart of convert_solps_plasma_eirene.py for wide-grid cases
(b2fgmtry with nCv/nFc/nVx headers, isClassicalGrid=0). Raw SOLPS
parsing is imported from the SOLPS-routines package by Jeremy Lore
(ORNL), https://github.com/ORNL-Fusion/SOLPS-routines — the same
readers handle structured and wide grids.

Wide grids reach the physical wall, so the structured converter's
vacuum-fill / wall-gap machinery is unnecessary: every polygonal B2
cell is fan-triangulated directly and each triangle indexes its parent
cell's plasma values. B on mesh vertices comes from the GEQDSK
equilibrium. Gradients (thermal force) and E = -grad(po) are computed
per cell by least squares over face-adjacent neighbours.

Usage:
  python convert_solps_wg_plasma.py RUN_DIR --b2fgmtry BASERUN/b2fgmtry \
      --gfile g038757.005000 --plasma-out plasma.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# structured-converter helpers we reuse verbatim
sys.path.insert(0, str(Path(__file__).parent))
from convert_solps_plasma_eirene import (  # noqa: E402
    _build_species_metadata,
    _interp_regular_to_points,
    _ordered_loops_from_edges,
    _read_geqdsk_bfield,
)

EV = 1.602176634e-19


def _readers():
    try:
        from solps_routines import readers
    except ImportError as err:
        raise ImportError(
            "solps_routines (SOLPS-routines, Jeremy Lore, ORNL) is required; "
            "add its src/ to PYTHONPATH"
        ) from err
    return readers


def triangulate_cells(geo):
    """Fan-triangulate the wide-grid cell polygons.

    Returns vtx_r, vtx_z, tri (ntri,3 0-based), tri_cell (ntri,) cell id.
    """
    vtx = np.asarray(geo["Vertices"], dtype=np.float64)      # (nVx, 2)
    faces = np.asarray(geo["Faces"], dtype=np.float64)       # (nCv, maxv), 1-based, NaN pad
    tris, tri_cell = [], []
    for icv in range(faces.shape[0]):
        row = faces[icv]
        vids = row[np.isfinite(row)].astype(np.int64) - 1
        if vids.size < 3:
            continue
        for k in range(1, vids.size - 1):
            tris.append((vids[0], vids[k], vids[k + 1]))
            tri_cell.append(icv)
    return (vtx[:, 0].copy(), vtx[:, 1].copy(),
            np.asarray(tris, dtype=np.int32),
            np.asarray(tri_cell, dtype=np.int32))


def cell_adjacency(geo):
    """Neighbour lists from cell->face lists (faces shared by two cells)."""
    ncv = int(geo["nCv"])
    cv_fc_p = np.asarray(geo["cvFcP"], dtype=np.int64)
    cv_fc = np.asarray(geo["cvFc"], dtype=np.int64)
    face_owner = {}
    nbrs = [[] for _ in range(ncv)]
    for icv in range(ncv):
        start, count = int(cv_fc_p[icv, 0]) - 1, int(cv_fc_p[icv, 1])
        for f in cv_fc[start:start + count]:
            f = int(f)
            if f in face_owner:
                j = face_owner[f]
                if j != icv:
                    nbrs[icv].append(j)
                    nbrs[j].append(icv)
            else:
                face_owner[f] = icv
    return nbrs


def lsq_gradient(values, rc, zc, nbrs):
    """Per-cell (d/dR, d/dZ) by least squares over face neighbours."""
    n = len(values)
    gr = np.zeros(n)
    gz = np.zeros(n)
    for i in range(n):
        js = nbrs[i]
        if len(js) < 2:
            continue
        dr = rc[js] - rc[i]
        dz = zc[js] - zc[i]
        dv = values[js] - values[i]
        a11, a12, a22 = np.dot(dr, dr), np.dot(dr, dz), np.dot(dz, dz)
        b1, b2 = np.dot(dr, dv), np.dot(dz, dv)
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-30:
            continue
        gr[i] = (a22 * b1 - a12 * b2) / det
        gz[i] = (a11 * b2 - a12 * b1) / det
    return gr, gz


def wall_loop(vtx_r, vtx_z, tri):
    """Ordered wall polyline = longest boundary loop of the triangulation."""
    edge_count = {}
    for t in tri:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e = (int(min(a, b)), int(max(a, b)))
            edge_count[e] = edge_count.get(e, 0) + 1
    boundary = [e for e, n in edge_count.items() if n == 1]
    loops = _ordered_loops_from_edges(boundary)
    loop = max(loops, key=len)
    return np.asarray([vtx_r[i] for i in loop]), np.asarray([vtx_z[i] for i in loop])


def convert(run_path: Path, b2fgmtry: Path, gfile: Path, plasma_out: Path):
    readers = _readers()

    print(f"reading {b2fgmtry}")
    geo = readers.read_b2fgmtry(str(b2fgmtry), use_cache=False)
    if not geo.get("isUnstructured"):
        raise RuntimeError("not a wide-grid b2fgmtry — use convert_solps_plasma_eirene.py")
    ncv = int(geo["nCv"])

    print(f"reading {run_path / 'b2fstate'}")
    state = readers.read_b2fstate(str(run_path / "b2fstate"), use_cache=False)
    ns = int(np.asarray(state["ns"]).ravel()[0]) if "ns" in state else state["na"].shape[-1]

    def cellf(name):
        a = np.asarray(state[name], dtype=np.float64)
        return a.reshape(ncv) if a.size == ncv else a.reshape(ncv, -1)

    ne = cellf("ne")
    te = cellf("te") / EV
    ti = cellf("ti") / EV
    na = cellf("na")
    ua = cellf("ua")
    po = cellf("po") if "po" in state else np.zeros(ncv)

    # species metadata -> charged ions
    zamax = np.asarray(state["zamax"]).ravel()
    zamin = np.asarray(state["zamin"]).ravel()
    zn = np.asarray(state["zn"]).ravel()
    am = np.asarray(state["am"]).ravel()
    names, masses, charges, is_neutral = _build_species_metadata(zamax, zamin, zn, am, ns)
    ion_idx = np.where(~is_neutral)[0]
    print(f"species: {names}  ions: {[names[i] for i in ion_idx]}")
    main = int(ion_idx[0])

    # mesh
    vtx_r, vtx_z, tri, tri_cell = triangulate_cells(geo)
    print(f"mesh: {vtx_r.size} vertices, {tri.shape[0]} triangles, {ncv} cells")

    # gradients + E on cells
    rc = vtx_r[tri].mean(axis=1)
    # cell centres: average triangle centroids per cell (area-agnostic, fine for LSQ)
    cell_r = np.zeros(ncv)
    cell_z = np.zeros(ncv)
    cnt = np.zeros(ncv)
    zc_t = vtx_z[tri].mean(axis=1)
    np.add.at(cell_r, tri_cell, rc)
    np.add.at(cell_z, tri_cell, zc_t)
    np.add.at(cnt, tri_cell, 1.0)
    good = cnt > 0
    cell_r[good] /= cnt[good]
    cell_z[good] /= cnt[good]

    nbrs = cell_adjacency(geo)
    gte_r, gte_z = lsq_gradient(te, cell_r, cell_z, nbrs)
    gti_r, gti_z = lsq_gradient(ti, cell_r, cell_z, nbrs)
    gpo_r, gpo_z = lsq_gradient(po, cell_r, cell_z, nbrs)
    e_r, e_z = -gpo_r, -gpo_z

    # equilibrium B at vertices
    print(f"reading equilibrium {gfile}")
    equ = _read_geqdsk_bfield(gfile)
    pts = np.column_stack([vtx_r, vtx_z])
    er_, ez_ = np.asarray(equ["r"]), np.asarray(equ["z"])
    vbr = _interp_regular_to_points(er_, ez_, np.asarray(equ["br"]), pts)
    vbt = _interp_regular_to_points(er_, ez_, np.asarray(equ["bt"]), pts)
    vbz = _interp_regular_to_points(er_, ez_, np.asarray(equ["bz"]), pts)
    vbpol = np.sqrt(vbr**2 + vbz**2)
    vbmag = np.sqrt(vbpol**2 + vbt**2)

    # wall
    wr, wz = wall_loop(vtx_r, vtx_z, tri)
    print(f"wall loop: {wr.size} points")

    ncell = ncv
    with h5py.File(plasma_out, "w") as f:
        f.create_dataset("equilibrium/r", data=er_)
        f.create_dataset("equilibrium/z", data=ez_)
        f.create_dataset("equilibrium/psi", data=np.asarray(equ["equ_psi"]))
        f.create_dataset("equilibrium/btf", data=float(equ["btf"]))
        f.create_dataset("equilibrium/rtf", data=float(equ["rtf"]))
        f.create_dataset("equilibrium/psib", data=float(equ["psib"]))
        f.create_dataset("equilibrium/psi_axis", data=float(equ["psi_axis"]))

        sdt = h5py.string_dtype()
        ion_names = [names[i] for i in ion_idx]
        f.create_dataset("ion_species/names", data=np.array(ion_names, dtype=object), dtype=sdt)
        f.create_dataset("ion_species/elements",
                         data=np.array([n.rstrip("+0123456789") for n in ion_names],
                                       dtype=object), dtype=sdt)
        f.create_dataset("ion_species/spec_index", data=ion_idx.astype(np.int32))
        f.create_dataset("ion_species/main_ion_spec_index",
                         data=np.array([main], dtype=np.int32))
        f.create_dataset("ion_species/mass_amu", data=masses[ion_idx])
        f.create_dataset("ion_species/charge_state_z", data=charges[ion_idx])

        f.create_dataset("mesh/vtx_r", data=vtx_r)
        f.create_dataset("mesh/vtx_z", data=vtx_z)
        f.create_dataset("mesh/vtx_br", data=vbr)
        f.create_dataset("mesh/vtx_bz", data=vbz)
        f.create_dataset("mesh/vtx_bt", data=vbt)
        f.create_dataset("mesh/vtx_bmag", data=vbmag)
        f.create_dataset("mesh/vtx_bpol", data=vbpol)
        f.create_dataset("mesh/vtx_btor", data=vbt)
        f.create_dataset("mesh/triangles", data=tri)
        f.create_dataset("mesh/cell_index", data=tri_cell)
        f.create_dataset("mesh/tri_source_kind",
                         data=np.ones(tri.shape[0], dtype=np.int8))
        f.create_dataset("mesh/wall_gap_mask",
                         data=np.zeros(tri.shape[0], dtype=bool))
        f.create_dataset("mesh/dens_e", data=ne)
        f.create_dataset("mesh/temp_e", data=te)
        f.create_dataset("mesh/dens_i", data=na[:, main])
        f.create_dataset("mesh/temp_i", data=ti)
        f.create_dataset("mesh/parr_flow", data=ua[:, main])
        f.create_dataset("mesh/grad_te_r", data=gte_r)
        f.create_dataset("mesh/grad_te_z", data=gte_z)
        f.create_dataset("mesh/grad_ti_r", data=gti_r)
        f.create_dataset("mesh/grad_ti_z", data=gti_z)
        f.create_dataset("mesh/e_r", data=e_r)
        f.create_dataset("mesh/e_z", data=e_z)
        f.create_dataset("mesh/e_t", data=np.zeros(ncell))
        f.create_dataset("mesh/ions/dens",
                         data=np.stack([na[:, i] for i in ion_idx]))
        f.create_dataset("mesh/ions/temp",
                         data=np.stack([ti for _ in ion_idx]))
        f.create_dataset("mesh/ions/parr_flow",
                         data=np.stack([ua[:, i] for i in ion_idx]))
        f.create_dataset("wall/r", data=wr)
        f.create_dataset("wall/z", data=wz)
        f.attrs["grid"] = "wide"
        f.attrs["source_run"] = str(run_path)
    print(f"wrote {plasma_out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_path", type=Path)
    p.add_argument("--b2fgmtry", type=Path, required=True)
    p.add_argument("--gfile", type=Path, required=True)
    p.add_argument("--plasma-out", type=Path, default=Path("plasma.h5"))
    a = p.parse_args()
    convert(a.run_path, a.b2fgmtry, a.gfile, a.plasma_out)


if __name__ == "__main__":
    main()
