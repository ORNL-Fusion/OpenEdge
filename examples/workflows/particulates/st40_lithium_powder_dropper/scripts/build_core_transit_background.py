#!/usr/bin/env python3
"""Add an explicit, labelled central-core continuation to ST40 plasma.h5.

Native SOLPS triangles always take precedence in OpenEdge. The regular-grid
fields written here are sampled only in the unmeshed central hole (plus a
two-cell continuity halo); every other out-of-mesh location is explicit vacuum.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import h5py
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from check_equilibrium_orientation import boundary_loops, check, signed_area


POSITIVE_FIELDS = ("temp_e", "temp_i", "dens_e", "dens_i")
SIGNED_FIELDS = ("parr_flow", "e_r", "e_t", "e_z")
HEAT_FLUX_FIELDS = ("q_par", "q_perp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def harmonic_fill(
    inside: np.ndarray,
    boundary: np.ndarray,
    dr: float,
    dz: float,
) -> np.ndarray:
    """Solve Laplace(field)=0 on ``inside`` with neighbouring Dirichlet data."""
    nz, nr = inside.shape
    index = np.full((nz, nr), -1, dtype=np.int64)
    index[inside] = np.arange(int(inside.sum()))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(int(inside.sum()), dtype=np.float64)
    neighbours = ((-1, 0, 1.0 / dz**2), (1, 0, 1.0 / dz**2),
                  (0, -1, 1.0 / dr**2), (0, 1, 1.0 / dr**2))
    for iz, ir in np.argwhere(inside):
        row = int(index[iz, ir])
        diagonal = 0.0
        for jz, jr, weight in neighbours:
            zz, rr = iz + jz, ir + jr
            if not (0 <= zz < nz and 0 <= rr < nr):
                raise RuntimeError("core-fill mask touches regular-grid boundary")
            diagonal += weight
            if inside[zz, rr]:
                rows.append(row)
                cols.append(int(index[zz, rr]))
                data.append(-weight)
            else:
                rhs[row] += weight * boundary[zz, rr]
        rows.append(row)
        cols.append(row)
        data.append(diagonal)
    matrix = coo_matrix((data, (rows, cols)), shape=(len(rhs), len(rhs))).tocsr()
    result = np.zeros_like(boundary)
    result[inside] = spsolve(matrix, rhs)
    return result


def build(base: Path, output: Path, nr: int, nz: int) -> None:
    orientation = check(base)
    if not orientation["pass"]:
        raise RuntimeError("base plasma fails equilibrium orientation gate")

    with h5py.File(base, "r") as h5:
        triangles = h5["mesh/triangles"][:]
        rv = h5["mesh/vtx_r"][:]
        zv = h5["mesh/vtx_z"][:]
        loops = boundary_loops(triangles)
        inner = min(loops, key=lambda q: signed_area(rv[q], zv[q]))
        core_polygon = np.column_stack((rv[inner], zv[inner]))
        cell_points = np.column_stack((h5["b2/r_center"][:],
                                       h5["b2/z_center"][:]))
        missing_q = [name for name in HEAT_FLUX_FIELDS
                     if f"b2/{name}" not in h5 or f"mesh/{name}" not in h5]
        if missing_q:
            raise RuntimeError(
                "base plasma is missing physical heat flux: "
                + ", ".join(missing_q)
            )
        native = {name: h5[f"b2/{name}"][:] for name in
                  POSITIVE_FIELDS + SIGNED_FIELDS + HEAT_FLUX_FIELDS}

    r = np.linspace(0.0, 1.05, nr)
    z = np.linspace(-1.0, 1.0, nz)
    rr, zz = np.meshgrid(r, z)
    points = np.column_stack((rr.ravel(), zz.ravel()))
    inside = MplPath(core_polygon).contains_points(points).reshape(nz, nr)
    halo = binary_dilation(inside, iterations=2) & ~inside
    tree = cKDTree(cell_points)
    _, nearest = tree.query(points, k=1)
    nearest = nearest.reshape(nz, nr)

    arrays: dict[str, np.ndarray] = {}
    for name in POSITIVE_FIELDS + SIGNED_FIELDS + HEAT_FLUX_FIELDS:
        boundary = native[name][nearest]
        if name in POSITIVE_FIELDS:
            floor = max(float(np.min(native[name][native[name] > 0])) * 1.0e-12,
                        np.finfo(float).tiny)
            solved = harmonic_fill(inside, np.log(np.maximum(boundary, floor)),
                                   r[1] - r[0], z[1] - z[0])
            filled = np.exp(solved)
        else:
            filled = harmonic_fill(inside, boundary, r[1] - r[0], z[1] - z[0])
        out = np.zeros((nz, nr), dtype=np.float64)
        out[inside] = filled[inside]
        out[halo] = boundary[halo]
        arrays[name] = out

    dte_dz, dte_dr = np.gradient(arrays["temp_e"], z, r)
    dti_dz, dti_dr = np.gradient(arrays["temp_i"], z, r)
    arrays.update({
        "grad_te_r": dte_dr,
        "grad_te_z": dte_dz,
        "grad_ti_r": dti_dr,
        "grad_ti_z": dti_dz,
    })
    provenance = np.zeros((nz, nr), dtype=np.uint8)
    provenance[inside] = 1       # harmonic core extension
    provenance[halo] = 2         # native nearest-cell continuity halo

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base, output)
    with h5py.File(output, "r+") as h5:
        for name in ("r", "z", "core_fill_mask") + tuple(arrays):
            if name in h5:
                del h5[name]
        h5.create_dataset("r", data=r)
        h5.create_dataset("z", data=z)
        for name, values in arrays.items():
            h5.create_dataset(name, data=values, compression="gzip",
                              compression_opts=4, shuffle=True)
        ds = h5.create_dataset("core_fill_mask", data=provenance,
                               compression="gzip", compression_opts=4)
        ds.attrs["labels"] = "0=explicit_vacuum,1=harmonic_core,2=native_halo"
        h5.attrs["core_fill_model"] = "log-harmonic positive; harmonic signed"
        h5.attrs["core_fill_heatflux_model"] = (
            "harmonic continuation of native SOLPS q_par/q_perp"
        )
        h5.attrs["core_fill_base_sha256"] = sha256(base)
        h5.attrs["core_fill_native_precedence"] = True
        h5.attrs["core_fill_orientation_gate"] = "PASS"
        h5.attrs["core_fill_grid_shape_nz_nr"] = np.array((nz, nr), dtype=np.int32)

    print(f"wrote {output}")
    print(f"  base sha256: {sha256(base)}")
    print(f"  output sha256: {sha256(output)}")
    print(f"  core nodes: {inside.sum()}, halo nodes: {halo.sum()}")
    print(f"  regular grid: nz={nz}, nr={nr}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", type=Path, required=True,
        help=("orientation-corrected SOLPS run_dir HDF5 carrying the native "
              "q_par/q_perp fields; this intermediate is not tracked"),
    )
    parser.add_argument("--output", type=Path,
                        default=root / "input/plasma_st40_solps_corefill.h5")
    parser.add_argument("--nr", type=int, default=211,
                        help="R nodes over [0,1.05] m (default 5 mm)")
    parser.add_argument("--nz", type=int, default=401,
                        help="Z nodes over [-1,1] m (default 5 mm)")
    args = parser.parse_args()
    build(args.base, args.output, args.nr, args.nz)


if __name__ == "__main__":
    main()
