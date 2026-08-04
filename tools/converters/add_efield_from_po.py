#!/usr/bin/env python3
"""Fill mesh/e_r, e_z, e_t in an OpenEdge plasma.h5 from its stored potential.

For h5 files built without balance.nc (E written as zeros) but which carry the
B2 potential via fort.31 (`mesh/pob`, volts): compute E = -grad(po) on the B2
cell centers with the same structured least-squares fit the converter uses,
and overwrite mesh/e_r, mesh/e_z, mesh/e_t (+ b2/ copies) in place.

Usage: add_efield_from_po.py plasma.h5 [--nx 90 --ny 41]
Array order convention (matches converter): flat k = ix + nx*iy (order='F').
"""
from __future__ import annotations
import argparse
import numpy as np
import h5py


def grad_rz_lsq(F, R, Z, valid):
    """Least-squares gradient of F(R,Z) on a structured curvilinear grid."""
    nx, ny = F.shape
    gr = np.zeros_like(F)
    gz = np.zeros_like(F)

    def accumulate(ix0, iy0, radius):
        a11 = a12 = a22 = b1 = b2 = 0.0
        npt = 0
        f0, r0, z0 = F[ix0, iy0], R[ix0, iy0], Z[ix0, iy0]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                ix, iy = ix0 + dx, iy0 + dy
                if not (0 <= ix < nx and 0 <= iy < ny) or not valid[ix, iy]:
                    continue
                dR, dZ = R[ix, iy] - r0, Z[ix, iy] - z0
                ds2 = dR * dR + dZ * dZ
                if ds2 < 1e-12:
                    continue
                w = 1.0 / ds2
                df = F[ix, iy] - f0
                a11 += w * dR * dR
                a12 += w * dR * dZ
                a22 += w * dZ * dZ
                b1 += w * dR * df
                b2 += w * dZ * df
                npt += 1
        det = a11 * a22 - a12 * a12
        if npt < 3 or abs(det) <= 1e-24:
            return None
        return ((a22 * b1 - a12 * b2) / det, (-a12 * b1 + a11 * b2) / det)

    for ix0 in range(nx):
        for iy0 in range(ny):
            if not valid[ix0, iy0]:
                continue
            sol = accumulate(ix0, iy0, 1) or accumulate(ix0, iy0, 2)
            if sol:
                gr[ix0, iy0], gz[ix0, iy0] = sol
    return gr, gz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5file")
    p.add_argument("--nx", type=int, default=90)
    p.add_argument("--ny", type=int, default=41)
    args = p.parse_args()
    nx, ny = args.nx, args.ny

    with h5py.File(args.h5file, "r+") as f:
        po = np.asarray(f["mesh/pob"])                  # [V], flat F-order
        poly = np.asarray(f["b2/cell_polygons"])        # (ncell, 4, 2) (R, Z)
        assert po.size == nx * ny == poly.shape[0]

        c = poly.mean(axis=1)
        x, y = poly[:, :, 0], poly[:, :, 1]
        area = 0.5 * np.abs(np.sum(
            x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))

        shape = (nx, ny)
        P = po.reshape(shape, order="F")
        R = c[:, 0].reshape(shape, order="F")
        Z = c[:, 1].reshape(shape, order="F")
        valid = (area > 1e-8).reshape(shape, order="F")  # skip guard cells

        gr, gz = grad_rz_lsq(P, R, Z, valid)
        e_r = np.where(valid, -gr, 0.0)
        e_z = np.where(valid, -gz, 0.0)
        for a in (e_r, e_z):
            a[~np.isfinite(a)] = 0.0

        flat_r = e_r.reshape(-1, order="F")
        flat_z = e_z.reshape(-1, order="F")
        flat_t = np.zeros_like(flat_r)                  # axisymmetric SOLPS
        for name, data in [("e_r", flat_r), ("e_z", flat_z), ("e_t", flat_t)]:
            for grp in ("mesh", "b2"):
                key = f"{grp}/{name}"
                if key in f:
                    f[key][...] = data
                else:
                    f.create_dataset(key, data=data)

        mag = np.hypot(flat_r, flat_z)
        print(f"E filled from po: |E| max {mag.max():.3e} V/m, "
              f"median (nonzero) {np.median(mag[mag > 0]):.3e} V/m, "
              f"nonzero {np.count_nonzero(mag)}/{mag.size}")


if __name__ == "__main__":
    main()
