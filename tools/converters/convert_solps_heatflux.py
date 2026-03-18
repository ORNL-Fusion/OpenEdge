#!/usr/bin/env python3
"""
SOLPS heatflux -> OpenEdge converter (no quixote dependency).

Extracts heat-flux density from SOLPS face-centered fluxes (fht, fhj, sx, sy)
read directly from b2fstate, and writes heatflux.h5.

Usage:
    python solps_heatflux2openedge.py /path/to/solps_run \\
        --heatflux-out heatflux.h5 --method jeremy_no_jv --nr 300 --nz 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import griddata

# Import SOLPS parsers from the main converter
from convert_solps_plasma import b2f_read_dims, b2f_extract, read_b2fgmtry, \
    _cell_centers_from_corners, _regular_grid, _interp_field


def compute_heatflux_cell_center(run_path: Path, method: str = "jeremy_no_jv") -> tuple:
    """
    Compute heat-flux from SOLPS face fluxes on a cell-centered grid.

    Methods:
      - jeremy_no_jv: |(fht_x - fhj_x)/sx|
      - jeremy_total: |fht_x/sx|
      - vector_mag:   sqrt((fht_x/sx)^2 + (fht_y/sy)^2)

    Returns (rc, zc, qmag) where rc/zc are cell centers and qmag is heat flux.
    """
    b2fstate = run_path / "b2fstate"
    b2fgmtry = run_path / "b2fgmtry"

    nx, ny, ns = b2f_read_dims(b2fstate)
    gmtry = read_b2fgmtry(str(b2fgmtry))
    rc, zc = _cell_centers_from_corners(gmtry["crx"], gmtry["cry"], nx, ny)

    # Total heat flux: use fht if available, otherwise fhe + fhi
    try:
        fht = b2f_extract("fht", b2fstate)  # (nx+2, ny+2, 2)
    except KeyError:
        fhe = b2f_extract("fhe", b2fstate)
        fhi = b2f_extract("fhi", b2fstate)
        if fhe.ndim == 2:
            fhe = fhe.reshape(nx + 2, ny + 2, -1, order="F")
        if fhi.ndim == 2:
            fhi = fhi.reshape(nx + 2, ny + 2, -1, order="F")
        fht = fhe + fhi
        print("Using fhe + fhi as total heat flux (fht not in b2fstate)")

    # Face areas: use sx if available, otherwise gs(:,:,0) from geometry
    # (consistent with Jeremy's SOLPS postprocessing: sx = Geo.gs(:,:,1))
    try:
        sx = b2f_extract("sx", b2fstate)    # (nx+2, ny+2)
    except KeyError:
        try:
            gs = b2f_extract("gs", b2fgmtry)  # (nx+2, ny+2, 3)
            if gs.ndim == 2:
                gs = gs.reshape(nx + 2, ny + 2, -1, order="F")
            sx = gs[:, :, 0]  # poloidal face area
            print("Using gs(:,:,0) from b2fgmtry as poloidal face area")
        except KeyError:
            sx = b2f_extract("hx", b2fgmtry)
            print("WARNING: Using hx from b2fgmtry as face area fallback (gs not found)")

    if fht.ndim == 2:
        fht = fht.reshape(nx + 2, ny + 2, -1, order="F")

    # Clip guard cells (ix=0,nx+1 and iy=0,ny+1) which have unphysical
    # face areas and can produce extreme heat flux density values.
    s = np.s_[1:-1, 1:-1]  # interior slice
    rc, zc = rc[s], zc[s]
    fht = fht[s]
    sx = sx[s]

    qx = np.divide(fht[:, :, 0], sx, out=np.full_like(sx, np.nan), where=sx > 0)

    if method == "jeremy_no_jv":
        try:
            fhj = b2f_extract("fhj", b2fstate)
            if fhj.ndim == 2:
                fhj = fhj.reshape(nx + 2, ny + 2, -1, order="F")
            fhj = fhj[s]
            qx = np.divide(fht[:, :, 0] - fhj[:, :, 0], sx,
                            out=np.full_like(sx, np.nan), where=sx > 0)
        except KeyError:
            print("fhj not in b2fstate — falling back to jeremy_total method")

    if method in ("jeremy_no_jv", "jeremy_total"):
        qx_c = 0.5 * (qx + np.roll(qx, -1, axis=0))
        qx_c[-1, :] = np.nan
        return rc, zc, np.abs(qx_c)

    if method == "vector_mag":
        try:
            sy = b2f_extract("sy", b2fstate)
        except KeyError:
            try:
                gs = b2f_extract("gs", b2fgmtry)
                if gs.ndim == 2:
                    gs = gs.reshape(nx + 2, ny + 2, -1, order="F")
                sy = gs[:, :, 1]  # radial face area
                print("Using gs(:,:,1) from b2fgmtry as radial face area")
            except KeyError:
                sy = b2f_extract("hy", b2fgmtry)
                print("WARNING: Using hy from b2fgmtry as radial face area fallback")
        sy = sy[s]
        qy = np.divide(fht[:, :, 1], sy, out=np.full_like(sy, np.nan), where=sy > 0)
        qx_c = 0.5 * (qx + np.roll(qx, -1, axis=0))
        qx_c[-1, :] = np.nan
        qy_c = 0.5 * (qy + np.roll(qy, -1, axis=1))
        qy_c[:, -1] = np.nan
        return rc, zc, np.sqrt(qx_c**2 + qy_c**2)

    raise ValueError(f"Unknown heatflux method: {method}")


def convert_solps_heatflux(
    run_path: Path,
    heatflux_out: Path,
    nr: int = 300,
    nz: int = 300,
    rmin: float | None = None,
    rmax: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    method: str = "jeremy_no_jv",
    plot: bool = False,
    plot_out: Path | None = None,
) -> None:
    rc, zc, qmag = compute_heatflux_cell_center(run_path, method)
    r, z, rr, zz = _regular_grid(rc, zc, nr, nz, rmin, rmax, zmin, zmax)

    src_pts = np.column_stack((rc.reshape(-1), zc.reshape(-1)))
    tgt_pts = np.column_stack((rr.reshape(-1), zz.reshape(-1)))
    qmag_grid = _interp_field(src_pts, qmag, tgt_pts, nz, nr)
    qmag_grid = np.nan_to_num(qmag_grid, nan=0.0, posinf=0.0, neginf=0.0)

    with h5py.File(heatflux_out, "w") as f:
        f.create_dataset("r", data=r)
        f.create_dataset("z", data=z)
        gg = f.create_group("grid")
        gf = f.create_group("fields")
        rr2d, zz2d = np.meshgrid(r, z)
        gg.create_dataset("R", data=rr2d)
        gg.create_dataset("Z", data=zz2d)
        gg.create_dataset("Rc", data=r)
        gg.create_dataset("Zc", data=z)
        gf.create_dataset("q_mag", data=qmag_grid)
        f.create_dataset("R", data=rr2d)
        f.create_dataset("Z", data=zz2d)
        f.create_dataset("q_mag", data=qmag_grid)

    print(f"Wrote heatflux: {heatflux_out} (method={method})")
    print(f"Grid: nr={nr}, nz={nz}, R=[{r[0]:.4f},{r[-1]:.4f}], Z=[{z[0]:.4f},{z[-1]:.4f}]")

    if plot:
        import matplotlib.pyplot as plt
        extent = [float(r.min()), float(r.max()), float(z.min()), float(z.max())]
        fig, ax = plt.subplots(1, 1, figsize=(6, 8), constrained_layout=True)
        log_q = np.full_like(qmag_grid, np.nan)
        m = qmag_grid > 0
        log_q[m] = np.log10(qmag_grid[m])
        im = ax.imshow(log_q, origin="lower", extent=extent, aspect="auto", cmap="inferno")
        ax.set_title(f"log10(q_mag) [W/m^2] ({method})")
        ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
        fig.colorbar(im, ax=ax, shrink=0.9)
        out_png = plot_out or Path("solps_heatflux.png")
        fig.savefig(out_png, dpi=180); plt.close(fig)
        print(f"Wrote plot: {out_png}")


def main():
    p = argparse.ArgumentParser(description="Extract SOLPS heatflux to OpenEdge HDF5 (no quixote)")
    p.add_argument("run_path", type=Path, help="SOLPS run directory")
    p.add_argument("--heatflux-out", type=Path, default=Path("heatflux.h5"))
    p.add_argument("--nr", type=int, default=300)
    p.add_argument("--nz", type=int, default=300)
    p.add_argument("--rmin", type=float, default=None)
    p.add_argument("--rmax", type=float, default=None)
    p.add_argument("--zmin", type=float, default=None)
    p.add_argument("--zmax", type=float, default=None)
    p.add_argument("--method", default="jeremy_total",
                   choices=["jeremy_no_jv", "jeremy_total", "vector_mag"])
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-out", type=Path, default=None)
    args = p.parse_args()
    convert_solps_heatflux(**vars(args))


if __name__ == "__main__":
    main()
