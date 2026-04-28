#!/usr/bin/env python3
"""Compare SOLPS-native and OpenEdge heat-flux fields safely.

This script avoids mixing mesh conventions:
- SOLPS arrays are plotted with quixote.GridDataPlot on the native SOLPS mesh.
- OpenEdge heatflux/plasma arrays are plotted with pcolormesh on the regular (R,Z) grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import quixote as qx
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.path import Path as MplPath
from quixote import GridDataPlot


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Compare SOLPS and OpenEdge heat-flux plotting")
    p.add_argument("--run", type=Path, required=True, help="SOLPS run path")
    p.add_argument("--heatflux", type=Path, default=base / "input/heatflux.fix.h5", help="OpenEdge input/heatflux.fix.h5")
    p.add_argument("--plasma", type=Path, default=base / "plasma.h5", help="OpenEdge plasma.h5")
    p.add_argument("--wall", type=Path, default=base / "input" / "wall.surf", help="Wall surface file")
    p.add_argument("--core", type=Path, default=base / "input" / "core.surf", help="Core surface file")
    p.add_argument("--no-domain-mask", action="store_true", help="Disable wall/core masking on OpenEdge grids")
    p.add_argument("--save", type=Path, default=None, help="Optional output image path (e.g. fig.png)")
    p.add_argument("--vmin", type=float, default=None, help="LogNorm vmin for q plots")
    p.add_argument("--vmax", type=float, default=None, help="LogNorm vmax for q plots")
    return p.parse_args()


def compute_solps_qmag(shot: qx.SolpsData) -> np.ndarray:
    fht = np.asarray(shot.fht, dtype=float)
    sx = np.asarray(shot.sx, dtype=float)
    sy = np.asarray(shot.sy, dtype=float)

    qtx = np.divide(fht[..., 0], sx, out=np.full_like(sx, np.nan), where=sx > 0)
    qty = np.divide(fht[..., 1], sy, out=np.full_like(sy, np.nan), where=sy > 0)

    qx_c = 0.5 * (qtx + np.roll(qtx, -1, axis=0))
    qx_c[-1, :] = np.nan
    qy_c = 0.5 * (qty + np.roll(qty, -1, axis=1))
    qy_c[:, -1] = np.nan

    q_mag = np.sqrt(qx_c * qx_c + qy_c * qy_c)
    return np.ma.masked_less_equal(q_mag, 0.0)


def load_openedge_heatflux(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        if "grid" in f and "fields" in f:
            r2d = f["grid/R"][:]
            z2d = f["grid/Z"][:]
            q = f["fields/q_mag"][:]
        else:
            r2d = f["R"][:]
            z2d = f["Z"][:]
            q = f["q_mag"][:]
    return r2d, z2d, np.ma.masked_less_equal(np.asarray(q, dtype=float), 0.0)


def load_openedge_plasma(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        r = f["r"][:]
        z = f["z"][:]
        te = f["temp_e"][:]
        e_par = f["e_par"][:]
    rr, zz = np.meshgrid(r, z)
    te_ma = np.ma.masked_less_equal(np.asarray(te, dtype=float), 0.0)
    epar_ma = np.ma.masked_invalid(np.asarray(e_par, dtype=float))
    return rr, zz, te_ma, epar_ma


def stats(name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        print(f"{name}: no finite values")
        return
    print(
        f"{name}: shape={a.shape}, min={finite.min():.3e}, "
        f"p50={np.percentile(finite, 50):.3e}, p99={np.percentile(finite, 99):.3e}, "
        f"max={finite.max():.3e}, <=0 frac={(finite <= 0).mean():.3%}"
    )


def read_surface_points(path: Path) -> np.ndarray:
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    npts = None
    points_start = None
    for i, line in enumerate(txt):
        s = line.strip().lower()
        if s.endswith("points"):
            try:
                npts = int(s.split()[0])
            except (ValueError, IndexError):
                pass
        if s == "points":
            points_start = i + 1
            break
    if npts is None or points_start is None:
        raise RuntimeError(f"Could not parse surface points from {path}")

    pts: list[tuple[float, float]] = []
    for line in txt[points_start:]:
        s = line.strip()
        if not s:
            if pts:
                break
            continue
        tok = s.split()
        if len(tok) < 3:
            continue
        pts.append((float(tok[1]), float(tok[2])))
        if len(pts) >= npts:
            break
    if len(pts) < 3:
        raise RuntimeError(f"Surface in {path} has too few points: {len(pts)}")
    return np.asarray(pts, dtype=float)


def annulus_mask(
    r2d: np.ndarray, z2d: np.ndarray, wall_pts: np.ndarray, core_pts: np.ndarray
) -> np.ndarray:
    pts = np.column_stack((r2d.ravel(), z2d.ravel()))
    in_wall = MplPath(wall_pts, closed=True).contains_points(pts).reshape(r2d.shape)
    in_core = MplPath(core_pts, closed=True).contains_points(pts).reshape(r2d.shape)
    valid = in_wall & (~in_core)
    return ~valid


def main() -> None:
    args = parse_args()
    shot = qx.SolpsData(str(args.run))

    q_solps = compute_solps_qmag(shot)
    r2d, z2d, q_oe = load_openedge_heatflux(args.heatflux)
    rr, zz, te_oe, epar_oe = load_openedge_plasma(args.plasma)
    wall_pts = core_pts = None

    if not args.no_domain_mask:
        wall_pts = read_surface_points(args.wall)
        core_pts = read_surface_points(args.core)
        oe_mask = annulus_mask(r2d, z2d, wall_pts, core_pts)
        q_oe = np.ma.masked_where(oe_mask, q_oe)
        te_oe = np.ma.masked_where(oe_mask, te_oe)
        epar_oe = np.ma.masked_where(oe_mask, epar_oe)

    stats("SOLPS q_mag", q_solps.filled(np.nan))
    stats("OpenEdge q_mag", q_oe.filled(np.nan))
    stats("OpenEdge temp_e", te_oe.filled(np.nan))
    stats("OpenEdge e_par", epar_oe.filled(np.nan))

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("0.92")

    combined = np.hstack([
        q_solps.compressed() if np.ma.isMaskedArray(q_solps) else q_solps.ravel(),
        q_oe.compressed() if np.ma.isMaskedArray(q_oe) else q_oe.ravel(),
    ])
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        raise RuntimeError("No finite positive q values to plot")

    vmin = args.vmin if args.vmin is not None else float(np.percentile(combined, 1.0))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(combined, 99.5))
    norm_q = LogNorm(vmin=max(vmin, 1e-30), vmax=max(vmax, max(vmin, 1e-29)))

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(18, 4), dpi=200, constrained_layout=True)

    GridDataPlot(shot, q_solps, canvas=ax0, cmap=cmap, norm="log", xlim=[2.0, 6.0], ylim=[-4.0, 4.0])
    ax0.set_title("SOLPS-native |q| (GridDataPlot)")
    ax0.set_xlabel("R [m]")
    ax0.set_ylabel("Z [m]")

    m1 = ax1.pcolormesh(r2d, z2d, q_oe, shading="auto", cmap=cmap, norm=norm_q)
    ax1.set_title("OpenEdge |q| (pcolormesh)")
    ax1.set_xlabel("R [m]")
    ax1.set_ylabel("Z [m]")
    fig.colorbar(m1, ax=ax1, label="|q| [W/m^2]")

    m2 = ax2.pcolormesh(rr, zz, te_oe, shading="auto", cmap="magma", norm=LogNorm())
    ax2.set_title("OpenEdge temp_e (pcolormesh)")
    ax2.set_xlabel("R [m]")
    ax2.set_ylabel("Z [m]")
    fig.colorbar(m2, ax=ax2, label="temp_e [eV]")

    epar_f = epar_oe.compressed() if np.ma.isMaskedArray(epar_oe) else np.ravel(epar_oe)
    epar_f = epar_f[np.isfinite(epar_f)]
    if epar_f.size > 0:
        epar_abs = np.abs(epar_f)
        linthresh = max(float(np.percentile(epar_abs, 20.0)), 1e-8)
        vmax_e = max(float(np.percentile(epar_abs, 99.0)), linthresh * 10.0)
        norm_e = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-vmax_e, vmax=vmax_e)
    else:
        norm_e = None
    m3 = ax3.pcolormesh(rr, zz, epar_oe, shading="auto", cmap="RdBu_r", norm=norm_e)
    ax3.set_title("OpenEdge e_par (pcolormesh)")
    ax3.set_xlabel("R [m]")
    ax3.set_ylabel("Z [m]")
    fig.colorbar(m3, ax=ax3, label="e_par [V/m]")
    
    for ax in (ax0, ax1, ax2, ax3):
        ax.set_xlim(2.0, 6.0)
        ax.set_ylim(-4.0, 4.0)
        ax.grid(ls="--", alpha=0.3)
        if wall_pts is not None and core_pts is not None:
            ax.plot(wall_pts[:, 0], wall_pts[:, 1], "k-", lw=1.2)
            ax.plot(core_pts[:, 0], core_pts[:, 1], "g-", lw=1.2)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=250, bbox_inches="tight")
        print(f"Saved figure: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
