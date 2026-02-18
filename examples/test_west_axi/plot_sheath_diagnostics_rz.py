#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm


def read_last_grid_block(path: Path):
    lines = path.read_text().splitlines()
    i = 0
    last_rows, last_cols = None, None
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        i += 2
        if i >= len(lines) or not lines[i].startswith("ITEM: NUMBER OF CELLS"):
            break
        i += 1
        ncell = int(lines[i].strip())
        i += 1
        if i >= len(lines) or not lines[i].startswith("ITEM: BOX BOUNDS"):
            break
        i += 4
        if i >= len(lines) or not lines[i].startswith("ITEM: CELLS"):
            break
        cols = lines[i].split()[2:]
        i += 1
        rows = []
        for _ in range(ncell):
            if i >= len(lines) or lines[i].startswith("ITEM:"):
                break
            parts = lines[i].split()
            if len(parts) == len(cols):
                rows.append(parts)
            i += 1
        last_rows, last_cols = rows, cols
    if not last_rows or not last_cols:
        raise RuntimeError(f"Could not parse grid dump: {path}")
    return last_cols, last_rows


def build_mesh(cols, rows, fields):
    ix, iy = cols.index("xc"), cols.index("yc")
    cmap = {c: j for j, c in enumerate(cols)}
    for f in fields:
        if f not in cmap:
            raise RuntimeError(f"Missing column: {f}")
    rvals = sorted({float(r[ix]) for r in rows})
    zvals = sorted({float(r[iy]) for r in rows})
    nr, nz = len(rvals), len(zvals)
    R = np.array(rvals).reshape(1, nr).repeat(nz, axis=0)
    Z = np.array(zvals).reshape(nz, 1).repeat(nr, axis=1)
    rowm = {(float(r[ix]), float(r[iy])): r for r in rows}
    out = {}
    for f in fields:
        arr = np.full((nz, nr), np.nan)
        j = cmap[f]
        for iz, zv in enumerate(zvals):
            for ir, rv in enumerate(rvals):
                rec = rowm.get((rv, zv))
                if rec is not None:
                    arr[iz, ir] = float(rec[j])
        out[f] = arr
    return R, Z, out


def read_surface_polygon(path: Path):
    pts, segs, section = {}, [], None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low == "points":
            section = "points"
            continue
        if low == "lines":
            section = "lines"
            continue
        p = line.split()
        if section == "points" and len(p) >= 3 and p[0].isdigit():
            pts[int(p[0])] = (float(p[1]), float(p[2]))
        elif section == "lines" and len(p) >= 3 and p[0].isdigit():
            segs.append((int(p[1]), int(p[2])))
    if not pts:
        return None
    if not segs:
        return [pts[i] for i in sorted(pts)]
    chain = [segs[0][0], segs[0][1]]
    rem = segs[1:]
    while rem:
        last = chain[-1]
        idx = None
        for i, (a, b) in enumerate(rem):
            if a == last:
                chain.append(b); idx = i; break
            if b == last:
                chain.append(a); idx = i; break
        if idx is None:
            break
        rem.pop(idx)
        if chain[-1] == chain[0]:
            break
    if chain[0] != chain[-1]:
        chain.append(chain[0])
    return [pts[i] for i in chain if i in pts]


def _finite(a):
    x = np.asarray(a, dtype=float)
    return x[np.isfinite(x)]


def _pclip(a, pmin, pmax):
    x = _finite(a)
    if x.size == 0:
        return (0.0, 1.0)
    lo = float(np.nanpercentile(x, pmin))
    hi = float(np.nanpercentile(x, pmax))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
    return (lo, hi)


def _draw_panel(ax, R, Z, fld, ttl, cmap, args, kind="linear"):
    mm = np.ma.masked_invalid(fld)
    if kind == "log":
        pos = np.asarray(mm.compressed())
        pos = pos[pos > 0.0]
        if pos.size == 0:
            im = ax.pcolormesh(R, Z, mm, shading="auto", cmap=cmap)
        else:
            vmin = float(np.nanpercentile(pos, max(args.pmin, 0.1)))
            vmax = float(np.nanpercentile(pos, args.pmax))
            if vmax <= vmin:
                vmax = vmin * 10.0
            im = ax.pcolormesh(R, Z, mm, shading="auto", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    elif kind == "sym":
        lo, hi = _pclip(mm, args.pmin, args.pmax)
        vmax = max(abs(lo), abs(hi))
        if vmax <= 0.0:
            vmax = 1.0
        im = ax.pcolormesh(R, Z, mm, shading="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    else:
        lo, hi = _pclip(mm, args.pmin, args.pmax)
        im = ax.pcolormesh(R, Z, mm, shading="auto", cmap=cmap, vmin=lo, vmax=hi)
    cb = plt.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(ttl)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="output/sheath.grid")
    ap.add_argument("--out", default="output/west_sheath_diagnostics.rz.png")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--pmin", type=float, default=2.0, help="lower percentile for color scaling")
    ap.add_argument("--pmax", type=float, default=98.0, help="upper percentile for color scaling")
    ap.add_argument("--elog", action="store_true", help="use log scale for |E_sheath|")
    ap.add_argument("--no-mask-active", action="store_true", help="do not mask non-active cells")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    dumpf = (base / args.dump).resolve() if not Path(args.dump).is_absolute() else Path(args.dump)

    fields = [
        "c_csheath[1]", "c_csheath[2]", "c_csheath[3]", "c_csheath[4]",
        "c_csheath[5]", "c_csheath[7]", "c_csheath[8]", "c_csheath[9]", "c_csheath[10]"
    ]
    cols, rows = read_last_grid_block(dumpf)
    R, Z, m = build_mesh(cols, rows, fields)

    ex, ey, ez = m["c_csheath[1]"], m["c_csheath[2]"], m["c_csheath[3]"]
    dphi = m["c_csheath[4]"]
    mpar = m["c_csheath[5]"]
    alpha = m["c_csheath[7]"]
    active = m["c_csheath[8]"]
    bdotn = m["c_csheath[9]"]
    dist = m["c_csheath[10]"]

    emag = np.sqrt(np.maximum(0.0, ex*ex + ey*ey + ez*ez))
    if args.no_mask_active:
        amag = emag
        dphi_plot = dphi
        mpar_plot = mpar
        bdotn_plot = bdotn
        dist_plot = dist
        alpha_plot = alpha
    else:
        mask = active > 0.5
        amag = np.where(mask, emag, np.nan)
        dphi_plot = np.where(mask, dphi, np.nan)
        mpar_plot = np.where(mask, mpar, np.nan)
        bdotn_plot = np.where(mask, bdotn, np.nan)
        dist_plot = np.where(mask, dist, np.nan)
        alpha_plot = np.where(mask, alpha, np.nan)

    wall = read_surface_polygon(base / "input" / "wall.txt")
    core = read_surface_polygon(base / "input" / "core.txt")

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), dpi=200, constrained_layout=True)
    fig.patch.set_facecolor("#d9d9d9")
    for ax in axs.ravel():
        ax.set_facecolor("#d9d9d9")

    panels = [
        (amag, r"$|E_{sheath}|$ [V/m]", "inferno", "log" if args.elog else "linear"),
        (dphi_plot, r"$\Delta\Phi_{sheath}$ [eV]", "magma", "linear"),
        (mpar_plot, r"$M_{\parallel}$", "viridis", "linear"),
        (bdotn_plot, r"$\hat{b}\cdot\hat{n}$", "coolwarm", "sym"),
        (dist_plot, "dist to wall [m]", "cividis", "linear"),
        (alpha_plot, r"$\alpha$ [deg]", "plasma", "linear"),
    ]

    for ax, (fld, ttl, cmap, kind) in zip(axs.ravel(), panels):
        _draw_panel(ax, R, Z, fld, ttl, cmap, args, kind=kind)
        if core:
            cp = np.array(core)
            ax.plot(cp[:, 0], cp[:, 1], color="#3ea6ff", lw=1.2)
        if wall:
            wp = np.array(wall)
            ax.plot(wp[:, 0], wp[:, 1], color="black", lw=1.2)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls="--", alpha=0.25)

    fig.suptitle("WEST sheath diagnostics (Borodkina): |E| vs DeltaPhi", fontsize=14)
    out = (base / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(f"Wrote: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
