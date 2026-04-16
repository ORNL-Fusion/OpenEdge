#!/usr/bin/env python3
"""
Plot OpenEdge/SPARTA grid density dump in WEST-style R-Z subplots.

Example:
  python3 plot_grid_density_west.py \
    --dump output/grid.west \
    --wall input/wall.surf \
    --out output/grid_density.west.png \
    --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D


def parse_surface(path: Path):
    import re

    lines = path.read_text(encoding="utf-8").splitlines()
    npts = nlines = None
    i_points = i_lines = None

    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        m_pts = re.match(r"^\s*(\d+)\s+points\s*$", s)
        m_lns = re.match(r"^\s*(\d+)\s+lines\s*$", s)
        if m_pts:
            npts = int(m_pts.group(1)); continue
        if m_lns:
            nlines = int(m_lns.group(1)); continue
        if s == "points":
            i_points = i + 1; continue
        if s == "lines":
            i_lines = i + 1; continue

    if i_points is None or i_lines is None or npts is None or nlines is None:
        raise RuntimeError(f"Could not parse surface file: {path}")

    pts = {}
    k = i_points
    while k < len(lines) and len(pts) < npts:
        ln = lines[k].strip(); k += 1
        if not ln: continue
        parts = ln.split()
        if len(parts) < 3: continue
        try:
            pid = int(parts[0])
            pts[pid] = (float(parts[1]), float(parts[2]))
        except ValueError:
            continue

    seg_r, seg_z = [], []
    line_pairs = []
    k = i_lines; nread = 0
    while k < len(lines) and nread < nlines:
        ln = lines[k].strip(); k += 1
        if not ln: continue
        parts = ln.split()
        if len(parts) < 3: continue
        try:
            p1 = int(parts[1]); p2 = int(parts[2])
        except ValueError:
            continue
        if p1 in pts and p2 in pts:
            r1, z1 = pts[p1]; r2, z2 = pts[p2]
            seg_r.extend([r1, r2, np.nan]); seg_z.extend([z1, z2, np.nan])
            line_pairs.append((p1, p2))
        nread += 1

    poly_ids = []
    if line_pairs:
        poly_ids = [line_pairs[0][0], line_pairs[0][1]]
        remaining = line_pairs[1:]
        while remaining:
            last = poly_ids[-1]; match = None
            for i, (a, b) in enumerate(remaining):
                if a == last: poly_ids.append(b); match = i; break
                if b == last: poly_ids.append(a); match = i; break
            if match is None: break
            remaining.pop(match)
            if poly_ids[-1] == poly_ids[0]: break
    if not poly_ids: poly_ids = sorted(pts)

    poly = np.array([pts[i] for i in poly_ids if i in pts], dtype=float)
    if poly.size and not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    return np.array(seg_r), np.array(seg_z), poly


def parse_grid_dump(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    blocks = {}
    while i < len(lines):
        if lines[i].strip() != "ITEM: TIMESTEP":
            i += 1; continue
        ts = int(lines[i + 1].strip()); i += 2

        if lines[i].strip() != "ITEM: NUMBER OF CELLS":
            raise RuntimeError("Bad dump format: NUMBER OF CELLS")
        nc = int(lines[i + 1].strip()); i += 2

        if not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError("Bad dump format: BOX BOUNDS")
        i += 4

        if not lines[i].startswith("ITEM: CELLS"):
            raise RuntimeError("Bad dump format: CELLS header")
        header = lines[i].split()[2:]
        i += 1

        cols = [[] for _ in header]
        for _ in range(nc):
            row = lines[i].split(); i += 1
            for j in range(len(header)):
                cols[j].append(float(row[j]))
        blocks[ts] = (header, [np.array(c) for c in cols])

    if not blocks: raise RuntimeError(f"No timestep blocks found in {path}")
    return blocks


def default_display_bins(xmin, xmax, ymin, ymax, target, npts):
    target = max(32, int(target))
    dx = max(xmax - xmin, 1e-12); dy = max(ymax - ymin, 1e-12)
    total_bins = min(target * target, max(32 * 32, int(np.ceil(max(npts, 1) / 1.5))))
    if dx >= dy:
        nx = max(32, min(target, int(round(np.sqrt(total_bins * dx / dy)))))
        ny = max(32, min(target, int(round(total_bins / nx))))
    else:
        ny = max(32, min(target, int(round(np.sqrt(total_bins * dy / dx)))))
        nx = max(32, min(target, int(round(total_bins / ny))))
    return nx, ny


def make_binned_grid(xc, yc, val, nx, ny, reducer, bounds):
    xmin, xmax, ymin, ymax = bounds
    x_edges = np.linspace(xmin, xmax, nx + 1)
    y_edges = np.linspace(ymin, ymax, ny + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:]); y = 0.5 * (y_edges[:-1] + y_edges[1:])
    sum_grid = np.histogram2d(yc, xc, bins=[y_edges, x_edges], weights=val)[0]
    count_grid = np.histogram2d(yc, xc, bins=[y_edges, x_edges])[0]
    valid = count_grid > 0.0
    if reducer == "sum": return x, y, sum_grid, valid
    if reducer == "max":
        ix = np.searchsorted(x_edges, xc, side="right") - 1
        iy = np.searchsorted(y_edges, yc, side="right") - 1
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        grid = np.full((ny, nx), -np.inf, dtype=float)
        np.maximum.at(grid, (iy[ok], ix[ok]), val[ok])
        valid = np.isfinite(grid); grid[~valid] = 0.0
        return x, y, grid, valid
    grid = np.divide(sum_grid, count_grid, out=np.zeros_like(sum_grid), where=valid)
    return x, y, grid, valid


def gaussian_kernel1d(sigma):
    if sigma <= 0: return np.array([1.0])
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=float)
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    return ker / ker.sum()


def convolve1d_reflect(arr, ker, axis):
    pad = len(ker) // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="reflect")
    return np.apply_along_axis(lambda m: np.convolve(m, ker, mode="valid"), axis, padded)


def smooth_masked_grid(grid, valid, sigma):
    if sigma <= 0: return grid, valid
    ker = gaussian_kernel1d(sigma)
    num = convolve1d_reflect(np.where(valid, grid, 0.0), ker, 0)
    num = convolve1d_reflect(num, ker, 1)
    w = convolve1d_reflect(valid.astype(float), ker, 0)
    w = convolve1d_reflect(w, ker, 1)
    out = np.divide(num, w, out=np.zeros_like(num), where=w > 1e-12)
    return out, w > 1e-6


def centers_to_edges(v):
    if v.size == 0: return np.array([])
    if v.size == 1: return np.array([v[0] - 0.5, v[0] + 0.5])
    mids = 0.5 * (v[:-1] + v[1:])
    first = v[0] - 0.5 * (v[1] - v[0])
    last  = v[-1] + 0.5 * (v[-1] - v[-2])
    return np.concatenate(([first], mids, [last]))


def mask_outside_wall(x, y, grid, polygon, valid):
    if polygon.size == 0:
        return np.ma.masked_where(~valid, grid)
    rs, zs = np.meshgrid(x, y)
    path = MplPath(polygon)
    inside = path.contains_points(
        np.column_stack([rs.ravel(), zs.ravel()]), radius=1e-10
    ).reshape(grid.shape)
    return np.ma.masked_where((~inside) | (~valid), grid)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(here / "output" / "grid.west"))
    ap.add_argument("--wall", default=str(here / "input" / "wall.surf"))
    ap.add_argument("--timestep", default="last")
    ap.add_argument("--cols", nargs="+", default=None,
                    help="Column names (from header) to plot")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--smooth", type=float, default=None)
    ap.add_argument("--bins", nargs=2, type=int, default=None)
    ap.add_argument("--target-bins", type=int, default=220)
    ap.add_argument("--bin-reduce", choices=("mean", "sum", "max"), default="mean")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--log-span", type=float, default=4.0)
    ap.add_argument("--log-vmax-quantile", type=float, default=99.5)
    ap.add_argument("--min-density", type=float, default=0.0)
    ap.add_argument("--cmap", default="plasma")
    ap.add_argument("--out", default="output/grid_plot.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--puff-point", nargs=2, type=float, default=None,
                    help="R Z of puff marker to overlay (optional)")
    args = ap.parse_args()

    blocks = parse_grid_dump(Path(args.dump))
    ts = max(blocks.keys()) if args.timestep == "last" else int(args.timestep)
    header, cols = blocks[ts]
    hmap = {h: i for i, h in enumerate(header)}

    xc = cols[hmap["xc"]]; yc = cols[hmap["yc"]]
    if args.cols is None:
        val_names = [h for h in header if h not in ("id", "xc", "yc", "zc")]
    else:
        val_names = args.cols
    values = [cols[hmap[n]] for n in val_names]
    labels = args.labels if args.labels else val_names

    wall_r, wall_z, wall_poly = parse_surface(Path(args.wall))

    xmin = float(np.min(xc)); xmax = float(np.max(xc))
    ymin = float(np.min(yc)); ymax = float(np.max(yc))
    if wall_poly.size:
        xmin = min(xmin, float(wall_poly[:, 0].min())); xmax = max(xmax, float(wall_poly[:, 0].max()))
        ymin = min(ymin, float(wall_poly[:, 1].min())); ymax = max(ymax, float(wall_poly[:, 1].max()))

    if args.bins: bin_nx, bin_ny = args.bins
    else: bin_nx, bin_ny = default_display_bins(xmin, xmax, ymin, ymax,
                                                 args.target_bins, len(xc))
    smooth_sigma = args.smooth if args.smooth is not None else 0.6

    nsp = len(values)
    fig, axs = plt.subplots(1, nsp, figsize=(5.8 * nsp, 5.6), constrained_layout=True)
    if nsp == 1: axs = [axs]

    for s, dens in enumerate(values):
        ax = axs[s]
        x, y, grid, valid = make_binned_grid(xc, yc, dens, bin_nx, bin_ny,
                                              args.bin_reduce,
                                              (xmin, xmax, ymin, ymax))
        if smooth_sigma > 0:
            grid, valid = smooth_masked_grid(grid, valid, smooth_sigma)

        grid_plot = mask_outside_wall(x, y, grid, wall_poly, valid)
        mask_floor = max(args.min_density, 0.0)

        vmin = vmax = None
        if args.log:
            grid_plot = np.ma.masked_less_equal(grid_plot, mask_floor)
            grid_plot = np.ma.log10(grid_plot)
            log_vals = np.asarray(grid_plot.compressed(), dtype=float)
            if log_vals.size > 0:
                vmax = (float(np.max(log_vals)) if args.log_vmax_quantile >= 100
                        else float(np.percentile(log_vals, args.log_vmax_quantile)))
                vmin = float(np.min(log_vals))
                if args.log_span > 0: vmin = max(vmin, vmax - args.log_span)
            cbar_label = r"$\log_{10}$ value"
        elif mask_floor > 0:
            grid_plot = np.ma.masked_less_equal(grid_plot, mask_floor)
            cbar_label = "value"
        else:
            cbar_label = "value"

        r_edges, z_edges = np.meshgrid(centers_to_edges(x), centers_to_edges(y))
        m = ax.pcolormesh(r_edges, z_edges, grid_plot, shading="flat",
                          cmap=args.cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(m, ax=ax, pad=0.01, fraction=0.04, shrink=0.85,
                     label=cbar_label)

        ax.plot(wall_r, wall_z, "k", lw=1.5)
        if args.puff_point is not None:
            ax.plot(args.puff_point[0], args.puff_point[1], marker='*',
                    markersize=20, color='yellow', markeredgecolor='k',
                    markeredgewidth=1.2, label='D$_2$ puff')
            ax.legend(loc='upper right', fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(labels[s], fontsize=12)
        ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
        ax.grid(alpha=0.25, linestyle="--")

    title = args.title if args.title else f"WEST grid dump, timestep={ts}"
    fig.suptitle(title, fontsize=14)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
