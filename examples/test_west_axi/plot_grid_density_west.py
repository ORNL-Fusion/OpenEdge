#!/usr/bin/env python3
"""
Plot OpenEdge/SPARTA grid density dump in WEST-style R-Z subplots.

Example:
  python3 plot_grid_density_west.py \
    --dump output/tmp.grid.density \
    --wall input/wall.txt \
    --out output/grid_density.west.png \
    --show
      python3 plot_grid_density_west.py --show --log --sum-all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path as MplPath


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
            npts = int(m_pts.group(1))
            continue
        if m_lns:
            nlines = int(m_lns.group(1))
            continue
        if s == "points":
            i_points = i + 1
            continue
        if s == "lines":
            i_lines = i + 1
            continue

    if i_points is None or i_lines is None or npts is None or nlines is None:
        raise RuntimeError(
            f"Could not parse surface file: {path} "
            f"(need '<N> points', '<N> lines', 'Points', 'Lines' sections)"
        )

    pts = {}
    k = i_points
    while k < len(lines) and len(pts) < npts:
        ln = lines[k].strip()
        k += 1
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            pts[pid] = (float(parts[1]), float(parts[2]))
        except ValueError:
            continue

    seg_r = []
    seg_z = []
    line_pairs = []
    k = i_lines
    nread = 0
    while k < len(lines) and nread < nlines:
        ln = lines[k].strip()
        k += 1
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            p1 = int(parts[1])
            p2 = int(parts[2])
        except ValueError:
            continue
        if p1 in pts and p2 in pts:
            r1, z1 = pts[p1]
            r2, z2 = pts[p2]
            seg_r.extend([r1, r2, np.nan])
            seg_z.extend([z1, z2, np.nan])
            line_pairs.append((p1, p2))
        nread += 1

    poly_ids = []
    if line_pairs:
        poly_ids = [line_pairs[0][0], line_pairs[0][1]]
        remaining = line_pairs[1:]
        while remaining:
            last = poly_ids[-1]
            match = None
            for i, (a, b) in enumerate(remaining):
                if a == last:
                    poly_ids.append(b)
                    match = i
                    break
                if b == last:
                    poly_ids.append(a)
                    match = i
                    break
            if match is None:
                break
            remaining.pop(match)
            if poly_ids[-1] == poly_ids[0]:
                break
    if not poly_ids:
        poly_ids = sorted(pts)

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
            i += 1
            continue
        ts = int(lines[i + 1].strip())
        i += 2

        if lines[i].strip() != "ITEM: NUMBER OF CELLS":
            raise RuntimeError("Bad dump format: NUMBER OF CELLS")
        nc = int(lines[i + 1].strip())
        i += 2

        if not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError("Bad dump format: BOX BOUNDS")
        i += 4  # ITEM + 3 lines

        if not lines[i].startswith("ITEM: CELLS"):
            raise RuntimeError("Bad dump format: CELLS header")
        header = lines[i].split()[2:]
        i += 1

        cols = [[] for _ in header]
        for _ in range(nc):
            row = lines[i].split()
            i += 1
            for j in range(len(header)):
                cols[j].append(float(row[j]))
        blocks[ts] = (header, [np.array(c) for c in cols])

    if not blocks:
        raise RuntimeError(f"No timestep blocks found in {path}")
    return blocks


def make_grid(xc, yc, val):
    x = np.unique(xc)
    y = np.unique(yc)
    if x.size < 2 or y.size < 2:
        return None, None, None

    ix = np.searchsorted(x, xc)
    iy = np.searchsorted(y, yc)
    grid = np.zeros((y.size, x.size), dtype=float)
    np.add.at(grid, (iy, ix), val)
    xx, yy = np.meshgrid(x, y)
    return xx, yy, grid


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker /= np.sum(ker)
    return ker


def convolve1d_reflect(arr: np.ndarray, ker: np.ndarray, axis: int) -> np.ndarray:
    pad = len(ker) // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="reflect")
    return np.apply_along_axis(lambda m: np.convolve(m, ker, mode="valid"), axis, padded)


def smooth_grid(grid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return grid
    ker = gaussian_kernel1d(sigma)
    out = convolve1d_reflect(grid, ker, axis=0)
    out = convolve1d_reflect(out, ker, axis=1)
    return out


def coarsen_grid(rs: np.ndarray, zs: np.ndarray, grid: np.ndarray, factor: int):
    if factor <= 1:
        return rs, zs, grid
    ny, nx = grid.shape
    ny2 = ny // factor
    nx2 = nx // factor
    if ny2 < 1 or nx2 < 1:
        return rs, zs, grid

    ny_trim = ny2 * factor
    nx_trim = nx2 * factor
    rs2 = rs[:ny_trim, :nx_trim].reshape(ny2, factor, nx2, factor).mean(axis=(1, 3))
    zs2 = zs[:ny_trim, :nx_trim].reshape(ny2, factor, nx2, factor).mean(axis=(1, 3))
    grid2 = grid[:ny_trim, :nx_trim].reshape(ny2, factor, nx2, factor).mean(axis=(1, 3))
    return rs2, zs2, grid2


def mask_outside_wall(rs: np.ndarray, zs: np.ndarray, grid: np.ndarray, polygon: np.ndarray):
    if polygon.size == 0:
        return np.ma.array(grid, copy=False)
    path = MplPath(polygon)
    pts = np.column_stack([rs.ravel(), zs.ravel()])
    inside = path.contains_points(pts, radius=1.0e-10).reshape(grid.shape)
    return np.ma.masked_where(~inside, grid)


def main():
    here = Path(__file__).resolve().parent
    default_dump = here / "output" / "tmp.grid.density"
    default_wall = here / "input" / "wall.txt"

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump",
        default=str(default_dump),
        help=f"grid dump file (default: {default_dump})",
    )
    ap.add_argument(
        "--wall",
        default=str(default_wall),
        help=f"wall surface file (default: {default_wall})",
    )
    ap.add_argument("--timestep", default="last", help="last or explicit integer")
    ap.add_argument("--labels", nargs="*", default=None, help="subplot labels")
    ap.add_argument(
        "--sum-all",
        action="store_true",
        help="sum all value columns and plot a single total-density panel",
    )
    ap.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma in grid cells",
    )
    ap.add_argument(
        "--coarsen",
        type=int,
        default=1,
        help="block-average grid by this factor before plotting",
    )
    ap.add_argument(
        "--min-density",
        type=float,
        default=0.0,
        help="mask values below this density before plotting",
    )
    ap.add_argument("--out", default="grid_density_west.png")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--log", action="store_true", help="use LogNorm for positive values")
    ap.add_argument("--xlim", nargs=2, type=float, default=None)
    ap.add_argument("--ylim", nargs=2, type=float, default=None)
    args = ap.parse_args()

    blocks = parse_grid_dump(Path(args.dump))
    ts = max(blocks.keys()) if args.timestep == "last" else int(args.timestep)
    header, cols = blocks[ts]
    cmap = "plasma"

    # Expected: id xc yc f_fixID[*] ... or more fields
    hmap = {h: i for i, h in enumerate(header)}
    if "xc" not in hmap or "yc" not in hmap:
        raise RuntimeError("CELLS header must include xc and yc")
    xc = cols[hmap["xc"]]
    yc = cols[hmap["yc"]]

    val_idx = [i for i, h in enumerate(header) if h not in ("id", "xc", "yc", "zc")]
    if not val_idx:
        raise RuntimeError("No density/value columns found")

    values = [cols[i] for i in val_idx]
    names = [header[i] for i in val_idx]
    if args.sum_all:
        total = np.zeros_like(values[0], dtype=float)
        for arr in values:
            total += arr
        values = [total]
        names = ["total tungsten"]

    if args.labels:
        labels = args.labels[: len(values)]
        if len(labels) < len(values):
            labels += names[len(labels):]
    else:
        labels = names

    wall_r, wall_z, wall_poly = parse_surface(Path(args.wall))

    nsp = len(values)
    fig, axs = plt.subplots(1, nsp, figsize=(5.8 * nsp, 5.0), constrained_layout=True)
    if nsp == 1:
        axs = [axs]

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    for s, dens in enumerate(values):
        ax = axs[s]
        mesh = make_grid(xc, yc, dens)
        if mesh[0] is None:
            ax.set_title(f"{labels[s]} (insufficient grid)")
            continue
        rs, zs, grid = mesh
        if args.coarsen > 1:
            rs, zs, grid = coarsen_grid(rs, zs, grid, args.coarsen)
        if args.smooth > 0.0:
            grid = smooth_grid(grid, args.smooth)

        grid_plot = mask_outside_wall(rs, zs, grid, wall_poly)
        mask_floor = max(args.min_density, 0.0)
        if mask_floor > 0.0:
            grid_plot = np.ma.masked_less_equal(grid_plot, mask_floor)

        pos = np.asarray(grid_plot.compressed(), dtype=float)
        if args.log and pos.size > 0:
            norm = LogNorm(vmin=np.min(pos), vmax=np.max(pos))
            m = ax.pcolormesh(rs, zs, grid_plot, shading="auto", cmap=cmap, norm=norm)
        else:
            m = ax.pcolormesh(rs, zs, grid_plot, shading="auto", cmap=cmap)

        cbar = fig.colorbar(m, ax=ax, pad=0.01, fraction=0.04, shrink=0.85)
        cbar.set_label("Density [m$^{-3}$]", rotation=270, labelpad=15)

        ax.plot(wall_r, wall_z, "k", lw=2.5)
        ax.set_aspect("equal", adjustable="box")
        if args.xlim:
            ax.set_xlim(args.xlim[0], args.xlim[1])
        if args.ylim:
            ax.set_ylim(args.ylim[0], args.ylim[1])
        ax.set_title(labels[s])
        ax.set_xlabel("R [m]", weight="semibold")
        ax.set_ylabel("Z [m]", weight="semibold")
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(f"WEST grid density, timestep={ts}", y=1.02, fontsize=18)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
