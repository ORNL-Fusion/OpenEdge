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
import matplotlib.pyplot as plt
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
        return None, None, None, None

    ix = np.searchsorted(x, xc)
    iy = np.searchsorted(y, yc)
    grid = np.zeros((y.size, x.size), dtype=float)
    counts = np.zeros((y.size, x.size), dtype=float)
    np.add.at(grid, (iy, ix), val)
    np.add.at(counts, (iy, ix), 1.0)
    grid = np.divide(grid, counts, out=np.zeros_like(grid), where=counts > 0.0)
    return x, y, grid, counts > 0.0


def grid_fill_fraction(xc: np.ndarray, yc: np.ndarray) -> float:
    x = np.unique(xc)
    y = np.unique(yc)
    if x.size == 0 or y.size == 0:
        return 0.0
    return float(len(xc)) / float(x.size * y.size)


def centers_to_edges(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return np.array([], dtype=float)
    if v.size == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5], dtype=float)
    mids = 0.5 * (v[:-1] + v[1:])
    first = v[0] - 0.5 * (v[1] - v[0])
    last = v[-1] + 0.5 * (v[-1] - v[-2])
    return np.concatenate(([first], mids, [last]))


def default_display_bins(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    target: int,
    npts: int,
):
    target = max(32, int(target))
    dx = max(xmax - xmin, 1.0e-12)
    dy = max(ymax - ymin, 1.0e-12)
    total_bins = min(target * target, max(32 * 32, int(np.ceil(max(npts, 1) / 1.5))))
    if dx >= dy:
        nx = max(32, min(target, int(round(np.sqrt(total_bins * dx / dy)))))
        ny = max(32, min(target, int(round(total_bins / nx))))
    else:
        ny = max(32, min(target, int(round(np.sqrt(total_bins * dy / dx)))))
        nx = max(32, min(target, int(round(total_bins / ny))))
    return nx, ny


def make_binned_grid(
    xc: np.ndarray,
    yc: np.ndarray,
    val: np.ndarray,
    nx: int,
    ny: int,
    reducer: str,
    bounds,
):
    xmin, xmax, ymin, ymax = bounds
    x_edges = np.linspace(xmin, xmax, nx + 1)
    y_edges = np.linspace(ymin, ymax, ny + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])

    if reducer == "max":
        ix = np.searchsorted(x_edges, xc, side="right") - 1
        iy = np.searchsorted(y_edges, yc, side="right") - 1
        valid_pts = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        grid = np.full((ny, nx), -np.inf, dtype=float)
        np.maximum.at(grid, (iy[valid_pts], ix[valid_pts]), val[valid_pts])
        valid = np.isfinite(grid)
        grid[~valid] = 0.0
        return x, y, grid, valid

    sum_grid = np.histogram2d(yc, xc, bins=[y_edges, x_edges], weights=val)[0]
    count_grid = np.histogram2d(yc, xc, bins=[y_edges, x_edges])[0]
    valid = count_grid > 0.0

    if reducer == "sum":
        return x, y, sum_grid, valid

    grid = np.divide(sum_grid, count_grid, out=np.zeros_like(sum_grid), where=valid)
    return x, y, grid, valid


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


def smooth_masked_grid(grid: np.ndarray, valid: np.ndarray, sigma: float):
    if sigma <= 0.0:
        return grid, valid
    weights = valid.astype(float)
    smoothed_num = smooth_grid(np.where(valid, grid, 0.0), sigma)
    smoothed_den = smooth_grid(weights, sigma)
    out = np.divide(
        smoothed_num,
        smoothed_den,
        out=np.zeros_like(smoothed_num),
        where=smoothed_den > 1.0e-12,
    )
    return out, smoothed_den > 1.0e-6


def coarsen_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray, valid: np.ndarray, factor: int):
    if factor <= 1:
        return x, y, grid, valid
    ny, nx = grid.shape
    ny2 = ny // factor
    nx2 = nx // factor
    if ny2 < 1 or nx2 < 1:
        return x, y, grid, valid

    ny_trim = ny2 * factor
    nx_trim = nx2 * factor
    x2 = x[:nx_trim].reshape(nx2, factor).mean(axis=1)
    y2 = y[:ny_trim].reshape(ny2, factor).mean(axis=1)
    valid_f = valid[:ny_trim, :nx_trim].astype(float)
    weight2 = valid_f.reshape(ny2, factor, nx2, factor).sum(axis=(1, 3))
    grid2 = np.divide(
        (grid[:ny_trim, :nx_trim] * valid_f).reshape(ny2, factor, nx2, factor).sum(axis=(1, 3)),
        weight2,
        out=np.zeros((ny2, nx2), dtype=float),
        where=weight2 > 0.0,
    )
    return x2, y2, grid2, weight2 > 0.0


def load_psi_map(path: Path):
    """
    Return (r, z, psi_norm) from an OpenEdge plasma.h5 file.
    Tolerates files that only contain /psi + /psicore + /psisep
    (recomputes psi_norm = (psi - psicore) / (psisep - psicore)).
    Returns None if the file lacks psi information.
    """
    try:
        import h5py  # lazy import so pure-density plots don't need h5py
    except ImportError:
        print(f"WARN: h5py not available, skipping psi overlay from {path}")
        return None
    try:
        with h5py.File(str(path), "r") as f:
            if "r" not in f or "z" not in f:
                return None
            r = np.asarray(f["r"][...], dtype=float)
            z = np.asarray(f["z"][...], dtype=float)
            if "psi_norm" in f:
                psi_n = np.asarray(f["psi_norm"][...], dtype=float)
            elif "psi" in f and "psicore" in f and "psisep" in f:
                psi = np.asarray(f["psi"][...], dtype=float)
                psicore = float(np.asarray(f["psicore"][...]).reshape(-1)[0])
                psisep = float(np.asarray(f["psisep"][...]).reshape(-1)[0])
                if psicore == psisep:
                    return None
                psi_n = (psi - psicore) / (psisep - psicore)
            else:
                return None
    except (OSError, KeyError):
        return None
    return r, z, psi_n


def mask_outside_wall(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    polygon: np.ndarray,
    valid: np.ndarray,
):
    if polygon.size == 0:
        return np.ma.masked_where(~valid, grid)
    rs, zs = np.meshgrid(x, y)
    path = MplPath(polygon)
    pts = np.column_stack([rs.ravel(), zs.ravel()])
    inside = path.contains_points(pts, radius=1.0e-10).reshape(grid.shape)
    return np.ma.masked_where((~inside) | (~valid), grid)


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
        "--w0-and-total",
        action="store_true",
        help="plot two panels: W0 only and the sum over all value columns",
    )
    ap.add_argument(
        "--smooth",
        type=float,
        default=None,
        help="Gaussian smoothing sigma in display cells (default: auto for sparse binned plots)",
    )
    ap.add_argument(
        "--coarsen",
        type=int,
        default=1,
        help="block-average the display grid by this factor before plotting",
    )
    ap.add_argument(
        "--plot-mode",
        choices=("auto", "mesh", "binned"),
        default="auto",
        help="auto-detect sparse adaptive grids and render them as a binned heatmap",
    )
    ap.add_argument(
        "--bins",
        nargs=2,
        type=int,
        metavar=("NX", "NY"),
        default=None,
        help="display-grid bins for binned mode (default: auto from aspect ratio)",
    )
    ap.add_argument(
        "--target-bins",
        type=int,
        default=220,
        help="target bin count on the longer axis when --bins is not set",
    )
    ap.add_argument(
        "--bin-reduce",
        choices=("mean", "sum", "max"),
        default="mean",
        help="reducer used to combine simulation cells inside each display bin",
    )
    ap.add_argument(
        "--min-density",
        type=float,
        default=0.0,
        help="mask values below this density before plotting",
    )
    ap.add_argument("--out", default="grid_density_west.png")
    ap.add_argument("--show", action="store_true")
    ap.add_argument(
        "--log",
        action="store_true",
        help="plot log10(density) for positive values",
    )
    ap.add_argument(
        "--log-span",
        type=float,
        default=4.0,
        help="limit log color range to this many decades below the panel maximum; <=0 uses full range",
    )
    ap.add_argument(
        "--log-vmax-quantile",
        type=float,
        default=99.5,
        help="in log mode, anchor the upper color limit to this percentile instead of the absolute max; >=100 uses max",
    )
    ap.add_argument("--xlim", nargs=2, type=float, default=None)
    ap.add_argument("--ylim", nargs=2, type=float, default=None)
    ap.add_argument(
        "--plasma-h5",
        default=None,
        help="plasma.h5 with /psi_norm (+ /psicore/psisep); overlays psi_norm=0 (core) "
        "and psi_norm=1 (separatrix) contours. Default: input/plasma.h5 next to --wall.",
    )
    ap.add_argument(
        "--no-psi-overlay",
        action="store_true",
        help="disable psi contours even if plasma.h5 is found",
    )
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
    if args.sum_all and args.w0_and_total:
        raise RuntimeError("Use only one of --sum-all or --w0-and-total")
    if args.w0_and_total:
        total = np.zeros_like(values[0], dtype=float)
        for arr in values:
            total += arr
        values = [values[0], total]
        names = ["W0", "sum all"]
    elif args.sum_all:
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

    # Resolve psi-overlay source
    psi_overlay = None
    if not args.no_psi_overlay:
        if args.plasma_h5 is not None:
            psi_overlay = load_psi_map(Path(args.plasma_h5))
        else:
            # Guess: input/plasma.h5 next to the wall file
            guess = Path(args.wall).resolve().parent / "plasma.h5"
            if guess.exists():
                psi_overlay = load_psi_map(guess)
        if psi_overlay is not None:
            print(f"Overlaying psi_norm=0 (core) and psi_norm=1 (separatrix) contours")
    fill_fraction = grid_fill_fraction(xc, yc)
    render_mode = args.plot_mode
    if render_mode == "auto":
        render_mode = "binned" if fill_fraction < 0.25 else "mesh"

    xmin = float(np.min(xc))
    xmax = float(np.max(xc))
    ymin = float(np.min(yc))
    ymax = float(np.max(yc))
    if wall_poly.size:
        xmin = min(xmin, float(np.min(wall_poly[:, 0])))
        xmax = max(xmax, float(np.max(wall_poly[:, 0])))
        ymin = min(ymin, float(np.min(wall_poly[:, 1])))
        ymax = max(ymax, float(np.max(wall_poly[:, 1])))
    if args.bins:
        bin_nx, bin_ny = args.bins
    else:
        bin_nx, bin_ny = default_display_bins(
            xmin, xmax, ymin, ymax, args.target_bins, len(xc)
        )
    smooth_sigma = args.smooth
    if smooth_sigma is None:
        smooth_sigma = 0.25 if render_mode == "binned" else 0.0
    print(
        f"Render mode: {render_mode} "
        f"(occupied mesh fraction={fill_fraction:.2%}"
        + (
            f", display bins={bin_nx}x{bin_ny}, smooth={smooth_sigma:g})"
            if render_mode == "binned"
            else ")"
        )
    )

    nsp = len(values)
    fig, axs = plt.subplots(1, nsp, figsize=(5.8 * nsp, 5.6), constrained_layout=True)
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
        vmin = None
        vmax = None
        if render_mode == "binned":
            x, y, grid, valid = make_binned_grid(
                xc,
                yc,
                dens,
                bin_nx,
                bin_ny,
                args.bin_reduce,
                (xmin, xmax, ymin, ymax),
            )
        else:
            x, y, grid, valid = make_grid(xc, yc, dens)
        if x is None:
            ax.set_title(f"{labels[s]} (insufficient grid)")
            continue
        if args.coarsen > 1:
            x, y, grid, valid = coarsen_grid(x, y, grid, valid, args.coarsen)
        if smooth_sigma > 0.0:
            grid, valid = smooth_masked_grid(grid, valid, smooth_sigma)

        grid_plot = mask_outside_wall(x, y, grid, wall_poly, valid)
        mask_floor = max(args.min_density, 0.0)
        if args.log:
            grid_plot = np.ma.masked_less_equal(grid_plot, mask_floor)
            grid_plot = np.ma.log10(grid_plot)
            log_vals = np.asarray(grid_plot.compressed(), dtype=float)
            if log_vals.size > 0:
                if args.log_vmax_quantile >= 100.0:
                    vmax = float(np.max(log_vals))
                else:
                    vmax = float(np.percentile(log_vals, args.log_vmax_quantile))
                vmin = float(np.min(log_vals))
                if args.log_span > 0.0:
                    vmin = max(vmin, vmax - args.log_span)
                print(f"{labels[s]} log10 range: [{vmin:.2f}, {vmax:.2f}]")
            cbar_label = r"$\log_{10}$ Density [m$^{-3}$]"
        elif mask_floor > 0.0:
            grid_plot = np.ma.masked_less_equal(grid_plot, mask_floor)
            cbar_label = "Density [m$^{-3}$]"
        else:
            cbar_label = "Density [m$^{-3}$]"

        r_edges, z_edges = np.meshgrid(centers_to_edges(x), centers_to_edges(y))
        m = ax.pcolormesh(
            r_edges,
            z_edges,
            grid_plot,
            shading="flat",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        cbar = fig.colorbar(m, ax=ax, pad=0.01, fraction=0.04, shrink=0.85)
        cbar.set_label(cbar_label, rotation=270, labelpad=15)

        ax.plot(wall_r, wall_z, "k", lw=2.5)
        if psi_overlay is not None:
            pr, pz, psi_n = psi_overlay
            RR, ZZ = np.meshgrid(pr, pz)
            # Lightly hatch the absorb region (psi_norm < 0) so it's
            # unambiguous what fix reflect/psi removes.
            ax.contourf(
                RR, ZZ, psi_n,
                levels=[-1e6, 0.0],
                colors="none",
                hatches=["////"],
                extend="min",
            )
            ax.contour(RR, ZZ, psi_n, levels=[0.0], colors="cyan",
                       linewidths=1.6, linestyles="-")
            ax.contour(RR, ZZ, psi_n, levels=[1.0], colors="magenta",
                       linewidths=1.6, linestyles="--")
        ax.set_aspect("equal", adjustable="box")
        if args.xlim:
            ax.set_xlim(args.xlim[0], args.xlim[1])
        if args.ylim:
            ax.set_ylim(args.ylim[0], args.ylim[1])
        ax.set_title(labels[s])
        ax.set_xlabel("R [m]", weight="semibold")
        ax.set_ylabel("Z [m]", weight="semibold")
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(f"WEST grid density, timestep={ts}", fontsize=18)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
