#!/usr/bin/env python3
"""
Plot OpenEdge/SPARTA grid density dump in WEST-style R-Z subplots.

Example:
  python3 plot_grid_density_west.py \
    --dump output/grid.dens.A.west \
    --wall input/wall.surf \
    --out output/grid_density.west.png \
    --show
      python3 plot_grid_density_west.py --show --log
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


def resolve_default_dump(example_root: Path) -> Path:
    candidates = sorted((example_root / "output").glob("grid.dens.*.west"))
    if candidates:
        return candidates[-1]
    return example_root / "output" / "grid.dens.A.west"


def resolve_default_wall(example_root: Path) -> Path:
    surf = example_root / "input" / "wall.surf"
    if surf.exists():
        return surf
    return example_root / "input" / "wall.txt"


def map_plot_coords(x: np.ndarray, y: np.ndarray, coord_layout: str):
    if coord_layout == "zr":
        return y, x
    return x, y


def map_surface_coords(seg_a: np.ndarray, seg_b: np.ndarray, poly: np.ndarray, coord_layout: str):
    if coord_layout != "zr":
        return seg_a, seg_b, poly
    poly_out = poly[:, [1, 0]] if poly.size else poly
    return seg_b, seg_a, poly_out


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


def _read_h5_scalar(f, *keys):
    for key in keys:
        if key in f:
            return float(np.asarray(f[key][...]).reshape(-1)[0])
    raise KeyError(keys[0])


def _read_h5_array(f, *keys):
    for key in keys:
        if key in f:
            return np.asarray(f[key][...], dtype=float), key
    raise KeyError(keys[0])


def load_psi_map(path: Path, warn: bool = False):
    """
    Return psi-overlay data from an OpenEdge plasma.h5 file.

    Supports both top-level /r,/z,/psi and SOLEDGE-style /config/r,/config/z,
    /config/psi layouts. If only the /config/* mesh exists and /R0 + /a0 are
    present, coordinates are rescaled into OpenEdge physical meters.

    Returns (r, z, psi, psicore, psisep) where psi is the raw poloidal flux
    field and psicore/psisep are the raw contour levels. Returns None if the
    file lacks enough psi information to draw contours.
    """
    path = path.expanduser()
    if not path.exists():
        if warn:
            print(f"WARN: psi overlay file not found: {path}")
        return None
    try:
        import h5py  # lazy import so pure-density plots don't need h5py
    except ImportError:
        if warn:
            print(f"WARN: h5py not available, skipping psi overlay from {path}")
        return None
    try:
        with h5py.File(str(path), "r") as f:
            try:
                r, r_key = _read_h5_array(f, "r", "config/r")
                z, z_key = _read_h5_array(f, "z", "config/z")
                psi, psi_key = _read_h5_array(f, "psi", "config/psi")
            except KeyError:
                if warn:
                    print(
                        f"WARN: {path} is missing psi coordinates/field; "
                        "expected /r,/z,/psi or /config/r,/config/z,/config/psi"
                    )
                return None

            try:
                psicore = _read_h5_scalar(f, "psicore", "config/psicore")
                psisep = _read_h5_scalar(f, "psisep", "config/psisep1", "config/psisep")
            except KeyError:
                if warn:
                    print(
                        f"WARN: {path} is missing /psicore or /psisep; cannot draw raw psi contours"
                    )
                return None

            if r_key.startswith("config/") and z_key.startswith("config/"):
                if "R0" in f and "a0" in f:
                    r0_init = float(np.asarray(f["R0"][...]).reshape(-1)[0])
                    a0_init = float(np.asarray(f["a0"][...]).reshape(-1)[0])
                    if a0_init != 0.0:
                        r = (r - r0_init) / a0_init
                        z = z / a0_init
                if r.ndim == 2 and z.ndim == 2:
                    r = np.asarray(r[:, 0], dtype=float)
                    z = np.asarray(z[0, :], dtype=float)

            if psicore == psisep:
                if warn:
                    print(f"WARN: {path} has psicore == psisep; cannot draw distinct psi contours")
                return None

            if warn:
                print(
                    f"Using {psi_key} with /psicore={psicore:.6e} "
                    f"and /psisep={psisep:.6e}"
                )
    except (OSError, KeyError):
        if warn:
            print(f"WARN: failed reading psi overlay from {path}")
        return None

    r = np.asarray(r, dtype=float).squeeze()
    z = np.asarray(z, dtype=float).squeeze()
    if r.ndim != 1 or z.ndim != 1:
        if warn:
            print(f"WARN: psi coordinates in {path} are not 1-D after loading")
        return None

    if psi.shape == (r.size, z.size):
        psi = psi.T
    elif psi.shape != (z.size, r.size):
        if warn:
            print(
                f"WARN: psi array shape {psi.shape} is incompatible with "
                f"r/z sizes {(r.size, z.size)} in {path}"
            )
        return None
    return r, z, psi, psicore, psisep


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
    # Script lives in coupled_test/openedge/; wall.surf is at
    # examples/test_solps_coupling/input/, two levels up.
    example_root = here.parent.parent
    default_dump = here / "output" / "grid_density.txt"
    default_wall = resolve_default_wall(example_root)

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
    ap.add_argument(
        "--coord-layout",
        choices=("zr", "rz"),
        default="rz",
        help="physical coordinate order of stored x/y-like columns and wall points: "
        "'rz' (default, OpenEdge 2D Cartesian) means xc=R,yc=Z and wall=(R,Z); "
        "'zr' (legacy axi) means xc=Z,yc=R and wall=(Z,R)",
    )
    ap.add_argument("--timestep", default="last", help="last or explicit integer")
    ap.add_argument("--labels", nargs="*", default=None, help="subplot labels")
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
    ap.add_argument("--out", default="grid_density.png")
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
        help="plasma.h5 with /psi + /psicore + /psisep (or /config/* equivalents); "
        "overlays psi=psicore and psi=psisep. Default: input/plasma.h5 next to --wall.",
    )
    ap.add_argument(
        "--hide-psi-separatrix",
        action="store_true",
        help="hide the /psisep separatrix contour overlay",
    )
    ap.add_argument(
        "--no-psi-overlay",
        action="store_true",
        help="disable psi contours even if plasma.h5 is found",
    )
    args = ap.parse_args()

    dump_path = Path(args.dump).expanduser()
    if not dump_path.exists():
        candidates = sorted((example_root / "output").glob("grid.dens.*.west"))
        hint = ""
        if candidates:
            hint = " Available dumps: " + ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Grid dump not found: {dump_path}.{hint}")

    wall_path = Path(args.wall).expanduser()
    if not wall_path.exists():
        raise FileNotFoundError(f"Wall surface not found: {wall_path}")

    blocks = parse_grid_dump(dump_path)
    ts = max(blocks.keys()) if args.timestep == "last" else int(args.timestep)
    header, cols = blocks[ts]
    cmap = "plasma"

    # Expected: id xc yc f_fixID[*] ... or more fields
    hmap = {h: i for i, h in enumerate(header)}
    if "xc" not in hmap or "yc" not in hmap:
        raise RuntimeError("CELLS header must include xc and yc")
    xc_raw = cols[hmap["xc"]]
    yc_raw = cols[hmap["yc"]]
    xc, yc = map_plot_coords(xc_raw, yc_raw, args.coord_layout)

    val_idx = [i for i, h in enumerate(header) if h not in ("id", "xc", "yc", "zc")]
    if not val_idx:
        raise RuntimeError("No density/value columns found")

    values = [cols[i] for i in val_idx]
    names = [header[i] for i in val_idx]
    total = np.zeros_like(values[0], dtype=float)
    for arr in values:
        total += arr

    if args.w0_and_total:
        values = [values[0], total]
        names = ["mist", "total Li"]
    else:
        values = [total]
        names = ["total Li"]

    if args.labels:
        labels = args.labels[: len(values)]
        if len(labels) < len(values):
            labels += names[len(labels):]
    else:
        labels = names

    wall_a, wall_b, wall_poly_raw = parse_surface(wall_path)
    wall_r, wall_z, wall_poly = map_surface_coords(
        wall_a, wall_b, wall_poly_raw, args.coord_layout
    )

    # Resolve psi-overlay source
    psi_overlay = None
    psi_overlay_src = None
    if not args.no_psi_overlay:
        if args.plasma_h5 is not None:
            psi_overlay_src = Path(args.plasma_h5).expanduser()
            psi_overlay = load_psi_map(psi_overlay_src, warn=True)
        else:
            # Guess: input/plasma.h5 next to the wall file
            guess = wall_path.resolve().parent / "plasma.h5"
            if guess.exists():
                psi_overlay_src = guess
                psi_overlay = load_psi_map(guess, warn=True)
        if psi_overlay is not None:
            print(
                "Overlaying psi=psicore core cutoff and psi=psisep "
                f"separatrix contours from {psi_overlay_src}"
            )
            if args.hide_psi_separatrix:
                print("Separatrix overlay disabled by --hide-psi-separatrix")
    fill_fraction = grid_fill_fraction(xc, yc)
    render_mode = args.plot_mode
    if render_mode == "auto":
        render_mode = "binned" if fill_fraction < 0.25 else "mesh"

    data_xmin = float(np.min(xc))
    data_xmax = float(np.max(xc))
    data_ymin = float(np.min(yc))
    data_ymax = float(np.max(yc))
    if wall_poly.size:
        xmin = float(np.min(wall_poly[:, 0]))
        xmax = float(np.max(wall_poly[:, 0]))
        ymin = float(np.min(wall_poly[:, 1]))
        ymax = float(np.max(wall_poly[:, 1]))
    else:
        xmin = data_xmin
        xmax = data_xmax
        ymin = data_ymin
        ymax = data_ymax
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
    print(
        "Coordinate layout: "
        + ("xc=Z,yc=R -> plotting R horizontal, Z vertical" if args.coord_layout == "zr"
           else "xc=R,yc=Z -> plotting R horizontal, Z vertical")
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
            pr, pz, psi_raw, psicore, psisep = psi_overlay
            RR, ZZ = np.meshgrid(pr, pz)
            psi_core_side = np.sign(psisep - psicore) * (psi_raw - psicore)
            # Lightly hatch the absorbed core-side region so it's unambiguous
            # what fix reflect/psi removes.
            ax.contourf(
                RR, ZZ, psi_core_side,
                levels=[-1e6, 0.0],
                colors="none",
                hatches=["////"],
                extend="min",
            )
            ax.contour(
                RR,
                ZZ,
                psi_raw,
                levels=[psicore],
                colors="cyan",
                linewidths=1.6,
                linestyles="-",
            )
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color="cyan",
                    lw=1.8,
                    linestyle="-",
                    label=r"$\psi=\psi_{\mathrm{core}}$",
                ),
            ]
            if not args.hide_psi_separatrix:
                ax.contour(
                    RR,
                    ZZ,
                    psi_raw,
                    levels=[psisep],
                    colors="magenta",
                    linewidths=1.6,
                    linestyles="--",
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="magenta",
                        lw=1.8,
                        linestyle="--",
                        label=r"$\psi=\psi_{\mathrm{sep}}$",
                    )
                )
            ax.legend(handles=legend_handles, loc="upper right", framealpha=0.85, fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        if args.xlim:
            ax.set_xlim(args.xlim[0], args.xlim[1])
        else:
            ax.set_xlim(xmin, xmax)
        if args.ylim:
            ax.set_ylim(args.ylim[0], args.ylim[1])
        else:
            ax.set_ylim(ymin, ymax)
        ax.set_title(labels[s])
        ax.set_xlabel("R [m]", weight="semibold")
        ax.set_ylabel("Z [m]", weight="semibold")
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(f"Li droplet grid density, timestep={ts}", fontsize=18)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()


