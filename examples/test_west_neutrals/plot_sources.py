#!/usr/bin/env python3
"""Plot per-cell volumetric source rates (particle + power) on WEST geometry.

Uses the same binned-mean + Gaussian-smooth + pcolormesh stack as
test_west_axi/analysis/plot_grid_density_west.py, inlined so this script is
self-contained (can be scp'd and run wherever grid.west lives).

Dump layout produced by `in.west_neutrals`:
    cols 1-3    : id xc yc
    cols 4-6    : f_fden[*]   (D2, D, D+ number densities)
    cols 7-26   : f_frate[*]  = chem/adas rate tally, `units rate`
                    7-10   particle source   [m^-3 s^-1]   (ion, rec, CX, dis)
                    11-14  momentum source x [kg m^-2 s^-2]
                    15-18  momentum source y
                    19-22  momentum source z
                    23-26  energy source     [W/m^3]
    col  27     : vol
    cols 28-42  : f_fmom[*]

Top row: particle source rate S_n [m^-3 s^-1].
Bottom row: volumetric power density S_E [W m^-3].
"""

import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

DUMP = 'output/grid.west'
WALL = 'input/wall.surf'
CORE = 'input/core.surf'
OUT  = 'output/west_source_terms.png'

DT = 1e-8                      # timestep [s] -- title only
TARGET_BINS = 220              # display bins on the longer axis
SMOOTH_SIGMA = 0.8             # Gaussian sigma (in display cells)
LOG_SPAN = 4.0                 # log-color dynamic range (decades)
LOG_VMAX_Q = 99.5              # percentile anchoring upper color limit


# ----- dump parsing ---------------------------------------------------------

def parse_last_snapshot(path):
    timesteps = []
    current = None
    state = None
    with open(path) as f:
        for line in f:
            if line.startswith('ITEM: TIMESTEP'):
                current = {'time': None, 'cells': []}
                timesteps.append(current); state = 'time'; continue
            if line.startswith('ITEM: NUMBER OF CELLS'):
                state = 'ncells'; continue
            if line.startswith('ITEM: BOX BOUNDS'):
                state = 'bounds'; continue
            if line.startswith('ITEM: CELLS'):
                state = 'cells'; continue
            parts = line.split()
            if not parts:
                continue
            if state == 'time':
                current['time'] = int(parts[0]); state = None
            elif state == 'ncells':
                state = None
            elif state == 'cells' and len(parts) >= 26:
                try:
                    current['cells'].append([float(x) for x in parts])
                except ValueError:
                    pass
    return timesteps[-1]['time'], np.array(timesteps[-1]['cells'])


# ----- surface parsing (adapted from plot_grid_density_west.py) -------------

def parse_surface(path: Path):
    if not path.exists():
        return np.array([]), np.array([]), np.empty((0, 2))
    lines = path.read_text(encoding='utf-8').splitlines()
    npts = nlines = None
    i_points = i_lines = None
    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        m_pts = re.match(r'^\s*(\d+)\s+points\s*$', s)
        m_lns = re.match(r'^\s*(\d+)\s+lines\s*$', s)
        if m_pts: npts = int(m_pts.group(1)); continue
        if m_lns: nlines = int(m_lns.group(1)); continue
        if s == 'points': i_points = i + 1; continue
        if s == 'lines':  i_lines  = i + 1; continue
    if i_points is None or i_lines is None:
        return np.array([]), np.array([]), np.empty((0, 2))

    pts = {}
    k = i_points
    while k < len(lines) and len(pts) < (npts or 0):
        parts = lines[k].split(); k += 1
        if len(parts) < 3: continue
        try:
            pts[int(parts[0])] = (float(parts[1]), float(parts[2]))
        except ValueError:
            continue

    seg_r, seg_z, pairs = [], [], []
    k = i_lines; nread = 0
    while k < len(lines) and nread < (nlines or 0):
        parts = lines[k].split(); k += 1
        if len(parts) < 3: continue
        try:
            p1 = int(parts[1]); p2 = int(parts[2])
        except ValueError:
            continue
        if p1 in pts and p2 in pts:
            r1, z1 = pts[p1]; r2, z2 = pts[p2]
            seg_r.extend([r1, r2, np.nan])
            seg_z.extend([z1, z2, np.nan])
            pairs.append((p1, p2))
        nread += 1

    poly_ids = []
    if pairs:
        poly_ids = [pairs[0][0], pairs[0][1]]
        remaining = pairs[1:]
        while remaining:
            last = poly_ids[-1]; match = None
            for i, (a, b) in enumerate(remaining):
                if a == last: poly_ids.append(b); match = i; break
                if b == last: poly_ids.append(a); match = i; break
            if match is None: break
            remaining.pop(match)
            if poly_ids[-1] == poly_ids[0]: break

    poly = np.array([pts[i] for i in poly_ids if i in pts], dtype=float)
    if poly.size and not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return np.array(seg_r), np.array(seg_z), poly


# ----- binning / smoothing / masking (from plot_grid_density_west.py) ------

def centers_to_edges(v):
    if v.size == 0: return np.array([], dtype=float)
    if v.size == 1: return np.array([v[0] - 0.5, v[0] + 0.5], dtype=float)
    mids = 0.5 * (v[:-1] + v[1:])
    first = v[0] - 0.5 * (v[1] - v[0])
    last  = v[-1] + 0.5 * (v[-1] - v[-2])
    return np.concatenate(([first], mids, [last]))


def default_display_bins(xmin, xmax, ymin, ymax, target, npts):
    target = max(32, int(target))
    dx = max(xmax - xmin, 1e-12); dy = max(ymax - ymin, 1e-12)
    total = min(target * target, max(32 * 32, int(np.ceil(max(npts, 1) / 1.5))))
    if dx >= dy:
        nx = max(32, min(target, int(round(np.sqrt(total * dx / dy)))))
        ny = max(32, min(target, int(round(total / nx))))
    else:
        ny = max(32, min(target, int(round(np.sqrt(total * dy / dx)))))
        nx = max(32, min(target, int(round(total / ny))))
    return nx, ny


def make_binned_grid(xc, yc, val, nx, ny, bounds):
    xmin, xmax, ymin, ymax = bounds
    x_edges = np.linspace(xmin, xmax, nx + 1)
    y_edges = np.linspace(ymin, ymax, ny + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    sum_g = np.histogram2d(yc, xc, bins=[y_edges, x_edges], weights=val)[0]
    cnt_g = np.histogram2d(yc, xc, bins=[y_edges, x_edges])[0]
    valid = cnt_g > 0.0
    grid = np.divide(sum_g, cnt_g, out=np.zeros_like(sum_g), where=valid)
    return x, y, grid, valid


def _gaussian_kernel1d(sigma):
    if sigma <= 0: return np.array([1.0])
    r = max(1, int(np.ceil(3.0 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum()


def _conv1d(arr, ker, axis):
    pad = len(ker) // 2
    pw = [(0, 0)] * arr.ndim; pw[axis] = (pad, pad)
    padded = np.pad(arr, pw, mode='reflect')
    return np.apply_along_axis(lambda m: np.convolve(m, ker, mode='valid'),
                               axis, padded)


def smooth_masked_grid(grid, valid, sigma):
    if sigma <= 0: return grid, valid
    ker = _gaussian_kernel1d(sigma)
    w = valid.astype(float)
    num = _conv1d(_conv1d(np.where(valid, grid, 0.0), ker, 0), ker, 1)
    den = _conv1d(_conv1d(w, ker, 0), ker, 1)
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return out, den > 1e-6


def mask_outside_wall(x, y, grid, polygon, valid):
    if polygon.size == 0:
        return np.ma.masked_where(~valid, grid)
    rs, zs = np.meshgrid(x, y)
    path = MplPath(polygon)
    inside = path.contains_points(
        np.column_stack([rs.ravel(), zs.ravel()]), radius=1e-10
    ).reshape(grid.shape)
    return np.ma.masked_where(~inside | ~valid, grid)


# ----- rendering ------------------------------------------------------------

def render(ax, xc, yc, val, bounds, wall_poly, wall_r, wall_z,
           core_r, core_z, title, cmap, cbar_label):
    xmin, xmax, ymin, ymax = bounds
    nx, ny = default_display_bins(xmin, xmax, ymin, ymax, TARGET_BINS, len(xc))
    x, y, grid, valid = make_binned_grid(xc, yc, val, nx, ny, bounds)
    grid, valid = smooth_masked_grid(grid, valid, SMOOTH_SIGMA)
    grid = mask_outside_wall(x, y, grid, wall_poly, valid)

    grid = np.ma.masked_less_equal(grid, 0.0)
    grid_log = np.ma.log10(grid)
    compressed = np.asarray(grid_log.compressed(), dtype=float)
    if compressed.size == 0:
        ax.text(0.5, 0.5, 'no events', transform=ax.transAxes,
                ha='center', va='center', color='gray')
    else:
        vmax = float(np.percentile(compressed, LOG_VMAX_Q))
        vmin = max(float(np.min(compressed)), vmax - LOG_SPAN)
        R, Z = np.meshgrid(centers_to_edges(x), centers_to_edges(y))
        m = ax.pcolormesh(R, Z, grid_log, shading='flat',
                          cmap=cmap, vmin=vmin, vmax=vmax)
        cb = plt.colorbar(m, ax=ax, pad=0.01, fraction=0.046, shrink=0.88)
        cb.set_label(cbar_label, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    if wall_r.size:
        ax.plot(wall_r, wall_z, 'k', lw=1.6)
    if core_r.size:
        ax.plot(core_r, core_z, color='cyan', lw=0.9, alpha=0.85)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, linestyle='--')


# ----- main -----------------------------------------------------------------

t, arr = parse_last_snapshot(DUMP)
print(f'Final dump step {t}  ({t*DT*1e3:.2f} ms), cells = {len(arr)}')

x = arr[:, 1]; y = arr[:, 2]

wall_r, wall_z, wall_poly = parse_surface(Path(WALL))
core_r, core_z, _         = parse_surface(Path(CORE))

if wall_poly.size:
    inside = MplPath(wall_poly).contains_points(np.c_[x, y])
    arr = arr[inside]
    x = arr[:, 1]; y = arr[:, 2]
    print(f'  kept {inside.sum()} / {len(inside)} cells inside wall polygon')

rate_n = arr[:, 6:10]     # particle source [m^-3 s^-1]
rate_E = arr[:, 22:26]    # energy   source [W/m^3]
vol    = arr[:, 26]

int_n = (rate_n * vol[:, None]).sum(axis=0)   # [s^-1]
int_E = (rate_E * vol[:, None]).sum(axis=0)   # [W]
rxn = ['Ion', 'Rec', 'CX ', 'Dis']
print('Domain-integrated source rates:')
for i, name in enumerate(rxn):
    print(f'  {name}: S_n = {int_n[i]:.3e} /s   S_E = {int_E[i]:.3e} W')

xmin = float(min(x.min(), wall_poly[:, 0].min())) if wall_poly.size else float(x.min())
xmax = float(max(x.max(), wall_poly[:, 0].max())) if wall_poly.size else float(x.max())
ymin = float(min(y.min(), wall_poly[:, 1].min())) if wall_poly.size else float(y.min())
ymax = float(max(y.max(), wall_poly[:, 1].max())) if wall_poly.size else float(y.max())
bounds = (xmin, xmax, ymin, ymax)

titles = ['Ionization', 'Recombination', 'Charge exchange', 'Dissociation']
cmaps_n = ['Reds', 'Purples', 'Blues', 'Greens']
cmaps_E = ['OrRd', 'BuPu',    'PuBu',  'YlGn']

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
})

fig, axes = plt.subplots(2, 4, figsize=(17, 11), sharex=True, sharey=True,
                          constrained_layout=True)

for j in range(4):
    render(axes[0, j], x, y, rate_n[:, j], bounds,
           wall_poly, wall_r, wall_z, core_r, core_z,
           f'$S_n$ -- {titles[j]}',
           cmaps_n[j],
           r'$\log_{10}\ S_n$  [m$^{-3}$ s$^{-1}$]')
    render(axes[1, j], x, y, rate_E[:, j], bounds,
           wall_poly, wall_r, wall_z, core_r, core_z,
           f'$S_E$ -- {titles[j]}',
           cmaps_E[j],
           r'$\log_{10}\ S_E$  [W m$^{-3}$]')

for ax in axes[-1, :]:
    ax.set_xlabel('R [m]', weight='semibold')
for ax in axes[:, 0]:
    ax.set_ylabel('Z [m]', weight='semibold')

fig.suptitle(
    f'WEST neutral-transport volumetric source rates  '
    f'(smoothed, t = {t*DT*1e3:.2f} ms)\n'
    r'Top: particle source $S_n$ [m$^{-3}$ s$^{-1}$].  '
    r'Bottom: power density $S_E$ [W m$^{-3}$].',
    fontsize=13)

out = Path(OUT)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches='tight')
print(f'Saved {out}')
