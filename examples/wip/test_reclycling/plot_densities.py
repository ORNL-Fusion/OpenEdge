#!/usr/bin/env python3
"""Plot D2, D, D+ density maps from test_reclycling output at the final timestep.

Also overlays the wall and core surfaces.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

DUMP = 'output/grid.recycling'
WALL = 'wall.surf'
CORE = 'core.surf'
OUT  = 'output/densities_map.png'


def parse_last_snapshot(path):
    """Return arrays xc, yc, nD2, nD, nDp for the last TIMESTEP block."""
    timesteps = []
    current = None
    with open(path) as f:
        for line in f:
            if line.startswith('ITEM: TIMESTEP'):
                current = {'time': None, 'ncells': None, 'cells': []}
                timesteps.append(current)
                state = 'time'
                continue
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
                current['ncells'] = int(parts[0]); state = None
            elif state == 'cells':
                if len(parts) >= 6:
                    current['cells'].append([float(x) for x in parts])
    last = timesteps[-1]
    arr = np.array(last['cells'])
    # cols: id, xc, yc, nD2, nD, nDp
    return last['time'], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]


def parse_surf(path):
    """Return (points, lines) from a SPARTA 2D surf file."""
    pts = []
    lines = []
    with open(path) as f:
        mode = None
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if 'Points' in s:
                mode = 'pts'; continue
            if 'Lines' in s:
                mode = 'lines'; continue
            parts = s.split()
            if mode == 'pts' and len(parts) >= 3:
                try:
                    pts.append((float(parts[1]), float(parts[2])))
                except ValueError:
                    pass
            elif mode == 'lines' and len(parts) >= 3:
                try:
                    lines.append((int(parts[1]), int(parts[2])))
                except ValueError:
                    pass
    return pts, lines


def overlay_surf(ax, path, color='k', lw=1.2):
    if not os.path.exists(path):
        return
    pts, ls = parse_surf(path)
    if not pts or not ls:
        return
    for a, b in ls:
        # SPARTA uses 1-based indexing
        if 0 < a <= len(pts) and 0 < b <= len(pts):
            x0, y0 = pts[a - 1]
            x1, y1 = pts[b - 1]
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=0.85)


t, x, y, nD2, nD, nDp = parse_last_snapshot(DUMP)
print(f'Final timestep: {t}  ncells: {len(x)}')
print(f'D2  range: {nD2.min():.2e}  {nD2.max():.2e}')
print(f'D   range: {nD.min():.2e}   {nD.max():.2e}')
print(f'D+  range: {nDp.min():.2e}  {nDp.max():.2e}')

# Bin onto a regular grid for imshow
nx, ny = 100, 100
xg = np.linspace(x.min(), x.max(), nx + 1)
yg = np.linspace(y.min(), y.max(), ny + 1)

def bin2d(vals):
    """Bin averaged cell-center values onto a nx x ny grid."""
    H, _, _ = np.histogram2d(x, y, bins=[xg, yg], weights=vals)
    C, _, _ = np.histogram2d(x, y, bins=[xg, yg])
    out = np.where(C > 0, H / np.maximum(C, 1), np.nan)
    return out.T  # (ny, nx) for imshow

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

def plot_species(ax, grid, title, cmap='viridis'):
    finite = grid[np.isfinite(grid) & (grid > 0)]
    vmin = max(finite.min(), finite.max() * 1e-4) if finite.size else 1e10
    vmax = finite.max() if finite.size else 1e15
    im = ax.imshow(grid, origin='lower',
                   extent=[xg[0], xg[-1], yg[0], yg[-1]],
                   aspect='equal', cmap=cmap,
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    overlay_surf(ax, WALL, color='white', lw=1.0)
    overlay_surf(ax, CORE, color='cyan',  lw=1.0)
    ax.set_xlabel('R [m]')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r'density  [m$^{-3}$]')

plot_species(axes[0], bin2d(nD2), r'$D_2$ density',    cmap='Blues')
plot_species(axes[1], bin2d(nD),  r'$D$ density',       cmap='Reds')
plot_species(axes[2], bin2d(nDp), r'$D^+$ density',     cmap='Greens')
axes[0].set_ylabel('Z [m]')

fig.suptitle(f'test_reclycling: D2/D/D+ neutral transport with wall recycling\n'
             f'WEST-like axi geometry, Boris pusher, 4-reaction chem/adas '
             f'(t = {t*1e-7*1e3:.2f} ms)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f'Saved: {OUT}')
