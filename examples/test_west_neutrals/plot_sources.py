#!/usr/bin/env python3
"""Plot per-cell source terms (ionization, recombination, CX, dissociation)
from test_west_neutrals at the final dump timestep."""

import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DUMP = 'output/grid.west'
WALL = 'input/wall.surf'
CORE = 'input/core.surf'
OUT  = 'output/west_source_terms.png'


def parse_last_snapshot(path):
    """Columns in dump: id xc yc f_fden[1..3] f_fchem[1..4]  (9 columns)"""
    timesteps = []
    current = None
    with open(path) as f:
        for line in f:
            if line.startswith('ITEM: TIMESTEP'):
                current = {'time': None, 'cells': []}
                timesteps.append(current)
                state = 'time'; continue
            if line.startswith('ITEM: NUMBER OF CELLS'):
                state = 'ncells'; continue
            if line.startswith('ITEM: BOX BOUNDS'):
                state = 'bounds'; continue
            if line.startswith('ITEM: CELLS'):
                state = 'cells'; continue
            parts = line.split()
            if not parts: continue
            if state == 'time':
                current['time'] = int(parts[0]); state = None
            elif state == 'ncells':
                state = None
            elif state == 'cells' and len(parts) >= 9:
                try:
                    current['cells'].append([float(x) for x in parts])
                except ValueError:
                    pass
    last = timesteps[-1]
    arr = np.array(last['cells'])
    return last['time'], arr


def parse_surf(path):
    pts = []; lines = []; mode = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'): continue
            if 'Points' in s: mode = 'pts'; continue
            if 'Lines'  in s: mode = 'lines'; continue
            parts = s.split()
            if mode == 'pts' and len(parts) >= 3:
                try: pts.append((float(parts[1]), float(parts[2])))
                except ValueError: pass
            elif mode == 'lines' and len(parts) >= 3:
                try: lines.append((int(parts[1]), int(parts[2])))
                except ValueError: pass
    return pts, lines


def overlay_surf(ax, path, color='k', lw=1.2):
    if not os.path.exists(path): return
    pts, ls = parse_surf(path)
    for a, b in ls:
        if 0 < a <= len(pts) and 0 < b <= len(pts):
            x0, y0 = pts[a - 1]; x1, y1 = pts[b - 1]
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=0.85)


t, arr = parse_last_snapshot(DUMP)
print(f'Final dump step {t}, cells = {len(arr)}')

x = arr[:, 1]
y = arr[:, 2]
# f_fden: D2=3, D=4, D+=5  (3-column species density)
# f_fchem: ioniz=6, recomb=7, CX=8, diss=9
nIon   = arr[:, 6]
nRec   = arr[:, 7]
nCx    = arr[:, 8]
nDiss  = arr[:, 9]
print(f'Totals - Ion:{nIon.sum():.0f}  Rec:{nRec.sum():.0f}  CX:{nCx.sum():.0f}  Diss:{nDiss.sum():.0f}')

nx, ny = 120, 160
xg = np.linspace(x.min(), x.max(), nx + 1)
yg = np.linspace(y.min(), y.max(), ny + 1)

def bin2d(vals):
    H, _, _ = np.histogram2d(x, y, bins=[xg, yg], weights=vals)
    return H.T

def plot_panel(ax, grid, title, cmap):
    finite = grid[grid > 0]
    if finite.size == 0:
        ax.text(0.5, 0.5, 'no events', transform=ax.transAxes,
                ha='center', va='center', color='gray')
        vmin, vmax = 1, 10
    else:
        vmin = max(1.0, finite.max() * 1e-3)
        vmax = finite.max()
    grid_plot = np.where(grid > 0, grid, np.nan)
    im = ax.imshow(grid_plot, origin='lower',
                   extent=[xg[0], xg[-1], yg[0], yg[-1]],
                   aspect='equal', cmap=cmap,
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    overlay_surf(ax, WALL, color='k',    lw=0.9)
    overlay_surf(ax, CORE, color='cyan', lw=0.8)
    ax.set_xlabel('R [m]')
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='events per cell')

fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
plot_panel(axes[0], bin2d(nIon),  r'Ionization: $D \rightarrow D^+$',     'Reds')
plot_panel(axes[1], bin2d(nRec),  r'Recombination: $D^+ \rightarrow D$',  'Purples')
plot_panel(axes[2], bin2d(nCx),   r'Charge exchange: $D^+ + D$',          'Blues')
plot_panel(axes[3], bin2d(nDiss), r'Dissociation: $D_2 \rightarrow 2D$',  'Greens')
axes[0].set_ylabel('Z [m]')

fig.suptitle(f'OpenEdge per-cell source tallies on WEST  (t = {t*1e-7*1e3:.2f} ms)\n'
             f'Cumulative events per cell -- Gkeyll-ready output from new '
             f'fix chem/adas source-tally channel',
             fontsize=12)
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches='tight')
print(f'Saved {OUT}')
