#!/usr/bin/env python3
"""Plot per-cell source terms (counts + energy deposition) on WEST geometry.

2 rows x 4 columns:
  Row 1: event counts (ionization / recomb / CX / dissociation)
  Row 2: energy deposited per cell [J] from the same reactions

Energy is the sum of 0.5 * m * |v|^2 over all reactant particles that
fired that reaction inside the cell. Divide by a coupling window dt to
get volumetric power density [W/m^3] -- the input Gkeyll's C^rad or a
plasma energy-balance coupling needs.
"""

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
            if not parts: continue
            if state == 'time':
                current['time'] = int(parts[0]); state = None
            elif state == 'ncells':
                state = None
            elif state == 'cells' and len(parts) >= 26:
                try:
                    current['cells'].append([float(x) for x in parts])
                except ValueError:
                    pass
    last = timesteps[-1]
    return last['time'], np.array(last['cells'])


def parse_surf(path):
    pts = []; lines = []; mode = None
    if not os.path.exists(path): return pts, lines
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


def overlay(ax, path, color='k', lw=1.0):
    pts, ls = parse_surf(path)
    for a, b in ls:
        if 0 < a <= len(pts) and 0 < b <= len(pts):
            x0, y0 = pts[a - 1]; x1, y1 = pts[b - 1]
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=0.85)


t, arr = parse_last_snapshot(DUMP)
print(f'Final dump step {t}, cells = {len(arr)}')

x = arr[:, 1]; y = arr[:, 2]

# Column layout per save_snapshots.py
counts = arr[:, 6:10]    # ion, rec, cx, dis
energy = arr[:, 22:26]   # ion, rec, cx, dis
print(f'Total counts - Ion:{counts[:,0].sum():.0f}  Rec:{counts[:,1].sum():.0f}  '
      f'CX:{counts[:,2].sum():.0f}  Diss:{counts[:,3].sum():.0f}')
print(f'Total energy [J] - Ion:{energy[:,0].sum():.3e}  CX:{energy[:,2].sum():.3e}  '
      f'Diss:{energy[:,3].sum():.3e}')

nx, ny = 120, 160
xg = np.linspace(x.min(), x.max(), nx + 1)
yg = np.linspace(y.min(), y.max(), ny + 1)

def bin2d(vals):
    H, _, _ = np.histogram2d(x, y, bins=[xg, yg], weights=vals)
    return H.T

def panel(ax, grid, title, cmap, unit):
    finite = grid[grid > 0]
    if finite.size == 0:
        ax.text(0.5, 0.5, 'no events', transform=ax.transAxes,
                ha='center', va='center', color='gray')
        vmin, vmax = 1, 10
    else:
        vmin = max(finite.max() * 1e-3, np.finfo(float).tiny)
        vmax = finite.max()
    grid_plot = np.where(grid > 0, grid, np.nan)
    im = ax.imshow(grid_plot, origin='lower',
                   extent=[xg[0], xg[-1], yg[0], yg[-1]],
                   aspect='equal', cmap=cmap,
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    overlay(ax, WALL, color='k',    lw=0.8)
    overlay(ax, CORE, color='cyan', lw=0.7)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=unit)

titles = ['Ionization', 'Recombination', 'Charge exchange', 'Dissociation']
cmaps_cnt = ['Reds', 'Purples', 'Blues', 'Greens']
cmaps_E   = ['OrRd', 'BuPu',    'PuBu',  'YlGn']

fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharey=True)
for j in range(4):
    panel(axes[0, j], bin2d(counts[:, j]),
          f'Count -- {titles[j]}', cmaps_cnt[j], 'events/cell')
    panel(axes[1, j], bin2d(energy[:, j]),
          f'Energy source -- {titles[j]}', cmaps_E[j], 'J/cell (cumulative)')
for ax in axes.flat:
    ax.set_xlabel('R [m]')
for ax in axes[:, 0]:
    ax.set_ylabel('Z [m]')

fig.suptitle(f'OpenEdge per-cell source tallies on WEST  (t = {t*1e-7*1e3:.2f} ms)\n'
             f'Top row: particle source count.  Bottom row: kinetic energy deposited.\n'
             f'Gkeyll handoff: divide by coupling window dt -> source rate / power density.',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches='tight')
print(f'Saved {OUT}')
