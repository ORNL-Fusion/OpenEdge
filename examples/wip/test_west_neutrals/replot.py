#!/usr/bin/env python3
"""Remake density / source plots from output/snapshots.npz
(no simulation re-run needed).

Usage:
  python3 replot.py                # uses last snapshot
  python3 replot.py --step INDEX   # which snapshot index to plot
  python3 replot.py --log          # keep log-scale colorbars (default)
  python3 replot.py --linear       # linear colorbars

Edit the style / colormaps / limits below to taste.
"""

import argparse, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

ap = argparse.ArgumentParser()
ap.add_argument('--step', type=int, default=-1,
                help='snapshot index (default = -1, last)')
ap.add_argument('--linear', action='store_true', help='linear colormap')
ap.add_argument('--npz', default='output/snapshots.npz')
ap.add_argument('--wall', default='input/wall.surf')
ap.add_argument('--core', default='input/core.surf')
args = ap.parse_args()

d = np.load(args.npz)
k = args.step
print(f'Snapshot step {d["steps"][k]}  t = {d["times_ms"][k]:.2f} ms')

x, y = d['x'], d['y']
xg, yg = d['xg'], d['yg']

def bin2d(vals):
    H, _, _ = np.histogram2d(x, y, bins=[xg, yg], weights=vals)
    C, _, _ = np.histogram2d(x, y, bins=[xg, yg])
    out = np.where(C > 0, H / np.maximum(C, 1), np.nan)
    return out.T

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

def panel(ax, grid, title, cmap, log=True):
    finite = grid[np.isfinite(grid) & (grid > 0)]
    if finite.size == 0:
        vmin, vmax = (1.0, 10.0) if log else (0, 1)
    else:
        vmin = max(finite.max() * 1e-3, finite.min()) if log else 0
        vmax = finite.max()
    if log:
        norm = LogNorm(vmin=max(vmin, 1e-30), vmax=max(vmax, vmin*10))
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(grid, origin='lower',
                   extent=[xg[0], xg[-1], yg[0], yg[-1]],
                   aspect='equal', cmap=cmap, norm=norm)
    overlay(ax, args.wall, color='k',    lw=0.8)
    overlay(ax, args.core, color='cyan', lw=0.7)
    ax.set_xlabel('R [m]')
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

log = not args.linear
# ---- densities ----
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
panel(axes[0], bin2d(d['nD2'][k]), r'$D_2$ density [m$^{-3}$]',  'Blues',  log)
panel(axes[1], bin2d(d['nD'][k]),  r'$D$ density [m$^{-3}$]',    'Reds',   log)
panel(axes[2], bin2d(d['nDp'][k]), r'$D^+$ density [m$^{-3}$]',  'Greens', log)
axes[0].set_ylabel('Z [m]')
fig.suptitle(f'WEST neutrals densities  (t = {d["times_ms"][k]:.2f} ms)')
plt.tight_layout()
out1 = 'output/replot_densities.png'
plt.savefig(out1, dpi=140, bbox_inches='tight'); print(f'Saved {out1}')
plt.close()

# ---- source terms ----
fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
panel(axes[0], bin2d(d['src_ion'][k]), 'Ionization',    'Reds',    log)
panel(axes[1], bin2d(d['src_rec'][k]), 'Recombination', 'Purples', log)
panel(axes[2], bin2d(d['src_cx'][k]),  'Charge Ex.',    'Blues',   log)
panel(axes[3], bin2d(d['src_dis'][k]), 'Dissociation',  'Greens',  log)
axes[0].set_ylabel('Z [m]')
fig.suptitle(f'WEST source-term tallies (cumulative)  (t = {d["times_ms"][k]:.2f} ms)')
plt.tight_layout()
out2 = 'output/replot_sources.png'
plt.savefig(out2, dpi=140, bbox_inches='tight'); print(f'Saved {out2}')
