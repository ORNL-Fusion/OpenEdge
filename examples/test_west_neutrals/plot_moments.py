#!/usr/bin/env python3
"""Plot per-species moments (density, bulk speed, temperature) on WEST.

3 rows x 3 columns (one column per species D2 / D / D+).
  Row 1: density n [m^-3]      (log scale)
  Row 2: bulk speed |u| [m/s]  (linear)
  Row 3: temperature T [K]     (linear, masked where n == 0)

These are exactly the moments of f_n that Gkeyll's C^iz / C^cx operators
consume (together with ADAS/Janev rate coefficients applied on the Gkeyll
side). Produced by SPARTA's compute grid ... species n u v w temp --
no OpenEdge-specific code needed.
"""

import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

NPZ  = 'output/snapshots.npz'
WALL = 'input/wall.surf'
CORE = 'input/core.surf'
OUT  = 'output/west_neutral_moments.png'

d = np.load(NPZ)
k = -1  # last snapshot
x, y = d['x'], d['y']
xg, yg = d['xg'], d['yg']


def bin2d_mean(vals, weights=None):
    """Average cell-center samples into a regular grid. If weights is given,
    compute a weighted mean (vals assumed to be already cell-averaged)."""
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


def overlay(ax, path, color='k', lw=0.8):
    pts, ls = parse_surf(path)
    for a, b in ls:
        if 0 < a <= len(pts) and 0 < b <= len(pts):
            x0, y0 = pts[a - 1]; x1, y1 = pts[b - 1]
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=0.9)


def panel(ax, grid, title, cmap, norm, unit):
    im = ax.imshow(grid, origin='lower',
                   extent=[xg[0], xg[-1], yg[0], yg[-1]],
                   aspect='equal', cmap=cmap, norm=norm)
    overlay(ax, WALL, color='k', lw=0.7)
    overlay(ax, CORE, color='cyan', lw=0.6)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=unit)


species = ['D$_2$', 'D', 'D$^+$']
n_arrs = [d['n_D2'][k],  d['n_D'][k],  d['n_Dp'][k]]
u_arrs = [d['u_D2'][k],  d['u_D'][k],  d['u_Dp'][k]]
v_arrs = [d['v_D2'][k],  d['v_D'][k],  d['v_Dp'][k]]
w_arrs = [d['w_D2'][k],  d['w_D'][k],  d['w_Dp'][k]]
T_arrs = [d['T_D2'][k],  d['T_D'][k],  d['T_Dp'][k]]

fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.5), sharey=True, sharex=True)
cmaps_n = ['Blues', 'Reds', 'Greens']

for j, sp in enumerate(species):
    n_raw = n_arrs[j]
    nbin = bin2d_mean(n_raw)
    # Bulk speed magnitude
    speed = np.sqrt(u_arrs[j]**2 + v_arrs[j]**2 + w_arrs[j]**2)
    # Mask speed and T where n=0 (else 'T' is a meaningless stat from 0 samples)
    mask = n_raw > 0
    speed_masked = np.where(mask, speed, np.nan)
    T_masked     = np.where(mask, T_arrs[j], np.nan)
    sbin = bin2d_mean(speed_masked)
    Tbin = bin2d_mean(T_masked)

    # density (log)
    finite = nbin[np.isfinite(nbin) & (nbin > 0)]
    if finite.size:
        nmax = finite.max(); nmin = max(finite.min(), nmax * 1e-4)
        norm_n = LogNorm(vmin=nmin, vmax=nmax)
    else:
        norm_n = LogNorm(vmin=1, vmax=10)
    panel(axes[0, j], nbin, f'n({sp}) [m$^{{-3}}$]', cmaps_n[j], norm_n, '')

    # bulk speed (linear)
    if np.isfinite(sbin).any():
        smax = np.nanmax(sbin)
        norm_s = Normalize(vmin=0, vmax=max(smax, 1.0))
    else:
        norm_s = Normalize(vmin=0, vmax=1)
    panel(axes[1, j], sbin, f'|u|({sp}) [m/s]', 'magma', norm_s, '')

    # temperature (linear, but clamp to reasonable bounds)
    finiteT = Tbin[np.isfinite(Tbin) & (Tbin > 0)]
    if finiteT.size:
        # clip to 1st..99th percentile to keep color scale informative
        lo = np.percentile(finiteT, 2)
        hi = np.percentile(finiteT, 98)
        norm_T = Normalize(vmin=lo, vmax=hi)
    else:
        norm_T = Normalize(vmin=0, vmax=1)
    panel(axes[2, j], Tbin, f'T({sp}) [K]', 'plasma', norm_T, '')

for ax in axes[:, 0]:
    ax.set_ylabel('Z [m]')
for ax in axes[-1, :]:
    ax.set_xlabel('R [m]')

fig.suptitle(f'OpenEdge per-species moments on WEST  (t = {d["times_ms"][k]:.2f} ms)\n'
             f'Rows: density / bulk speed / temperature. '
             f'Columns: D$_2$ / D / D$^+$.\n'
             f'Exact inputs that Gkeyll\'s C$^{{iz}}$ / C$^{{cx}}$ collision operators consume.',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches='tight')
print(f'Saved {OUT}')
