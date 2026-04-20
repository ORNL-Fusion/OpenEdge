#!/usr/bin/env python3
"""Compare convergence scan runs A / B / C.

Reads:
  log.{A,B,C}.openedge              stats time-series (np, c_csrc, c_cwmean, c_cwmax)
  output/grid.dens.{A,B,C}.west     per-cell n(W0..W20+) every Ndump steps

Produces:
  output/convergence_timeseries.png    n_peak(t), n_mean(t), np(t) for each run
  output/convergence_final_maps.png    final-snapshot R-Z density maps side by side

The time-series panel answers "how long until density plateaus?". The
side-by-side density maps answer "does more sampling change the peak?".
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent.parent
DT   = 5.0e-9
SLAB_Z = 0.10

# Tag -> (nlaunch_total, color, label)
RUNS = {
    'A': (10,  'C0', 'A: nlaunch=10'),
    'B': (100, 'C1', 'B: nlaunch=100'),
    'C': (300, 'C2', 'C: nlaunch=300'),
}


def parse_stats(path: Path):
    """Return {'Step': arr, 'Np': arr, 'c_csrc': arr, 'c_cwmean': arr, 'c_cwmax': arr}."""
    rows = []
    header = None
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if s.startswith('Step '):
            header = s.split(); rows = []; continue
        if header is None:
            continue
        if s.startswith('Loop time') or not s:
            header = None; continue
        parts = s.split()
        if len(parts) == len(header):
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                header = None
    if header is None or not rows:
        return None
    arr = np.array(rows)
    return {name: arr[:, i] for i, name in enumerate(header)}


def parse_all_snapshots(path: Path):
    """Return list of (t, header, rows-as-np-array) for every snapshot."""
    snaps = []
    cur = None
    state = None
    for ln in path.read_text().splitlines():
        if ln.startswith('ITEM: TIMESTEP'):
            cur = {'t': None, 'hdr': None, 'rows': []}
            snaps.append(cur); state = 't'; continue
        if ln.startswith('ITEM: NUMBER OF CELLS'):
            state = 'n'; continue
        if ln.startswith('ITEM: BOX BOUNDS'):
            state = 'b'; continue
        if ln.startswith('ITEM: CELLS'):
            cur['hdr'] = ln.split()[2:]; state = 'r'; continue
        parts = ln.split()
        if not parts:
            continue
        if state == 't':
            cur['t'] = int(parts[0]); state = None
        elif state == 'n':
            state = None
        elif state == 'r':
            try: cur['rows'].append([float(x) for x in parts])
            except ValueError: pass
    return [(s['t'], s['hdr'], np.array(s['rows'])) for s in snaps if s['rows']]


def parse_wall_polygon(path: Path):
    pts, segs = {}, []
    mode = None
    for ln in path.read_text().splitlines():
        s = ln.strip().lower()
        if not s or s.startswith('#'): continue
        if s.endswith('points'): mode = 'pts'; continue
        if s.endswith('lines'):  mode = 'lines'; continue
        parts = ln.split()
        if mode == 'pts' and len(parts) >= 3:
            try: pts[int(parts[0])] = (float(parts[1]), float(parts[2]))
            except ValueError: pass
        elif mode == 'lines' and len(parts) >= 3:
            try: segs.append((int(parts[1]), int(parts[2])))
            except ValueError: pass
    if not segs: return np.empty((0,2))
    nxt = {a: b for a, b in segs}
    start = segs[0][0]; poly = [start]; cur = nxt.get(start)
    while cur is not None and cur != start:
        poly.append(cur); cur = nxt.get(cur)
    return np.array([pts[i] for i in poly if i in pts])


# ---------------- time series panel --------------------------------------

def plot_timeseries():
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for tag, (nlaunch, color, label) in RUNS.items():
        logpath = HERE / f'log.{tag}.openedge'
        if not logpath.exists(): continue
        stats = parse_stats(logpath)
        if stats is None: continue
        t_ms = stats['Step'] * DT * 1e3
        axes[0].plot(t_ms, stats['Np'],         color=color, lw=1.1, label=label)
        axes[1].plot(t_ms, stats['c_cwmean'],   color=color, lw=1.1, label=label)
        axes[2].plot(t_ms, stats['c_cwmax'],    color=color, lw=1.1, label=label)

    axes[0].set_ylabel('Np (macros)'); axes[0].set_yscale('log')
    axes[1].set_ylabel(r'$\langle n(W_0)\rangle$ [m$^{-3}$]'); axes[1].set_yscale('log')
    axes[2].set_ylabel(r'peak $n(W_0)$ [m$^{-3}$]'); axes[2].set_yscale('log')
    axes[2].set_xlabel('time [ms]')
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(loc='lower right', fontsize=9)
    fig.suptitle('WEST convergence scan: np / mean / peak vs. physical time')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = HERE / 'output' / 'convergence_timeseries.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'saved {out}')


# ---------------- final density maps panel -------------------------------

def axi_factor(xc):
    R_avg = 0.5 * (xc.min() + xc.max())
    return 2.0 * np.pi * R_avg / SLAB_Z


def plot_final_maps():
    wall = parse_wall_polygon(HERE / 'input' / 'wall.surf')
    nx, ny = 180, 260

    fig, axes = plt.subplots(1, len(RUNS), figsize=(5 * len(RUNS), 7),
                              sharey=True, constrained_layout=True)
    for j, (tag, (nlaunch, color, label)) in enumerate(RUNS.items()):
        path = HERE / 'output' / f'grid.dens.{tag}.west'
        if not path.exists():
            axes[j].text(0.5, 0.5, 'missing', transform=axes[j].transAxes,
                          ha='center')
            continue
        snaps = parse_all_snapshots(path)
        if not snaps:
            continue
        t_final, hdr, rows = snaps[-1]
        xc = rows[:, 1]; yc = rows[:, 2]; vol = rows[:, 3]
        n_all = rows[:, 4:].sum(axis=1)
        af = axi_factor(xc)
        n_all_axi = n_all * af

        xg = np.linspace(xc.min(), xc.max(), nx+1)
        yg = np.linspace(yc.min(), yc.max(), ny+1)
        H_sum, _, _ = np.histogram2d(yc, xc, bins=[yg, xg], weights=n_all_axi)
        H_cnt, _, _ = np.histogram2d(yc, xc, bins=[yg, xg])
        grid = np.divide(H_sum, H_cnt, out=np.zeros_like(H_sum), where=H_cnt>0)
        grid = np.ma.masked_where(grid <= 0, grid)

        if wall.size:
            X, Y = np.meshgrid(0.5*(xg[1:]+xg[:-1]), 0.5*(yg[1:]+yg[:-1]))
            inside = MplPath(wall).contains_points(
                np.column_stack([X.ravel(), Y.ravel()])).reshape(grid.shape)
            grid = np.ma.masked_where(~inside, grid)

        ax = axes[j]
        im = ax.pcolormesh(xg, yg, np.ma.log10(grid), shading='flat',
                           cmap='plasma', vmin=9, vmax=13)
        if wall.size:
            ax.plot(wall[:, 0], wall[:, 1], 'k', lw=1.5)
        ax.set_aspect('equal')
        ax.set_xlabel('R [m]')
        if j == 0: ax.set_ylabel('Z [m]')
        # stats in title
        peak = n_all_axi.max()
        mean = n_all_axi[n_all_axi>0].mean() if (n_all_axi>0).any() else 0
        ax.set_title(f'{label}\nt={t_final*DT*1e3:.2f} ms   '
                     f'peak={peak:.2e}   mean={mean:.2e}')

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cb.set_label(r'$\log_{10}\,n(W_{\rm all})$ axi-corrected [m$^{-3}$]')
    fig.suptitle('Final-snapshot density maps   (axisymmetric-corrected)')
    out = HERE / 'output' / 'convergence_final_maps.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'saved {out}')


# ---------------- summary table -------------------------------------------

def print_summary():
    print()
    print('=== convergence scan summary ===')
    print(f'{"run":>4} {"nlaunch":>8} {"t_final [ms]":>13} {"np":>10}'
          f' {"n_peak [m-3]":>14} {"n_mean [m-3]":>14}')
    for tag, (nlaunch, _, _) in RUNS.items():
        path = HERE / 'output' / f'grid.dens.{tag}.west'
        if not path.exists():
            print(f'{tag:>4} {nlaunch:>8}    -- missing --')
            continue
        snaps = parse_all_snapshots(path)
        if not snaps: continue
        t_final, hdr, rows = snaps[-1]
        n_all = rows[:, 4:].sum(axis=1)
        af = axi_factor(rows[:, 1])
        n_all *= af
        logpath = HERE / f'log.{tag}.openedge'
        stats = parse_stats(logpath)
        np_final = stats['Np'][-1] if stats else float('nan')
        print(f'{tag:>4} {nlaunch:>8} {t_final*DT*1e3:>13.3f} {np_final:>10.0f}'
              f' {n_all.max():>14.3e}'
              f' {n_all[n_all>0].mean():>14.3e}')


if __name__ == '__main__':
    plot_timeseries()
    plot_final_maps()
    print_summary()
