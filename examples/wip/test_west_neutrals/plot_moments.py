#!/usr/bin/env python3
"""Plot per-species moments (n, u_R, u_Z, u_phi, T in eV) on WEST using the
smoothed binning / wall-masking approach. Each velocity component shown
separately with a diverging colormap (sign matters -- u/v/w are signed
Cartesian components, not magnitudes).

Layout per figure:  species in columns,  {n, u_R, u_Z, T_eV} in rows.
By default plots D2, D, D+ (3 columns, 4 rows = 12 panels).
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.colors import LogNorm, TwoSlopeNorm

# Reuse helpers from the smoother reference script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_grid_density_west import (
    parse_surface, parse_grid_dump, make_binned_grid, smooth_masked_grid,
    mask_outside_wall, centers_to_edges, default_display_bins,
)

K_PER_EV = 11604.505  # K per 1 eV

def plot_moment_fixed(ax, x, y, grid, valid, wall_poly, title, cmap, mode,
                      vmin, vmax, puff_point=None):
    """Version of plot_moment with externally specified vmin/vmax (for shared
    row-wise color scales)."""
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    masked = mask_outside_wall(x, y, grid, wall_poly, valid)
    norm = None
    if mode == 'log':
        masked = np.ma.masked_less_equal(masked, 0.0)
        norm = LogNorm(vmin=max(vmin, 1e-30), vmax=max(vmax, vmin * 10))
    elif mode == 'divergent':
        amp = max(abs(vmin), abs(vmax), 1e-30)
        norm = TwoSlopeNorm(vmin=-amp, vcenter=0.0, vmax=amp)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    r_edges, z_edges = np.meshgrid(centers_to_edges(x), centers_to_edges(y))
    m = ax.pcolormesh(r_edges, z_edges, masked, shading='flat',
                      cmap=cmap, norm=norm)
    if wall_poly.size:
        ax.plot(wall_poly[:, 0], wall_poly[:, 1], 'k', lw=1.0)
    if puff_point is not None:
        ax.plot(puff_point[0], puff_point[1], marker='*', markersize=14,
                color='yellow', markeredgecolor='k', markeredgewidth=0.8)
    plt.colorbar(m, ax=ax, pad=0.01, fraction=0.04, shrink=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('R [m]'); ax.set_ylabel('Z [m]')
    ax.grid(alpha=0.2, linestyle='--')


def plot_moment(ax, x, y, grid, valid, wall_poly, title, cmap,
                norm_mode='log', puff_point=None, sym=False,
                log_vmax_quantile=99.5, log_span=4.0):
    """norm_mode in {'log', 'linear', 'divergent'}. Temperature uses 'linear',
    signed velocities use 'divergent', density uses 'log'."""
    masked = mask_outside_wall(x, y, grid, wall_poly, valid)
    vmin = vmax = None

    if norm_mode == 'log':
        masked = np.ma.masked_less_equal(masked, 0.0)
        masked = np.ma.log10(masked)
        vals = np.asarray(masked.compressed(), dtype=float)
        if vals.size:
            vmax = (float(vals.max()) if log_vmax_quantile >= 100
                    else float(np.percentile(vals, log_vmax_quantile)))
            vmin = float(vals.min())
            if log_span > 0: vmin = max(vmin, vmax - log_span)
        norm = None
    elif norm_mode == 'divergent':
        vals = np.asarray(masked.compressed(), dtype=float)
        if vals.size:
            amp = max(abs(np.percentile(vals, 2)),
                      abs(np.percentile(vals, 98)))
            if amp == 0: amp = 1.0
            norm = TwoSlopeNorm(vmin=-amp, vcenter=0.0, vmax=amp)
        else:
            norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    else:  # 'linear'
        vals = np.asarray(masked.compressed(), dtype=float)
        if vals.size:
            vmin = float(np.percentile(vals, 2))
            vmax = float(np.percentile(vals, 98))
        norm = None

    r_edges, z_edges = np.meshgrid(centers_to_edges(x), centers_to_edges(y))
    m = ax.pcolormesh(r_edges, z_edges, masked, shading='flat',
                      cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    # Wall polygon outline
    if wall_poly.size:
        ax.plot(wall_poly[:, 0], wall_poly[:, 1], 'k', lw=1.0)
    if puff_point is not None:
        ax.plot(puff_point[0], puff_point[1], marker='*', markersize=14,
                color='yellow', markeredgecolor='k', markeredgewidth=0.8)
    plt.colorbar(m, ax=ax, pad=0.01, fraction=0.04, shrink=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('R [m]'); ax.set_ylabel('Z [m]')
    ax.grid(alpha=0.2, linestyle='--')


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default=str(here / 'output' / 'grid.west'))
    ap.add_argument('--wall', default=str(here / 'input' / 'wall.surf'))
    ap.add_argument('--timestep', default='last')
    ap.add_argument('--species', nargs='+', default=['D2', 'D', 'D+'],
                    help='species labels (same order as in dump f_fmom grouping)')
    ap.add_argument('--target-bins', type=int, default=160)
    ap.add_argument('--smooth', type=float, default=0.8)
    ap.add_argument('--puff', nargs=2, type=float, default=[2.1887, -0.6925])
    ap.add_argument('--out', default=str(here / 'output' / 'west_moments_smooth.png'))
    ap.add_argument('--title', default='WEST per-species moments of f_n  (smoothed)')
    args = ap.parse_args()

    blocks = parse_grid_dump(Path(args.dump))
    ts = max(blocks.keys()) if args.timestep == 'last' else int(args.timestep)
    header, cols = blocks[ts]
    hmap = {h: i for i, h in enumerate(header)}

    xc = cols[hmap['xc']]; yc = cols[hmap['yc']]

    # f_fmom layout (SPARTA compute grid ... species <quantities>,
    # 3 species, 5 quantities = 15 cols). Species-MAJOR ordering:
    #   [1]..[5]   = nrho, u, v, w, T  for D2
    #   [6]..[10]  = nrho, u, v, w, T  for D
    #  [11]..[15]  = nrho, u, v, w, T  for D+
    idx_n = [hmap[f'f_fmom[{1 + 5*j}]'] for j in range(3)]
    idx_u = [hmap[f'f_fmom[{2 + 5*j}]'] for j in range(3)]
    idx_v = [hmap[f'f_fmom[{3 + 5*j}]'] for j in range(3)]
    idx_w = [hmap[f'f_fmom[{4 + 5*j}]'] for j in range(3)]
    idx_T = [hmap[f'f_fmom[{5 + 5*j}]'] for j in range(3)]

    wall_r, wall_z, wall_poly = parse_surface(Path(args.wall))

    xmin = float(xc.min()); xmax = float(xc.max())
    ymin = float(yc.min()); ymax = float(yc.max())
    if wall_poly.size:
        xmin = min(xmin, float(wall_poly[:, 0].min())); xmax = max(xmax, float(wall_poly[:, 0].max()))
        ymin = min(ymin, float(wall_poly[:, 1].min())); ymax = max(ymax, float(wall_poly[:, 1].max()))

    nbx, nby = default_display_bins(xmin, xmax, ymin, ymax,
                                    args.target_bins, len(xc))

    def bin_smooth(vals, valid_pts=None):
        """Bin vals over cells; if valid_pts is given, use only those cells
        (avoids polluting moments with zeros from empty cells)."""
        if valid_pts is not None:
            use_xc = xc[valid_pts]; use_yc = yc[valid_pts]
            use_v  = vals[valid_pts]
        else:
            use_xc = xc; use_yc = yc; use_v = vals
        x, y, g, v = make_binned_grid(use_xc, use_yc, use_v, nbx, nby, 'mean',
                                      (xmin, xmax, ymin, ymax))
        g, v = smooth_masked_grid(g, v, args.smooth)
        return x, y, g, v

    sps = args.species
    ncol = len(sps); nrow = 5   # n, u, v, w, T
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.8 * nrow),
                             squeeze=False, constrained_layout=True)

    # First pass: bin all data so we can compute shared color limits per row.
    panel_data = []   # list of (x, y, gn, gu, gv, gw, gT, vn) per species
    for j, sp in enumerate(sps):
        n_raw = cols[idx_n[j]]
        u_raw = cols[idx_u[j]]
        v_raw = cols[idx_v[j]]
        w_raw = cols[idx_w[j]]
        T_raw = cols[idx_T[j]]
        nz = n_raw > 0
        x, y, gn, vn = bin_smooth(n_raw, valid_pts=nz)
        _, _, gu, _  = bin_smooth(u_raw, valid_pts=nz)
        _, _, gv, _  = bin_smooth(v_raw, valid_pts=nz)
        _, _, gw, _  = bin_smooth(w_raw, valid_pts=nz)
        _, _, gT, _  = bin_smooth(T_raw / K_PER_EV, valid_pts=nz)
        panel_data.append((x, y, gn, gu, gv, gw, gT, vn))

    def row_bounds(arrs, valids, mode):
        """Compute shared (vmin, vmax) using only cells inside the valid mask.
        mode: 'log', 'linear', or 'divergent'."""
        vals_all = []
        for a, v in zip(arrs, valids):
            m = np.asarray(a)
            ok = v & np.isfinite(m)
            vals_all.append(m[ok].ravel())
        vals = np.concatenate(vals_all) if vals_all else np.array([])
        if vals.size == 0:
            return (0.0, 1.0)
        if mode == 'log':
            positive = vals[vals > 0]
            if positive.size == 0: return (1e-30, 1.0)
            lo = np.percentile(positive, 2); hi = np.percentile(positive, 99.5)
            lo = max(lo, hi * 1e-4)
            return (lo, hi)
        if mode == 'divergent':
            # ignore zeros (empty cells); only use nonzero entries
            nonzero = vals[np.abs(vals) > 0]
            if nonzero.size == 0: return (-1.0, 1.0)
            amp = max(abs(np.percentile(nonzero, 2)),
                      abs(np.percentile(nonzero, 98)))
            return (-amp, amp) if amp > 0 else (-1.0, 1.0)
        # linear
        nonzero = vals[np.abs(vals) > 0]
        if nonzero.size == 0: return (0.0, 1.0)
        return (float(np.percentile(nonzero, 2)),
                float(np.percentile(nonzero, 98)))

    # Shared color limits per row (use density-based valid mask)
    valids = [pd[7] for pd in panel_data]
    n_vmin, n_vmax = row_bounds([pd[2] for pd in panel_data], valids, 'log')
    u_vmin, u_vmax = row_bounds([pd[3] for pd in panel_data], valids, 'divergent')
    v_vmin, v_vmax = row_bounds([pd[4] for pd in panel_data], valids, 'divergent')
    w_vmin, w_vmax = row_bounds([pd[5] for pd in panel_data], valids, 'divergent')
    T_vmin, T_vmax = row_bounds([pd[6] for pd in panel_data], valids, 'linear')
    print(f'Shared limits -- n: [{n_vmin:.2e}, {n_vmax:.2e}]')
    print(f'  u: [{u_vmin:.2f}, {u_vmax:.2f}] m/s')
    print(f'  v: [{v_vmin:.2f}, {v_vmax:.2f}] m/s')
    print(f'  w: [{w_vmin:.2f}, {w_vmax:.2f}] m/s')
    print(f'  T: [{T_vmin:.3f}, {T_vmax:.3f}] eV')

    # Render with fixed row-wise limits
    for j, sp in enumerate(sps):
        x, y, gn, gu, gv, gw, gT, vn = panel_data[j]
        plot_moment_fixed(axes[0, j], x, y, gn, vn, wall_poly,
                          f'n({sp}) [m$^{{-3}}$]', 'plasma', 'log',
                          n_vmin, n_vmax, args.puff)
        plot_moment_fixed(axes[1, j], x, y, gu, vn, wall_poly,
                          f'u({sp}) [m/s]  (x)', 'RdBu_r', 'divergent',
                          u_vmin, u_vmax, args.puff)
        plot_moment_fixed(axes[2, j], x, y, gv, vn, wall_poly,
                          f'v({sp}) [m/s]  (y)', 'RdBu_r', 'divergent',
                          v_vmin, v_vmax, args.puff)
        plot_moment_fixed(axes[3, j], x, y, gw, vn, wall_poly,
                          f'w({sp}) [m/s]  (z)', 'RdBu_r', 'divergent',
                          w_vmin, w_vmax, args.puff)
        plot_moment_fixed(axes[4, j], x, y, gT, vn, wall_poly,
                          f'T({sp}) [eV]', 'inferno', 'linear',
                          T_vmin, T_vmax, args.puff)

    fig.suptitle(f'{args.title}  (t = {ts*1e-7*1e3:.2f} ms)', fontsize=13)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches='tight')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
