#!/usr/bin/env python3
"""
A5c comparator: OpenEdge DIII-D neutrals vs SOLPS-EIRENE truth (lore2023_reference).

Loads:
  - OpenEdge grid dump (output/diii_d.grid) with f_frate[1..4] per cell
  - EIRENE truth (input/eirene_truth.h5) with S_iz, S_diss on SOLPS (R,Z) grid

Plots side-by-side ionization and dissociation source maps, plus the OpenEdge
D + D2 densities for context. All on shared log-scale colormaps.

NOTE on scope: OpenEdge was run puff-only with absorbing walls -- no wall
recycling. Expect the EIRENE map to be much broader (includes 7 recycling
strata); OpenEdge should show a localized plume near the puff location.
"""
import os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, '..', 'test_slab_stangeby2000'))
from compare_iz import read_grid_dump     # reuse the SPARTA-dump parser

def main():
    # ---- load OpenEdge dump (last snapshot). Default to recycle; fall back
    # to puff-only if recycle dump missing. Override via argv[1].
    if len(sys.argv) > 1:
        dump = sys.argv[1]
    else:
        cand = os.path.join(THIS, 'output', 'diii_d_recycle.grid')
        dump = cand if os.path.exists(cand) else os.path.join(THIS, 'output', 'diii_d.grid')
    print(f'reading {dump}')
    d = read_grid_dump(dump)
    # OpenEdge dumps xc/yc in SPARTA slot order. Detect axi vs cart by the
    # range of the y column: axi has y >= 0 (radial); cart 2D has y in [-1.4, 1.4].
    xc_raw = d['xc']; yc_raw = d['yc']
    axi = (yc_raw.min() >= -1e-6)
    if axi:
        # SPARTA axi: x = Z (axial), y = R (radial). Plot in physical (R, Z).
        Z_oe = xc_raw
        R_oe = yc_raw
    else:
        R_oe = xc_raw
        Z_oe = yc_raw
    n_D2  = d['f_fden[1]']
    n_D   = d['f_fden[2]']
    S_iz_oe   = d['f_frate[1]']    # ionization rate [m^-3 s^-1]
    S_diss_oe = d['f_frate[4]']    # dissociation rate

    # ---- load EIRENE truth
    with h5py.File(os.path.join(THIS, 'input', 'eirene_truth.h5'), 'r') as f:
        R   = f['R'][:]
        Z   = f['Z'][:]
        S_iz_ei   = f['S_iz'][:]
        S_diss_ei = f['S_diss'][:]
        te = f['te_eV'][:]
        ne = f['ne'][:]

    # ---- summary metrics
    # In SPARTA axi mode the per-cell volume is the true cylindrical
    # volume that f_frate is normalized by — so summing rate * (true cell
    # volume) gives the full-3D total source. Here we approximate the cell
    # volume from the base-grid spacing and the radial position.
    def oe_cell_vol(R_arr):
        dx_axial = (1.4 - (-1.4)) / 200    # base grid 200 in x = Z
        dy_radial = (2.4 - 0.0) / 100       # base grid 100 in y = R
        if axi:
            # 2*pi*R*dR*dZ — full toroidal revolution
            return 2.0 * np.pi * np.maximum(R_arr, 1e-6) * dx_axial * dy_radial
        else:
            # 2D Cartesian: dx*dy*dz_box (per-radian-wedge convention)
            return dx_axial * dy_radial * 1.0
    vol_oe = oe_cell_vol(R_oe)

    tot_iz_oe   = float(np.sum(S_iz_oe)   * vol_oe)
    tot_diss_oe = float(np.sum(S_diss_oe) * vol_oe)
    # EIRENE: we stored per-volume rates, need to re-integrate with vol
    # (which was part of eirene_truth.h5? no -- we stored S as per-vol)
    # Just report domain-integrated via cell-vol proxy: assume balance.nc
    # tally was divided by vol when we wrote S_iz etc. -> sum over cells
    # equals volume-weighted, but we didn't store vol. Skip for now.

    print(f'OpenEdge:        peak S_iz = {S_iz_oe.max():.2e}  peak S_diss = {S_diss_oe.max():.2e}')
    print(f'SOLPS-EIRENE:    peak S_iz = {S_iz_ei.max():.2e}  peak S_diss = {S_diss_ei.max():.2e}')
    print(f'                 (SOLPS-EIRENE/OpenEdge peak ratio = {S_iz_ei.max()/max(S_iz_oe.max(),1):.1e})')
    print(f'Total OpenEdge reaction count (dump rate x vol x run time):')
    print(f'  iz   ~= {tot_iz_oe:.2e} /s  (per-step rate-mode, ~1 window snapshot)')
    print(f'  diss ~= {tot_diss_oe:.2e} /s')

    # ---- 2x3 plot: OE density, OE S_iz, EI S_iz, OE S_diss, EI S_diss, plasma ctx
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    vmin_iz = 1e20
    vmax_iz = max(S_iz_oe.max(), S_iz_ei.max()) * 1.1
    vmin_di = 1e19
    vmax_di = max(S_diss_oe.max(), S_diss_ei.max()) * 1.1

    # OE density panels — always plot in physical (R, Z)
    ax = axes[0,0]
    posmask = n_D2 > 0
    sc = ax.scatter(R_oe[posmask], Z_oe[posmask], c=np.maximum(n_D2[posmask], 1e10), s=1,
                    norm=LogNorm(vmin=max(1e10, n_D2[posmask].min()), vmax=n_D2.max()),
                    cmap='plasma')
    plt.colorbar(sc, ax=ax); ax.set_title('OpenEdge $n_{D_2}$ [m$^{-3}$]')

    ax = axes[0,1]
    posmask = S_iz_oe > 0
    if posmask.any():
        sc = ax.scatter(R_oe[posmask], Z_oe[posmask], c=np.maximum(S_iz_oe[posmask], vmin_iz),
                        s=1, norm=LogNorm(vmin=vmin_iz, vmax=vmax_iz), cmap='viridis')
        plt.colorbar(sc, ax=ax)
    ax.set_title('OpenEdge $S_{iz}$ [m$^{-3}$s$^{-1}$]')

    ax = axes[0,2]
    posmask = S_iz_ei > 0
    im = ax.pcolormesh(R, Z, np.maximum(S_iz_ei, vmin_iz),
                       norm=LogNorm(vmin=vmin_iz, vmax=vmax_iz), cmap='viridis',
                       shading='nearest')
    plt.colorbar(im, ax=ax); ax.set_title('SOLPS-EIRENE $S_{iz}$ [m$^{-3}$s$^{-1}$]')

    ax = axes[1,0]
    posmask = n_D > 0
    sc = ax.scatter(R_oe[posmask], Z_oe[posmask], c=np.maximum(n_D[posmask], 1e10), s=1,
                    norm=LogNorm(vmin=max(1e10, n_D[posmask].min()), vmax=n_D.max()),
                    cmap='plasma')
    plt.colorbar(sc, ax=ax); ax.set_title('OpenEdge $n_D$ [m$^{-3}$]')

    ax = axes[1,1]
    posmask = S_diss_oe > 0
    if posmask.any():
        sc = ax.scatter(R_oe[posmask], Z_oe[posmask], c=np.maximum(S_diss_oe[posmask], vmin_di),
                        s=1, norm=LogNorm(vmin=vmin_di, vmax=vmax_di), cmap='magma')
        plt.colorbar(sc, ax=ax)
    ax.set_title('OpenEdge $S_{diss}$ [m$^{-3}$s$^{-1}$]')

    ax = axes[1,2]
    im = ax.pcolormesh(R, Z, np.maximum(S_diss_ei, vmin_di),
                       norm=LogNorm(vmin=vmin_di, vmax=vmax_di), cmap='magma',
                       shading='nearest')
    plt.colorbar(im, ax=ax); ax.set_title('SOLPS-EIRENE $S_{diss}$ [m$^{-3}$s$^{-1}$]')

    for row in axes:
        for ax in row:
            ax.set_aspect('equal')
            ax.set_xlim(0.95, 2.4); ax.set_ylim(-1.4, 1.4)
    for ax in axes[1]:
        ax.set_xlabel('R [m]')
    for ax in axes[:,0]:
        ax.set_ylabel('Z [m]')

    if 'eirene' in os.path.basename(dump):
        title_src = 'wall recycling, R=0.99'
    elif 'recycle' in os.path.basename(dump):
        title_src = 'recycle-proxy (div_lo)'
    else:
        title_src = 'single-puff'
    plt.suptitle(f'DIII-D neutrals: OpenEdge Mode A ({title_src}) vs SOLPS-EIRENE coupled '
                 f'(reference, NOT standalone EIRENE)', y=0.99)
    plt.tight_layout()
    out = os.path.join(THIS, 'output', 'compare_to_eirene.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
