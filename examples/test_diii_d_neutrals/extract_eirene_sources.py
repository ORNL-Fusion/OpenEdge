#!/usr/bin/env python3
"""
Extract EIRENE per-cell source maps from SOLPS balance.nc, sum over strata,
save to a compact HDF5, and plot heatmaps over (R, Z).

Key EIRENE tallies in SOLPS balance.nc (all shape (nstra, natm/nspec, ny, nx)):
  eirene_mc_paat_sna_bal  : particle source on atoms from atomic processes
                            (atom ionization -- negative: atom lost)
  eirene_mc_paml_sna_bal  : particle source on atoms from molecular diss.
                            (molecule -> 2 atoms -- positive on atom side)
  eirene_mc_pmat_sna_bal  : particle source on molecules from atoms
                            (molecule dissociation -- negative on molecule)
  eirene_mc_papl_sna_bal  : particle source on plasma from atoms
                            (ionization source on ion species -- positive)
  eirene_mc_piat_sna_bal  : particle source on atoms from plasma ions
                            (recombination source -- positive on atom)
  eirene_mc_pael_sne_bal  : electron source from atoms   (ionization / rec)
  eirene_mc_pmel_sne_bal  : electron source from molecules
  eirene_mc_eael_she_bal  : electron energy source from atoms
  eirene_mc_empl_shi_bal  : ion energy source from molecules

For OpenEdge comparison, the quantities of interest per cell are:
  S_iz   = -paat_sna_total    (positive ionization rate, m^-3 s^-1)
  S_cx   =  b2stcx_sna_bal   (CX source on ion species)
  S_diss = -pmat_sna_total    (positive dissociation rate of D2)
"""
import os, sys
import numpy as np
import h5py
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
SOLPS = '/home/cloud/solps-runs/diii-d/runners_d2/run_lore2023_reference/balance.nc'
OUT_H5 = os.path.join(THIS, 'input', 'eirene_truth.h5')
OUT_PLOT = os.path.join(THIS, 'output', 'eirene_truth.png')

def main():
    ds = Dataset(SOLPS)
    # cell-center coordinates (average of 4 corners)
    crx = ds['crx'][:]   # (4, ny, nx)
    cry = ds['cry'][:]
    R = crx.mean(axis=0)
    Z = cry.mean(axis=0)
    vol = ds['vol'][:]   # (ny, nx)  m^3

    # Plasma state (for context)
    ne = ds['ne'][:]
    te = ds['te'][:] / 1.602176634e-19   # eV
    ti = ds['ti'][:] / 1.602176634e-19   # eV

    # EIRENE sources: sum over strata (axis 0) + species axis where needed
    # SOLPS writes them as PER CELL, INTEGRATED over cell volume  (units: particles/s)
    # Divide by vol to get per-volume rate [m^-3 s^-1]
    # SOLPS/EIRENE tally naming: p_AB_sna = particle source on SPECIES A, from
    # PROCESS involving B. sum over strata + species dim.
    paat = ds['eirene_mc_paat_sna_bal'][:].sum(axis=(0, 1))   # atom loss to atomic-ioniz. (<=0)
    pmml = ds['eirene_mc_pmml_sna_bal'][:].sum(axis=(0, 1))   # molecule loss to mol-process (<=0)
    pmat = ds['eirene_mc_pmat_sna_bal'][:].sum(axis=(0, 1))   # atom GAIN from mol. diss. (>=0)
    piat = ds['eirene_mc_piat_sna_bal'][:].sum(axis=(0, 1))   # atom GAIN from ion-involving proc (>=0)
    scx  = ds['b2stcx_sna_bal'][:].sum(axis=0)                # B2 CX tally (any sign)

    # Convert to per-volume rates (balance.nc tallies are integrated per cell)
    S_iz   = -paat / vol     # atom ionization rate [m^-3 s^-1]
    S_diss = -pmml / vol     # D2 dissociation rate [m^-3 s^-1] (molecules lost)
    S_rec  =  piat / vol     # atom production from recombination
    S_cx   =  np.abs(scx) / vol

    print(f'grid shape: {R.shape}')
    print(f'R range: [{R.min():.3f}, {R.max():.3f}] m')
    print(f'Z range: [{Z.min():.3f}, {Z.max():.3f}] m')
    print(f'domain-integrated ionization (-paat)  : {(-paat).sum():.3e} part/s')
    print(f'domain-integrated dissociation (-pmml): {(-pmml).sum():.3e} part/s')
    print(f'domain-integrated recombination (piat): {  piat.sum():.3e} part/s')
    print(f'peak S_iz   = {S_iz.max():.3e} m^-3 s^-1')
    print(f'peak S_diss = {S_diss.max():.3e} m^-3 s^-1')

    # Write compact HDF5
    os.makedirs(os.path.dirname(OUT_H5), exist_ok=True)
    with h5py.File(OUT_H5, 'w') as f:
        f['R'] = R;  f['R'].attrs['units'] = 'm'
        f['Z'] = Z;  f['Z'].attrs['units'] = 'm'
        f['vol'] = vol
        f['S_iz']   = S_iz
        f['S_diss'] = S_diss
        f['S_rec']  = S_rec
        f['S_cx']   = S_cx
        f['ne'] = ne
        f['te_eV'] = te
        f['ti_eV'] = ti
    print(f'wrote {OUT_H5}')

    # 2x2 heatmap: Te, ne, S_iz, S_diss
    def panel(ax, data, title, vmin_fallback=1e15):
        pos = data[data > 0]
        if pos.size == 0:
            im = ax.pcolormesh(R, Z, data, cmap='viridis', shading='nearest')
        else:
            im = ax.pcolormesh(R, Z, np.maximum(data, pos.min()),
                               norm=matplotlib.colors.LogNorm(
                                   vmin=max(vmin_fallback, pos.min()), vmax=pos.max()),
                               cmap='viridis', shading='nearest')
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.12)
        ax.set_aspect('equal'); ax.set_xlabel('R [m]'); ax.set_title(title)

    fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True)
    panel(axes[0], te,     r'$T_e$ [eV]', vmin_fallback=1)
    panel(axes[1], ne,     r'$n_e$ [m$^{-3}$]', vmin_fallback=1e17)
    panel(axes[2], S_iz,   r'$S_{iz}$ [m$^{-3}$s$^{-1}$] EIRENE', vmin_fallback=1e15)
    panel(axes[3], S_diss, r'$S_{diss}$ (D$_2$) [m$^{-3}$s$^{-1}$] EIRENE', vmin_fallback=1e14)
    axes[0].set_ylabel('Z [m]')
    plt.suptitle('DIII-D SOLPS / EIRENE (reference) — lore2023_reference', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120, bbox_inches='tight')
    print(f'wrote {OUT_PLOT}')

if __name__ == '__main__':
    main()
