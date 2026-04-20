#!/usr/bin/env python3
"""
Analytic verification for the 1-D ionization slab.

Reference: P.C. Stangeby, "The Plasma Boundary of Magnetic Fusion Devices",
IoP 2000, Chapter 3.

Steady-state density of a monoenergetic neutral beam injected from x=0 into a
uniform ionizing plasma:

    n(x) = (Gamma_0 / v_n) * exp(-x / lambda_iz)
    lambda_iz = v_n / (n_e * <sigma v>_iz(T_e))

This script reads the OpenEdge grid dump (output/slab.grid) and overplots the
simulated n_D(x) against this prediction, using the ADAS rate at the run
conditions.
"""
import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
ADAS = os.path.join(THIS, '..', '..', 'database', 'adas', 'ADAS_Rates_1.h5')

# ---- Run parameters (must match in.slab) ------------------------------------
TE_EV   = 20.0
NE_M3   = 1.0e19
EN_PUFF_EV = 3.0                     # Franck-Condon D puff energy
NRHO_PUFF = 1.0e18                   # mixture nrho [m^-3]
MD_KG   = 3.345e-27                  # D mass
QE      = 1.602176634e-19

# ---- Ionization rate from ADAS (Z=1 H) --------------------------------------
def sv_iz(Te_eV, ne_m3, path=ADAS):
    with h5py.File(path, 'r') as f:
        Te_log = f['gridTemperature_Ionization'][:]
        ne_log = f['gridDensity_Ionization'][:]
        rate   = f['IonizationRateCoeff'][0]         # (nT, nN)
    lT = np.clip(np.log10(Te_eV), Te_log[0], Te_log[-1])
    lN = np.clip(np.log10(ne_m3 * 1e-6), ne_log[0], ne_log[-1])
    i = min(len(Te_log) - 2, max(0, np.searchsorted(Te_log, lT) - 1))
    j = min(len(ne_log) - 2, max(0, np.searchsorted(ne_log, lN) - 1))
    a = (lT - Te_log[i]) / (Te_log[i+1] - Te_log[i])
    b = (lN - ne_log[j]) / (ne_log[j+1] - ne_log[j])
    r00, r01, r10, r11 = rate[i,j], rate[i,j+1], rate[i+1,j], rate[i+1,j+1]
    lr = (1-a)*(1-b)*r00 + (1-a)*b*r01 + a*(1-b)*r10 + a*b*r11
    return 10**lr * 1e-6                              # cm^3/s -> m^3/s

# ---- SPARTA grid dump parser ------------------------------------------------
def read_grid_dump(path):
    """Return dict {colname: np.array} from a SPARTA ASCII grid dump."""
    with open(path) as f:
        lines = f.read().splitlines()
    # advance to last snapshot
    idx = [i for i,l in enumerate(lines) if l.startswith('ITEM: TIMESTEP')]
    start = idx[-1]
    n_cells = int(lines[start+3])
    header = lines[start+8].split()[2:]
    data = np.loadtxt(lines[start+9 : start+9+n_cells])
    return {k: data[:, i] for i, k in enumerate(header)}

# ---- Main -------------------------------------------------------------------
def main():
    dump_path = os.path.join(THIS, 'output', 'slab.grid')
    if not os.path.isfile(dump_path):
        sys.exit(f'No grid dump at {dump_path} -- run in.slab first.')
    d = read_grid_dump(dump_path)

    sv = sv_iz(TE_EV, NE_M3)
    v_n = np.sqrt(2 * EN_PUFF_EV * QE / MD_KG)
    nu_iz = NE_M3 * sv
    lam = v_n / nu_iz
    Gamma_0 = NRHO_PUFF * v_n

    # collapse to 1-D in x (bin by xc and average n over y, there's 1 y-cell)
    xc = d['xc']
    n_D = d['f_fden[1]']
    order = np.argsort(xc)
    x = xc[order]
    n_sim = n_D[order]

    n_analytic = (Gamma_0 / v_n) * np.exp(-x / lam)

    # summary: effective decay length from log-linear fit to sim (x in [lam, 4*lam])
    mask = (x > lam) & (x < 4 * lam) & (n_sim > 0)
    if mask.sum() > 5:
        slope, b = np.polyfit(x[mask], np.log(n_sim[mask]), 1)
        lam_fit = -1.0 / slope
    else:
        lam_fit = np.nan

    print(f'ADAS <sigma v>_iz(Te={TE_EV} eV, ne={NE_M3:.1e}) = {sv:.3e} m^3/s')
    print(f'v_n (D at {EN_PUFF_EV} eV) = {v_n:.3e} m/s')
    print(f'nu_iz = ne*<sv> = {nu_iz:.3e} /s')
    print(f'lambda_iz (analytic) = {lam*1000:.2f} mm')
    print(f'lambda_iz (sim fit)  = {lam_fit*1000:.2f} mm')
    if not np.isnan(lam_fit):
        print(f'relative error       = {100*(lam_fit-lam)/lam:+.1f} %')

    # ---- plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, log in zip(axes, [False, True]):
        ax.plot(x*1000, n_sim, 'o', ms=3, label='OpenEdge', color='C0')
        ax.plot(x*1000, n_analytic, '-', lw=1.5,
                label=r'$(\Gamma_0/v_n)\,e^{-x/\lambda_{iz}}$', color='C3')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel(r'$n_D$ [m$^{-3}$]')
        if log:
            ax.set_yscale('log')
            ax.set_ylim(1e-2 * n_analytic.max(), 2 * n_analytic.max())
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    axes[0].set_title(f'Te={TE_EV} eV, ne={NE_M3:.0e}, λ={lam*1000:.1f} mm')
    axes[1].set_title(f'sim λ_fit = {lam_fit*1000:.1f} mm'
                      if not np.isnan(lam_fit) else 'log scale')
    plt.tight_layout()
    out = os.path.join(THIS, 'output', 'compare.png')
    plt.savefig(out, dpi=120)
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
