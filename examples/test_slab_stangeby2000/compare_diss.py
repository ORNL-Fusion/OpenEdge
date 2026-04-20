#!/usr/bin/env python3
"""
A4 verification: D2 dissociation slab.

Reference: Stangeby 2000 Ch. 3 -- same exponential-decay geometry as A1
applied to the molecular reactant:
    n_D2(x) = (Gamma_0 / v_D2) * exp(-x / lambda_diss)
    lambda_diss = v_D2 / (n_e * <sigma v>_diss(T_e))

Rate from Janev/HYDHEL H.2 2.2.5 polynomial (the same coefficients compiled
into the neutral_diss.reactions file).

Also validates the 20-col source-term tally: f_frate[4] (dissociation rate
per cell, [m^-3 s^-1]) should equal nu_diss * n_D2(x) to within MC noise.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from compare_iz import read_grid_dump

QE   = 1.602176634e-19
MD2  = 6.690e-27     # must match neutral_diss.species Molmass
TE_EV   = 15.0
NE_M3   = 1.0e19
EN_PUFF_EV = 3.0
NRHO_PUFF  = 1.0e18

# Janev coefficients from neutral_diss.reactions (H.2 2.2.5)
JANEV_B = np.array([
    -2.787217511174e+01,  1.052252660075e+01, -4.973212347860e+00,
     1.451198183114e+00, -3.062790554644e-01,  4.433379509258e-02,
    -4.096344172875e-03,  2.159670289222e-04, -4.928545325189e-06])

def sv_diss(Te_eV):
    """Janev <sv>_diss in m^3/s."""
    lnT = np.log(Te_eV)
    lnsv = np.polynomial.polynomial.polyval(lnT, JANEV_B)
    return np.exp(lnsv) * 1e-6   # cm^3/s -> m^3/s

def main():
    dump = os.path.join(THIS, 'output', 'slab_diss.grid')
    if not os.path.isfile(dump):
        sys.exit(f'No dump at {dump} -- run in.slab_diss first.')
    d = read_grid_dump(dump)

    xc = d['xc']
    order = np.argsort(xc)
    x = xc[order]
    n_D2   = d['f_fden[1]'][order]
    n_D    = d['f_fden[2]'][order]
    S_diss = d['f_frate[4]'][order]            # dissociation rate [m^-3 s^-1]

    sv = sv_diss(TE_EV)
    v_D2 = np.sqrt(2 * EN_PUFF_EV * QE / MD2)
    nu_diss = NE_M3 * sv
    lam = v_D2 / nu_diss
    Gamma_0 = NRHO_PUFF * v_D2

    # (i) analytic n_D2(x) decay
    n_ana = (Gamma_0 / v_D2) * np.exp(-x / lam)

    # (ii) self-consistency: source rate should be nu_diss * n_D2(x)
    S_expected = nu_diss * n_D2

    # fits + errors
    mask = (x > lam) & (x < 4*lam) & (n_D2 > 0)
    slope, _ = np.polyfit(x[mask], np.log(n_D2[mask]), 1)
    lam_fit = -1.0 / slope

    # RMS on n_D2 and on source tally
    mfull = (x > 0.03*0.6) & (x < 0.95*0.6) & (n_ana > 0) & (n_D2 > 0)
    rms_n = np.sqrt(np.mean(((n_D2[mfull] - n_ana[mfull]) / n_ana[mfull])**2))
    rms_S = np.sqrt(np.mean(((S_diss[mfull] - S_expected[mfull])
                             / S_expected[mfull])**2))

    print(f'Janev <sv>_diss(Te={TE_EV} eV)        = {sv:.3e} m^3/s')
    print(f'v_D2 (D2 at {EN_PUFF_EV} eV)           = {v_D2:.3e} m/s')
    print(f'nu_diss                           = {nu_diss:.3e} /s')
    print(f'lambda_diss (analytic)            = {lam*1000:.2f} mm')
    print(f'lambda_diss (sim fit)             = {lam_fit*1000:.2f} mm  '
          f'({100*(lam_fit-lam)/lam:+.1f} %)')
    print(f'RMS rel err n_D2 (sim vs analytic)        = {100*rms_n:.1f} %')
    print(f'RMS rel err S_diss (tally vs nu_diss*n_D2) = {100*rms_S:.1f} %')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # n_D2
    axes[0].plot(x*1000, n_D2, 'o', ms=3, label='OpenEdge n_D2', color='C0')
    axes[0].plot(x*1000, n_ana, '-', lw=1.5,
                 label=r'analytic $(\Gamma_0/v_{D_2})e^{-x/\lambda}$', color='C3')
    axes[0].plot(x*1000, n_D, '.', ms=2, label='OpenEdge n_D (product)',
                 color='C2', alpha=0.5)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('x [mm]'); axes[0].set_ylabel(r'$n$ [m$^{-3}$]')
    axes[0].set_title(f'λ_fit = {lam_fit*1000:.1f} mm vs analytic {lam*1000:.1f} mm')
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    # S_diss tally vs nu_diss * n_D2
    axes[1].plot(x*1000, S_diss, 'o', ms=3, label=r'tally $S_{diss}$', color='C0')
    axes[1].plot(x*1000, S_expected, '-', lw=1.5,
                 label=r'$\nu_{diss}\,n_{D_2}$ (from sim)', color='C3')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('x [mm]'); axes[1].set_ylabel(r'$S_{diss}$ [m$^{-3}$ s$^{-1}$]')
    axes[1].set_title(f'tally self-consistency (RMS {100*rms_S:.1f} %)')
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    # relative errors
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].plot(x*1000, 100*(n_D2 - n_ana)/np.where(n_ana>0,n_ana,1), '-',
                 label='n_D2 vs analytic', color='C0')
    axes[2].plot(x*1000, 100*(S_diss - S_expected)/np.where(S_expected>0,S_expected,1), '-',
                 label='S_diss vs nu*n_D2', color='C3')
    axes[2].set_xlabel('x [mm]'); axes[2].set_ylabel('relative error [%]')
    axes[2].set_title('residuals')
    axes[2].set_ylim(-30, 30)
    axes[2].grid(True, alpha=0.3); axes[2].legend()

    plt.tight_layout()
    out = os.path.join(THIS, 'output', 'compare_diss.png')
    plt.savefig(out, dpi=120)
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
