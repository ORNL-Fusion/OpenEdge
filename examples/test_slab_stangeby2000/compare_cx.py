#!/usr/bin/env python3
"""
A3 verification: 1-D slab with CX thermalization.

A monoenergetic 3 eV D beam enters a plasma with bulk Ti = 50 eV. At each
spatial point, neutrals have three competing futures:
  - free flight
  - ionization (Mode A: delete, at rate nu_iz = ne <sigma v>_iz(Te))
  - CX: resample velocity from Maxwellian at Ti (rate nu_cx = ne <sigma v>_cx(Te))

There is no closed-form for T_D(x) with competing channels. We build a 1-D
Monte Carlo reference that reproduces exactly the `fix chem/adas`
implementation: per-step Poisson competition between iz and CX, CX product
drawn from an isotropic Maxwellian at Ti.

OpenEdge's `compute grid species temp` uses SPARTA's raw kinetic temperature
convention (see compute_grid.cpp MVSQ case):
    T_eV = m_D * <|v|^2> / (3 * QE)
i.e. NO mean subtraction. For a monoenergetic beam at E=3 eV this gives
T_SPARTA = 2*E/3 = 2 eV at the source; for a Maxwellian at Ti it gives Ti.
"""
import os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from compare_iz import sv_iz, read_grid_dump, NRHO_PUFF, MD_KG, QE

KB = 1.380649e-23

# ---- Match in.slab_cx exactly -----------------------------------------------
TE_EV     = 5.0
TI_EV     = 50.0
NE_M3     = 1.0e20
EN_PUFF_EV = 3.0
T_PUFF_K  = 350.0
L_BOX     = 0.6
N_MC      = 30_000
ADAS      = os.path.join(THIS, '..', '..', 'database', 'adas', 'ADAS_Rates_1.h5')

# ---- CX rate (same bilinear log-space interp as sv_iz) ----------------------
def sv_cx(Te_eV, ne_m3):
    with h5py.File(ADAS, 'r') as f:
        Te_log = f['gridTemperature_ChargeExchange'][:]
        ne_log = f['gridDensity_ChargeExchange'][:]
        rate   = f['ChargeExchangeRateCoeff'][0]
    lT = np.clip(np.log10(Te_eV), Te_log[0], Te_log[-1])
    lN = np.clip(np.log10(ne_m3 * 1e-6), ne_log[0], ne_log[-1])
    i = min(len(Te_log) - 2, max(0, np.searchsorted(Te_log, lT) - 1))
    j = min(len(ne_log) - 2, max(0, np.searchsorted(ne_log, lN) - 1))
    a = (lT - Te_log[i]) / (Te_log[i+1] - Te_log[i])
    b = (lN - ne_log[j]) / (ne_log[j+1] - ne_log[j])
    r00, r01, r10, r11 = rate[i,j], rate[i,j+1], rate[i+1,j], rate[i+1,j+1]
    lr = (1-a)*(1-b)*r00 + (1-a)*b*r01 + a*(1-b)*r10 + a*b*r11
    return 10**lr * 1e-6

# ---- Monte Carlo ------------------------------------------------------------
def _accumulate(x0, vx, dt, edges, bins_n, bins_vx, bins_vy, bins_vz, bins_v2,
                vx_, vy_, vz_, v2_):
    if vx == 0.0 or dt <= 0.0:
        return
    x1 = x0 + vx * dt
    xa, xb = (x0, x1) if x0 < x1 else (x1, x0)
    i0 = max(0, min(len(bins_n) - 1, np.searchsorted(edges, xa, side='right') - 1))
    i1 = max(0, min(len(bins_n) - 1, np.searchsorted(edges, xb, side='right') - 1))
    inv_v = 1.0 / abs(vx)
    for i in range(i0, i1 + 1):
        lo = max(edges[i], xa)
        hi = min(edges[i+1], xb)
        if hi > lo:
            dtau = (hi - lo) * inv_v
            bins_n[i]  += dtau
            bins_vx[i] += dtau * vx_
            bins_vy[i] += dtau * vy_
            bins_vz[i] += dtau * vz_
            bins_v2[i] += dtau * v2_

def mc_run(N, edges, rng):
    nu_iz = NE_M3 * sv_iz(TE_EV, NE_M3)
    nu_cx = NE_M3 * sv_cx(TE_EV, NE_M3)
    nu_tot = nu_iz + nu_cx
    p_iz = nu_iz / nu_tot

    v_th_i = np.sqrt(KB * (TI_EV * QE / KB) / MD_KG)    # isotropic thermal speed at Ti
    sigma_v_p = np.sqrt(KB * T_PUFF_K / MD_KG)
    v_n_puff = np.sqrt(2 * EN_PUFF_EV * QE / MD_KG)

    nb = len(edges) - 1
    bins_n  = np.zeros(nb)
    bins_vx = np.zeros(nb); bins_vy = np.zeros(nb); bins_vz = np.zeros(nb)
    bins_v2 = np.zeros(nb)

    for _ in range(N):
        while True:
            vx = v_n_puff + rng.normal() * sigma_v_p
            if vx > 0: break
        vy, vz = rng.normal() * sigma_v_p, rng.normal() * sigma_v_p
        x = 0.0

        while True:
            t_event = -np.log(rng.uniform()) / nu_tot
            # check exit
            if vx > 0:
                t_exit = (L_BOX - x) / vx
            elif vx < 0:
                t_exit = -x / vx
            else:
                t_exit = np.inf
            alive = t_event < t_exit
            t_seg = t_event if alive else t_exit

            v2 = vx*vx + vy*vy + vz*vz
            _accumulate(x, vx, t_seg, edges,
                        bins_n, bins_vx, bins_vy, bins_vz, bins_v2,
                        vx, vy, vz, v2)
            x += vx * t_seg
            if not alive:
                break

            if rng.uniform() < p_iz:
                break  # ionized
            # CX: resample v from isotropic Maxwellian at Ti
            vx = rng.normal() * v_th_i
            vy = rng.normal() * v_th_i
            vz = rng.normal() * v_th_i

    # convert to density [m^-3] and temperature [eV]
    dx = edges[1] - edges[0]
    Gamma_0 = NRHO_PUFF * v_n_puff
    n_mc = (Gamma_0 / N) * bins_n / dx

    T_mc = np.zeros(nb)
    mask = bins_n > 0
    T_mc[mask] = MD_KG * (bins_v2[mask] / bins_n[mask]) / (3.0 * QE)
    return n_mc, T_mc, nu_iz, nu_cx

# ---- Main -------------------------------------------------------------------
def main():
    dump = os.path.join(THIS, 'output', 'slab_cx.grid')
    if not os.path.isfile(dump):
        sys.exit(f'No grid dump at {dump} -- run in.slab_cx first.')
    d = read_grid_dump(dump)

    xc = d['xc']
    order = np.argsort(xc)
    x  = xc[order]
    n_sim = d['f_fmom[1]'][order]
    T_sim = d['f_fmom[2]'][order]                 # SPARTA writes T in K
    T_sim_eV = T_sim * KB / QE

    dx = x[1] - x[0]
    edges = np.concatenate([[x[0] - dx/2], x + dx/2])

    rng = np.random.default_rng(42)
    n_mc, T_mc_eV, nu_iz, nu_cx = mc_run(N_MC, edges, rng)

    print(f'nu_iz = {nu_iz:.3e} /s   nu_cx = {nu_cx:.3e} /s   cx/iz = {nu_cx/nu_iz:.2f}')
    print(f'asymptotic T_D (SPARTA raw <m|v|^2>/(3kB) convention) = Ti = {TI_EV:.1f} eV')
    print(f'puff T_D at x=0 (monoenergetic 3 eV beam)            = 2/3 * E = {2*EN_PUFF_EV/3:.2f} eV')
    mid = (x > 0.3*L_BOX) & (x < 0.8*L_BOX) & (n_mc > 0)
    print(f'mid-slab  T_D (sim)  = {T_sim_eV[mid].mean():.1f} eV')
    print(f'mid-slab  T_D (MC)   = {T_mc_eV[mid].mean():.1f} eV')

    mask = (x > 0.03*L_BOX) & (x < 0.97*L_BOX) & (n_mc > 0)
    rms_n = np.sqrt(np.mean(((n_sim[mask] - n_mc[mask]) / n_mc[mask])**2))
    rms_T = np.sqrt(np.mean(((T_sim_eV[mask] - T_mc_eV[mask]) / T_mc_eV[mask])**2))
    print(f'RMS rel err n_D(x)   = {100*rms_n:.1f} %')
    print(f'RMS rel err T_D(x)   = {100*rms_T:.1f} %')

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x*1000, n_sim, 'o', ms=3, label='OpenEdge', color='C0')
    ax[0].plot(x*1000, n_mc,  '-', lw=1.5, label=f'Python MC (N={N_MC})', color='C3')
    ax[0].set_yscale('log')
    ax[0].set_ylim(1e-3*max(n_sim.max(), n_mc.max()), 2*max(n_sim.max(), n_mc.max()))
    ax[0].set_xlabel('x [mm]'); ax[0].set_ylabel(r'$n_D$ [m$^{-3}$]')
    ax[0].set_title('A3: density'); ax[0].legend(); ax[0].grid(True, alpha=0.3)

    ax[1].plot(x*1000, T_sim_eV, 'o', ms=3, label='OpenEdge', color='C0')
    ax[1].plot(x*1000, T_mc_eV,  '-', lw=1.5, label='Python MC', color='C3')
    ax[1].axhline(TI_EV, ls='--', lw=1, color='k', alpha=0.5,
                  label=f'Ti = {TI_EV:.0f} eV (asymptote)')
    ax[1].set_xlabel('x [mm]'); ax[1].set_ylabel(r'$T_D$ [eV]')
    ax[1].set_title(f'A3: CX thermalization  (n_cx/n_iz = {nu_cx/nu_iz:.1f})')
    ax[1].legend(); ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(THIS, 'output', 'compare_cx.png')
    plt.savefig(out, dpi=120)
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
