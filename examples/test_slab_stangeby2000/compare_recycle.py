#!/usr/bin/env python3
"""
A2 verification: 1-D slab with reflecting wall at xhi.

The A1 closed form n(x) = (Gamma_0/v_n) exp(-x/lambda_iz) does NOT apply here
because a fraction of neutrals reaching x=L is reflected with a cosine
angular distribution at fixed energy (3 eV). There is no simple closed form
for the cosine case (you would need a full transport equation); instead we
build a 1-D Monte Carlo reference that implements exactly the OpenEdge
physics:
    - emit mono-energetic D at x=0 with thermal spread (350 K around v_n)
    - straight-line propagation in 3-D velocity space, binned in x
    - ionize with Poisson rate nu_iz = n_e * <sigma v>_iz
    - on hitting x=L: reflect 100%% with cosine polar angle, same |v|

If OpenEdge and the Python MC agree to within MC noise, the
surf_react wall_pwi EXCHANGE channel is verified.
"""
import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from compare_iz import sv_iz, read_grid_dump, TE_EV, NE_M3, EN_PUFF_EV, \
                       NRHO_PUFF, MD_KG, QE

L_BOX      = 0.6        # box xhi [m]
L_WALL     = 0.599      # surface position [m]
T_PUFF_K   = 350.0
N_MC       = 80_000
KB         = 1.380649e-23

# ----- Monte Carlo -----------------------------------------------------------
def accumulate(x0, vx, dt, edges, bins):
    """Add to `bins` the time spent in each bin over straight-line motion
    from (x0) in direction vx for duration dt. Assumes edges are sorted."""
    if vx == 0.0 or dt <= 0.0:
        return
    x1 = x0 + vx * dt
    xa, xb = (x0, x1) if x0 < x1 else (x1, x0)
    i0 = max(0, min(len(bins)-1, np.searchsorted(edges, xa, side='right') - 1))
    i1 = max(0, min(len(bins)-1, np.searchsorted(edges, xb, side='right') - 1))
    inv_v = 1.0 / abs(vx)
    for i in range(i0, i1 + 1):
        lo = max(edges[i],   xa)
        hi = min(edges[i+1], xb)
        if hi > lo:
            bins[i] += (hi - lo) * inv_v

def mc_run(nu_iz, v_n_ref, edges, N, rng):
    """Return density profile [m^-3] from a monoenergetic-drift Maxwellian
    puff into a slab with reflecting wall at L_WALL (cosine, same |v|)."""
    bins = np.zeros(len(edges) - 1)
    sigma_v = np.sqrt(KB * T_PUFF_K / MD_KG)
    v_refl = np.sqrt(2 * EN_PUFF_EV * QE / MD_KG)  # fixed-energy reflected speed

    for _ in range(N):
        # initial v: drift + thermal, reject if vx<=0 (flux into the box)
        while True:
            vx = v_n_ref + rng.normal() * sigma_v
            if vx > 0:
                break
        # vy, vz unused for 1-D residence but costs nothing to track
        t_life = -np.log(rng.uniform()) / nu_iz
        x = 0.0
        t = 0.0
        while t < t_life:
            if vx > 0:
                t_wall = (L_WALL - x) / vx
            elif vx < 0:
                t_wall = -x / vx
            else:
                break
            dt_seg = min(t_wall, t_life - t)
            accumulate(x, vx, dt_seg, edges, bins)
            x += vx * dt_seg
            t += dt_seg
            if t >= t_life - 1e-15:
                break
            if x >= L_WALL - 1e-9:
                # cosine reflection at fixed |v| = v_refl
                cos_th = np.sqrt(rng.uniform())
                vx = -v_refl * cos_th
                x = L_WALL - 1e-9
            elif x <= 1e-9:
                break  # exited at x=0

    # convert accumulated residence time [s] to density [m^-3]:
    #   each MC particle represents Gamma_0 / N real particles/s injected,
    #   so n(bin) = (Gamma_0/N) * sum_tau_in_bin / dx
    dx = edges[1] - edges[0]
    Gamma_0 = NRHO_PUFF * v_n_ref
    return (Gamma_0 / N) * bins / dx

# ----- Main ------------------------------------------------------------------
def main():
    dump = os.path.join(THIS, 'output', 'slab_recycle.grid')
    if not os.path.isfile(dump):
        sys.exit(f'No grid dump at {dump} -- run in.slab_recycle first.')
    d = read_grid_dump(dump)

    xc = d['xc']
    n_sim = d['f_fden[1]']
    order = np.argsort(xc)
    x = xc[order]
    n_sim = n_sim[order]

    # reconstruct bin edges from cell centers
    dx = x[1] - x[0]
    edges = np.concatenate([[x[0] - dx/2], x + dx/2])

    sv = sv_iz(TE_EV, NE_M3)
    v_n = np.sqrt(2 * EN_PUFF_EV * QE / MD_KG)
    nu_iz = NE_M3 * sv

    rng = np.random.default_rng(12345)
    n_mc = mc_run(nu_iz, v_n, edges, N_MC, rng)

    # also overplot the A1 pure-decay (no wall) for reference
    Gamma_0 = NRHO_PUFF * v_n
    n_a1 = (Gamma_0 / v_n) * np.exp(-x / (v_n / nu_iz))

    # error metric: RMSE over x in [0.1L, 0.9L] after MC-normalized
    mask = (x > 0.05 * L_BOX) & (x < 0.98 * L_BOX) & (n_mc > 0)
    rel_err = (n_sim[mask] - n_mc[mask]) / n_mc[mask]
    rms = np.sqrt(np.mean(rel_err**2))
    peak_sim = n_sim.max()
    peak_mc  = n_mc.max()

    print(f'<sigma v>_iz = {sv:.3e} m^3/s, nu_iz = {nu_iz:.3e} /s, lambda = {1e3*v_n/nu_iz:.1f} mm')
    print(f'peak density (sim / MC)  = {peak_sim:.3e} / {peak_mc:.3e} m^-3')
    print(f'RMS relative error       = {100*rms:.1f} % (over 0.05L < x < 0.98L)')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, log in zip(axes, [False, True]):
        ax.plot(x*1000, n_sim, 'o', ms=3, label='OpenEdge', color='C0')
        ax.plot(x*1000, n_mc,  '-', lw=1.5, label=f'Python MC (N={N_MC})', color='C3')
        ax.plot(x*1000, n_a1,  '--', lw=1.0, label='A1 exp(-x/λ) (no wall)', color='C7', alpha=0.7)
        ax.set_xlabel('x [mm]'); ax.set_ylabel(r'$n_D$ [m$^{-3}$]')
        ax.grid(True, alpha=0.3); ax.legend(loc='upper right')
        if log:
            ax.set_yscale('log')
            ax.set_ylim(1e-2*max(peak_sim,peak_mc), 2*max(peak_sim,peak_mc))
    axes[0].set_title(f'A2: reflecting wall at x={L_WALL*1000:.0f} mm, cosine @ 3 eV')
    axes[1].set_title(f'sim vs MC RMS err = {100*rms:.1f} %')
    plt.tight_layout()
    out = os.path.join(THIS, 'output', 'compare_recycle.png')
    plt.savefig(out, dpi=120)
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
