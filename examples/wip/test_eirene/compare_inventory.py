#!/usr/bin/env python3
"""
compare_inventory.py
--------------------
Post-processing for OpenEdge vs EIRENE benchmark.

Usage:
  python3 compare_inventory.py [openedge.log]

Reads OpenEdge stats output and plots species inventory vs time.
If EIRENE steady-state data is available, overlays it.

Outputs:
  output/species_inventory.png
  output/rate_check.txt   -- rate coefficient comparison at Te=20 eV
"""

import sys
import os
import re
import numpy as np

# --------------------------------------------------------------------------
# 1. Parse OpenEdge stats output
# --------------------------------------------------------------------------
def parse_openedge_log(logfile):
    """Parse OpenEdge stats lines with columns:
       Step CPU np N_D N_D2 N_D+
    """
    steps, cpu, np_total, nD, nD2, nDp = [], [], [], [], [], []
    with open(logfile, 'r') as f:
        in_stats = False
        for line in f:
            # Stats lines start after the header line "Step CPU ..."
            if line.strip().startswith('Step'):
                in_stats = True
                continue
            if in_stats:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        steps.append(int(parts[0]))
                        cpu.append(float(parts[1]))
                        np_total.append(int(parts[2]))
                        nD.append(int(parts[3]))
                        nD2.append(int(parts[4]))
                        nDp.append(int(parts[5]))
                    except (ValueError, IndexError):
                        in_stats = False
                        continue
                else:
                    in_stats = False

    dt = 1e-7  # must match in.eirene_compare
    t = np.array(steps) * dt * 1e6  # time in microseconds
    return t, np.array(nD), np.array(nD2), np.array(nDp), np.array(np_total)


# --------------------------------------------------------------------------
# 2. Rate coefficient comparison
# --------------------------------------------------------------------------
def check_rates(Te_eV=20.0, ne_m3=1e19):
    """Compute and print rate coefficients from both sources."""

    print(f"\n{'='*65}")
    print(f"  Rate coefficient comparison at Te = {Te_eV} eV, ne = {ne_m3:.1e} m-3")
    print(f"{'='*65}\n")

    # --- D2 dissociation: Janev polynomial ---
    # ln<sv> = sum b_n (ln Te)^n, <sv> in cm3/s
    b = [-2.787217511174e+01, 1.052252660075e+01, -4.973212347860e+00,
          1.451198183114e+00, -3.062790554644e-01,  4.433379509258e-02,
         -4.096344172875e-03,  2.159670289222e-04, -4.928545325189e-06]
    lnTe = np.log(Te_eV)
    ln_sv = sum(b[n] * lnTe**n for n in range(len(b)))
    sv_dissoc_cm3s = np.exp(ln_sv)
    sv_dissoc_m3s  = sv_dissoc_cm3s * 1e-6

    print(f"  D2 dissociation (Janev H.2 2.2.5):")
    print(f"    <sv> = {sv_dissoc_cm3s:.4e} cm3/s = {sv_dissoc_m3s:.4e} m3/s")
    print(f"    rate = ne * <sv> = {ne_m3 * sv_dissoc_m3s:.4e} /s")
    print(f"    tau  = 1/rate = {1.0/(ne_m3 * sv_dissoc_m3s)*1e6:.2f} us")

    # --- ADAS rates (ADF11 HDF5: log10(Te/eV), log10(ne/cm3), log10(<sv>/cm3/s)) ---
    adas_file = os.path.join(os.path.dirname(__file__),
                             '../../database/adas/ADAS_Rates_1.h5')
    if os.path.exists(adas_file):
        try:
            import h5py
            def bilinear_log(rate_log, logTe_grid, logNe_grid, logTe, logNe):
                """Bilinear interpolation in log-log space on a rank-2 slice."""
                iT = np.clip(np.searchsorted(logTe_grid, logTe) - 1, 0, len(logTe_grid) - 2)
                iN = np.clip(np.searchsorted(logNe_grid, logNe) - 1, 0, len(logNe_grid) - 2)
                x0, x1 = logTe_grid[iT], logTe_grid[iT + 1]
                y0, y1 = logNe_grid[iN], logNe_grid[iN + 1]
                tx = (logTe - x0) / (x1 - x0)
                ty = (logNe - y0) / (y1 - y0)
                f00 = rate_log[iT, iN]
                f10 = rate_log[iT + 1, iN]
                f01 = rate_log[iT, iN + 1]
                f11 = rate_log[iT + 1, iN + 1]
                return (f00 * (1 - tx) * (1 - ty) + f10 * tx * (1 - ty)
                        + f01 * (1 - tx) * ty + f11 * tx * ty)

            logTe = np.log10(Te_eV)
            logNe_cm3 = np.log10(ne_m3 * 1e-6)  # convert m^-3 -> cm^-3

            labels = {
                'IonizationRateCoeff':     ('D ionization (ADAS SCD)',     'Ionization'),
                'RecombinationRateCoeff':  ('D+ recombination (ADAS ACD)', 'Recombination'),
                'ChargeExchangeRateCoeff': ('D+ charge exchange (ADAS CCD)', 'ChargeExchange'),
            }
            with h5py.File(adas_file, 'r') as f:
                for dset_name, (label, suffix) in labels.items():
                    if dset_name not in f:
                        continue
                    # shape = (charge_state, nTe, nNe); Z=1 has one charge state
                    rate_log = f[dset_name][0, :, :]
                    logTe_grid = f[f'gridTemperature_{suffix}'][:]
                    logNe_grid = f[f'gridDensity_{suffix}'][:]
                    # Clamp query point into grid range
                    qTe = np.clip(logTe, logTe_grid[0], logTe_grid[-1])
                    qNe = np.clip(logNe_cm3, logNe_grid[0], logNe_grid[-1])
                    log10_sv_cm3s = bilinear_log(rate_log, logTe_grid, logNe_grid,
                                                 qTe, qNe)
                    sv_cm3s = 10.0 ** log10_sv_cm3s
                    sv_m3s = sv_cm3s * 1e-6
                    print(f"\n  {label}:")
                    print(f"    <sv> = {sv_cm3s:.4e} cm3/s = {sv_m3s:.4e} m3/s")
                    if sv_m3s > 0:
                        rate = ne_m3 * sv_m3s
                        print(f"    rate = ne * <sv> = {rate:.4e} /s")
                        print(f"    tau  = 1/rate = {1.0/rate*1e6:.2f} us")
        except ImportError:
            print("\n  [h5py not available -- install it to read ADAS rates]")
    else:
        print(f"\n  [ADAS file not found at {adas_file}]")
        print("  Run this from the test_eirene/ directory to find it.")

    print(f"\n{'='*65}\n")


# --------------------------------------------------------------------------
# 3. Plot
# --------------------------------------------------------------------------
def plot_inventory(t, nD, nD2, nDp, np_total, outdir='output'):
    """Plot species inventory vs time."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[matplotlib not available -- skipping plot]")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: particle counts
    ax1.plot(t, nD2, 'b-', linewidth=2, label='D$_2$')
    ax1.plot(t, nD,  'r-', linewidth=2, label='D')
    ax1.plot(t, nDp, 'g-', linewidth=2, label='D$^+$')
    ax1.plot(t, np_total, 'k--', linewidth=1, label='Total')
    ax1.set_ylabel('Particle count')
    ax1.legend(loc='right')
    ax1.set_title('OpenEdge: Species inventory vs time\n'
                  r'(Te=20 eV, ne=10$^{19}$ m$^{-3}$, 0D box)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.1, 200)

    # Bottom: ionization fraction
    total_atoms = 2*nD2 + nD + nDp  # total D nuclei
    if total_atoms.max() > 0:
        ion_frac = nDp / total_atoms.astype(float)
        ax2.plot(t, ion_frac, 'g-', linewidth=2)
    ax2.set_xlabel(r'Time ($\mu$s)')
    ax2.set_ylabel('Ionization fraction (D$^+$ / total D nuclei)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'species_inventory.png')
    plt.savefig(outpath, dpi=150)
    print(f"Saved: {outpath}")
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # Rate coefficient check (always runs)
    check_rates()

    # Parse and plot OpenEdge output
    logfile = sys.argv[1] if len(sys.argv) > 1 else 'output/openedge.log'
    if os.path.exists(logfile):
        print(f"Parsing OpenEdge log: {logfile}")
        t, nD, nD2, nDp, np_total = parse_openedge_log(logfile)
        if len(t) > 0:
            print(f"  Found {len(t)} data points, t = {t[0]:.1f} to {t[-1]:.1f} us")
            print(f"  Initial: D2={nD2[0]}, D={nD[0]}, D+={nDp[0]}")
            print(f"  Final:   D2={nD2[-1]}, D={nD[-1]}, D+={nDp[-1]}")
            plot_inventory(t, nD, nD2, nDp, np_total)
        else:
            print("  No data found in log file.")
    else:
        print(f"No OpenEdge log found at {logfile}")
        print("Run OpenEdge first, then re-run this script:")
        print("  cd openedge/")
        print("  mpirun -np 1 /path/to/spa_mpi -in in.eirene_compare | tee ../output/openedge.log")


