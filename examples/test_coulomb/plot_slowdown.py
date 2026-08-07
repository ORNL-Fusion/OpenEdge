#!/usr/bin/env python3
"""
Analysis script for Nanbu background collision slowing-down test.

Reads particle dump files from output/particles_slowdown* and checks:
  1. C3+ temperature relaxes toward background D+ temperature (2 eV)
  2. Relaxation matches the NRL cross-species equipartition rate
     (T-dependent tau, integrated as an ODE)
  3. Speed distribution evolves from 10 eV Maxwellian toward 2 eV

Exits 0 on PASS, 1 on FAIL (for regression use).

Usage:
    python3 plot_slowdown.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ---------- Pass/fail thresholds ----------
RMS_TOL   = 0.10   # rms |T_sim - T_ode| / (T0 - T_bg)
FINAL_TOL = 0.15   # |T_final_sim - T_final_ode| / (T0 - T_bg)

# ---------- Physical constants ----------
e_charge = 1.602176634e-19      # C
eps0     = 8.8541878128e-12     # F/m
kB       = 1.380649e-23         # J/K

# ---------- Simulation parameters (must match in.nanbu_slowdown) ----------
TYPE_C3p = 1   # only species in simulation

m_C3p    = 12.011 * 1.66053906660e-27   # C3+ mass [kg]
m_Dp     = 2.014  * 1.66053906660e-27   # D+ background mass [kg]
Z_C3p    = 3
Z_Dp     = 1
dt       = 1.0e-8               # s
ne_bg    = 1.0e17               # background electron density [m^-3]
ni_bg    = 1.0e17               # background ion density [m^-3]
Te_bg    = 10.0                 # background Te for Coulomb log [eV]
Ti_bg    = 2.0                  # background D+ temperature [eV]
N_C3p    = 5000


# ---------- Dump file parser ----------
def parse_dump_file(filepath):
    """Parse a single SPARTA particle dump file (one or more timesteps)."""
    results = {}
    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    ts = None
    natoms = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            ts = int(lines[i + 1].strip())
            i += 2
        elif line == "ITEM: NUMBER OF ATOMS":
            natoms = int(lines[i + 1].strip())
            i += 2
        elif line.startswith("ITEM: BOX BOUNDS"):
            i += 4
        elif line.startswith("ITEM: ATOMS"):
            headers = line.split()[2:]
            idx = {h: j for j, h in enumerate(headers)}
            ids   = np.zeros(natoms, dtype=int)
            types = np.zeros(natoms, dtype=int)
            vx    = np.zeros(natoms)
            vy    = np.zeros(natoms)
            vz    = np.zeros(natoms)
            for k in range(natoms):
                vals = lines[i + 1 + k].split()
                ids[k]   = int(vals[idx["id"]])
                types[k] = int(vals[idx["type"]])
                vx[k]    = float(vals[idx["vx"]])
                vy[k]    = float(vals[idx["vy"]])
                vz[k]    = float(vals[idx["vz"]])
            results[ts] = {"id": ids, "type": types,
                           "vx": vx, "vy": vy, "vz": vz}
            i += 1 + natoms
        else:
            i += 1
    return results


def load_all_dumps(outdir="output"):
    """Load particle dumps — handles both single file and per-step files."""
    outpath = Path(outdir)

    single = outpath / "particles_slowdown"
    if single.exists():
        return parse_dump_file(single)

    files = sorted(outpath.glob("particles_slowdown.*"),
                   key=lambda p: int(p.suffix[1:]))
    if not files:
        print(f"ERROR: No particle dump files found in {outdir}/")
        sys.exit(1)

    all_data = {}
    for f in files:
        all_data.update(parse_dump_file(f))
    return all_data


# ---------- Physics helpers ----------
def kinetic_temperature(vx, vy, vz, mass):
    """T = m <v^2> / (3 kB)"""
    if len(vx) == 0:
        return 0.0
    v2_mean = np.mean(vx**2 + vy**2 + vz**2)
    return mass * v2_mean / (3.0 * kB)


def coulomb_log(ne, Te_eV):
    """ln Lambda = max(2, ln(12 pi ne lambda_D^3))"""
    Te_J = Te_eV * e_charge
    lam_D = np.sqrt(eps0 * Te_J / (ne * e_charge**2))
    lnL = np.log(12.0 * np.pi * ne * lam_D**3)
    return max(2.0, lnL)


def spitzer_cross_species_tau(Ta_eV, Tb_eV, ma, mb, Za, Zb, nb, lnL):
    """
    Cross-species energy equipartition time (NRL Plasma Formulary):

        nu_eq = 1.8e-19 sqrt(ma mb) Za^2 Zb^2 nb lnL
                / (ma Tb + mb Ta)^(3/2)      [m in g, n in cm^-3, T in eV]

    Inputs SI (kg, m^-3); returns time [s] for species a to equilibrate
    with species b.
    """
    ma_g, mb_g = ma * 1e3, mb * 1e3
    nb_cm3 = nb * 1e-6
    nu = (1.8e-19 * np.sqrt(ma_g * mb_g) * Za**2 * Zb**2 * nb_cm3 * lnL
          / (ma_g * Tb_eV + mb_g * Ta_eV)**1.5)
    return 1.0 / nu


def relax_ode(T0_eV, Tb_eV, ma, mb, Za, Zb, nb, lnL, t):
    """Integrate dT/dt = -(T - Tb)/tau(T) with T-dependent NRL tau."""
    T = np.empty_like(t)
    T[0] = T0_eV
    for i in range(1, len(t)):
        tau = spitzer_cross_species_tau(T[i-1], Tb_eV, ma, mb, Za, Zb, nb, lnL)
        T[i] = T[i-1] - (t[i] - t[i-1]) * (T[i-1] - Tb_eV) / tau
    return T


# ---------- Main ----------
def main():
    data = load_all_dumps("output")
    steps = sorted(data.keys())
    times    = np.array(steps) * dt
    times_us = times * 1e6

    print(f"Loaded {len(steps)} timesteps: {steps[0]} to {steps[-1]}")
    print(f"Time range: {times_us[0]:.1f} -- {times_us[-1]:.1f} us")

    # Compute temperature at each timestep
    T_C3p_arr = np.zeros(len(steps))
    KE_arr    = np.zeros(len(steps))

    for i, step in enumerate(steps):
        d = data[step]
        vx, vy, vz = d["vx"], d["vy"], d["vz"]

        T_C3p_arr[i] = kinetic_temperature(vx, vy, vz, m_C3p)
        KE_arr[i] = 0.5 * m_C3p * np.sum(vx**2 + vy**2 + vz**2)

    # Convert to eV
    T_C3p_eV = T_C3p_arr * kB / e_charge

    # Spitzer analytical curve
    lnL = coulomb_log(ne_bg, Te_bg)

    # Characteristic equipartition time at initial conditions
    tau_s = spitzer_cross_species_tau(
        T_C3p_eV[0], Ti_bg, m_C3p, m_Dp, Z_C3p, Z_Dp, ni_bg, lnL)
    tau_s_us = tau_s * 1e6

    print(f"\n--- Physical parameters ---")
    print(f"n_bg(D+) = {ni_bg:.2e} m^-3")
    print(f"Ti_bg    = {Ti_bg:.1f} eV")
    print(f"Coulomb log ln(Lambda) = {lnL:.2f}")
    print(f"Spitzer equipartition time tau_eq = {tau_s_us:.2f} us")

    print(f"\n--- Results ---")
    print(f"T_C3+ (t=0):             {T_C3p_eV[0]:.2f} eV")
    print(f"T_C3+ (t={times_us[-1]:.0f}us): {T_C3p_eV[-1]:.2f} eV")
    print(f"Target T_bg:             {Ti_bg:.2f} eV")

    # Analytical relaxation: dT/dt = -(T - T_bg)/tau(T), NRL tau
    t_fine    = np.linspace(0, times[-1], 2000)
    t_fine_us = t_fine * 1e6
    T_ana = relax_ode(T_C3p_eV[0], Ti_bg, m_C3p, m_Dp, Z_C3p, Z_Dp,
                      ni_bg, lnL, t_fine)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Temperature relaxation
    ax = axes[0]
    ax.plot(times_us, T_C3p_eV, "b-", lw=1.5, label="C3+ (simulation)")
    ax.plot(t_fine_us, T_ana, "r:", lw=1.5, alpha=0.7,
            label=f"NRL ODE (tau0={tau_s_us:.1f} us)")
    ax.axhline(Ti_bg, color="k", ls="--", lw=0.8,
               label=f"T_bg = {Ti_bg:.1f} eV")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Temperature [eV]")
    ax.set_title("C3+ slowing down against D+ background")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Kinetic energy
    ax = axes[1]
    KE_rel = (KE_arr - KE_arr[0]) / KE_arr[0]
    ax.plot(times_us, KE_rel * 100, "g-", lw=1.5)
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("dKE / KE_0 [%]")
    ax.set_title("Kinetic energy change")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)

    # (c) Speed distribution: initial vs final
    ax = axes[2]
    d0 = data[steps[0]]
    df = data[steps[-1]]

    sp_0 = np.sqrt(d0["vx"]**2 + d0["vy"]**2 + d0["vz"]**2)
    sp_f = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)

    all_speeds = np.concatenate([sp_0, sp_f])
    bins = np.linspace(0, np.percentile(all_speeds, 99.5), 60)

    ax.hist(sp_0, bins=bins, density=True, alpha=0.4, color="r",
            label="t=0 (10 eV)")
    ax.hist(sp_f, bins=bins, density=True, alpha=0.4, color="b",
            label=f"t={times_us[-1]:.0f}us")
    ax.set_xlabel("Speed [m/s]")
    ax.set_ylabel("PDF")
    ax.set_title("C3+ speed distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Nanbu background collision: C3+ slowing down on D+", fontsize=13)
    plt.tight_layout()
    plt.savefig("slowdown.png", dpi=150)
    print(f"\nPlot saved to slowdown.png")

    # ---- Pass/fail ----
    T_ode = np.interp(times, t_fine, T_ana)
    scale = T_C3p_eV[0] - Ti_bg
    rms   = float(np.sqrt(np.mean((T_C3p_eV - T_ode)**2))) / scale
    final = abs(T_C3p_eV[-1] - T_ode[-1]) / scale

    checks = [
        ("T(t) rms vs NRL ODE", rms,   RMS_TOL),
        ("final T error",       final, FINAL_TOL),
    ]
    failed = False
    for name, val, tol in checks:
        ok = val < tol
        failed |= not ok
        print(f"  {name}: {val:.4f} (tol {tol}) {'ok' if ok else 'FAIL'}")
    print("PASS" if not failed else "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
