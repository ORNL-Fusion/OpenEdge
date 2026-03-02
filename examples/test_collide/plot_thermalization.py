#!/usr/bin/env python3
"""
Analysis script for Nanbu Coulomb collision thermalization test.

Reads particle dump files from output/particles.* and checks:
  1. D+ (hot) and C3+ (cold) relax toward common equilibrium temperature
  2. Relaxation matches Spitzer cross-species equipartition timescale
  3. Total momentum is conserved (exact per pair)
  4. Total kinetic energy is approximately conserved

Usage:
    python3 plot_thermalization.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ---------- Physical constants ----------
e_charge = 1.602176634e-19      # C
eps0     = 8.8541878128e-12     # F/m
kB       = 1.380649e-23         # J/K

# ---------- Simulation parameters (must match in.nanbu_thermalize) ----------
# SPARTA species type IDs (order in species command: D+=1, C3+=2)
TYPE_Dp  = 1
TYPE_C3p = 2

m_Dp     = 3.34449e-27          # D+ mass [kg]
m_C3p    = 1.99447e-26          # C3+ mass [kg]
Z_Dp     = 1
Z_C3p    = 3
fnum     = 1.0e7
V_box    = 0.01**3              # 1 cm^3 = 1e-6 m^3
dt       = 1.0e-8              # s
ne_bg    = 1.0e18              # background ne for Coulomb log [m^-3]
Te_bg    = 10.0                # background Te for Coulomb log [eV]
N_Dp     = 5000                # hot D+
N_C3p    = 5000                # cold C3+


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

    single = outpath / "particles"
    if single.exists():
        return parse_dump_file(single)

    files = sorted(outpath.glob("particles.*"),
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
    Cross-species energy equipartition time (NRL Plasma Formulary, SI).

        tau_ab = 3 sqrt(2 pi) (4 pi eps0)^2 ma mb
                 / (8 nb Za^2 Zb^2 e^4 lnL)
                 * (kTa/ma + kTb/mb)^(3/2)

    Returns time [s] for species a to equilibrate with species b.
    """
    kTa = Ta_eV * e_charge
    kTb = Tb_eV * e_charge
    numerator = 3.0 * np.sqrt(2.0 * np.pi) * (4.0 * np.pi * eps0)**2 * ma * mb
    denominator = 8.0 * nb * Za**2 * Zb**2 * e_charge**4 * lnL
    v_term = (kTa / ma + kTb / mb)**1.5
    return numerator / denominator * v_term


# ---------- Main ----------
def main():
    data = load_all_dumps("output")
    steps = sorted(data.keys())
    times    = np.array(steps) * dt
    times_us = times * 1e6

    print(f"Loaded {len(steps)} timesteps: {steps[0]} to {steps[-1]}")
    print(f"Time range: {times_us[0]:.1f} -- {times_us[-1]:.1f} us")

    # Split by species type
    T_Dp_arr  = np.zeros(len(steps))
    T_C3p_arr = np.zeros(len(steps))
    px_arr    = np.zeros(len(steps))
    py_arr    = np.zeros(len(steps))
    pz_arr    = np.zeros(len(steps))
    KE_arr    = np.zeros(len(steps))

    for i, step in enumerate(steps):
        d = data[step]
        types = d["type"]
        vx, vy, vz = d["vx"], d["vy"], d["vz"]

        is_Dp  = types == TYPE_Dp
        is_C3p = types == TYPE_C3p

        T_Dp_arr[i]  = kinetic_temperature(vx[is_Dp],  vy[is_Dp],  vz[is_Dp],  m_Dp)
        T_C3p_arr[i] = kinetic_temperature(vx[is_C3p], vy[is_C3p], vz[is_C3p], m_C3p)

        # Momentum (sum m_i * v_i for each species)
        px_arr[i] = np.sum(vx[is_Dp]) * m_Dp + np.sum(vx[is_C3p]) * m_C3p
        py_arr[i] = np.sum(vy[is_Dp]) * m_Dp + np.sum(vy[is_C3p]) * m_C3p
        pz_arr[i] = np.sum(vz[is_Dp]) * m_Dp + np.sum(vz[is_C3p]) * m_C3p

        # Kinetic energy
        KE_arr[i] = (0.5 * m_Dp  * np.sum(vx[is_Dp]**2  + vy[is_Dp]**2  + vz[is_Dp]**2) +
                     0.5 * m_C3p * np.sum(vx[is_C3p]**2 + vy[is_C3p]**2 + vz[is_C3p]**2))

    # Convert to eV
    T_Dp_eV  = T_Dp_arr  * kB / e_charge
    T_C3p_eV = T_C3p_arr * kB / e_charge

    # Spitzer analytical curve
    n_Dp  = N_Dp  * fnum / V_box
    n_C3p = N_C3p * fnum / V_box
    lnL   = coulomb_log(ne_bg, Te_bg)

    # Characteristic equipartition time at initial conditions
    tau_s = spitzer_cross_species_tau(
        T_Dp_eV[0], T_C3p_eV[0], m_Dp, m_C3p, Z_Dp, Z_C3p, n_C3p, lnL)
    tau_s_us = tau_s * 1e6

    print(f"\n--- Physical parameters ---")
    print(f"n_D+  = {n_Dp:.2e} m^-3")
    print(f"n_C3+ = {n_C3p:.2e} m^-3")
    print(f"Coulomb log ln(Lambda) = {lnL:.2f}")
    print(f"Spitzer equipartition time tau_eq = {tau_s_us:.2f} us")

    print(f"\n--- Initial conditions ---")
    print(f"T_D+  (t=0): {T_Dp_eV[0]:.2f} eV")
    print(f"T_C3+ (t=0): {T_C3p_eV[0]:.2f} eV")

    print(f"\n--- Final state ---")
    print(f"T_D+  (t={times_us[-1]:.0f}us): {T_Dp_eV[-1]:.2f} eV")
    print(f"T_C3+ (t={times_us[-1]:.0f}us): {T_C3p_eV[-1]:.2f} eV")

    # Conservation checks
    KE_rel = (KE_arr - KE_arr[0]) / KE_arr[0]
    p_mag  = np.sqrt(px_arr**2 + py_arr**2 + pz_arr**2)
    p_thermal = np.sqrt(N_Dp * m_Dp * kB * T_Dp_arr[0] +
                        N_C3p * m_C3p * kB * T_C3p_arr[0])
    p_rel = p_mag / p_thermal

    print(f"\n--- Conservation ---")
    print(f"dKE/KE:  max |dKE/KE| = {np.max(np.abs(KE_rel)):.2e},  "
          f"final = {KE_rel[-1]:.2e}")
    print(f"|p|/p_th: max = {np.max(p_rel):.2e},  final = {p_rel[-1]:.2e}")

    # Analytical relaxation (exponential with characteristic tau)
    # Energy conservation: N_Dp*(3/2)kT_Dp + N_C3p*(3/2)kT_C3p = const
    # For equal particle counts: T_eq weighted by energy sharing
    E_total = N_Dp * T_Dp_eV[0] + N_C3p * T_C3p_eV[0]
    T_eq_eV = E_total / (N_Dp + N_C3p)
    DT_Dp  = T_Dp_eV[0]  - T_eq_eV
    DT_C3p = T_C3p_eV[0] - T_eq_eV
    t_fine    = np.linspace(0, times[-1], 500)
    t_fine_us = t_fine * 1e6
    T_Dp_ana  = T_eq_eV + DT_Dp  * np.exp(-t_fine / tau_s)
    T_C3p_ana = T_eq_eV + DT_C3p * np.exp(-t_fine / tau_s)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Temperature relaxation
    ax = axes[0, 0]
    ax.plot(times_us, T_Dp_eV, "r-", lw=1.5, label="D+ (hot)")
    ax.plot(times_us, T_C3p_eV, "b-", lw=1.5, label="C3+ (cold)")
    ax.plot(t_fine_us, T_Dp_ana, "r:", lw=1.5, alpha=0.7,
            label=f"Spitzer (tau={tau_s_us:.1f} us)")
    ax.plot(t_fine_us, T_C3p_ana, "b:", lw=1.5, alpha=0.7)
    ax.axhline(T_eq_eV, color="k", ls="--", lw=0.8, label=f"T_eq = {T_eq_eV:.1f} eV")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Temperature [eV]")
    ax.set_title("Temperature relaxation: D+ vs C3+")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Energy conservation
    ax = axes[0, 1]
    ax.plot(times_us, KE_rel * 100, "g-", lw=1.5)
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("dKE / KE_0 [%]")
    ax.set_title("Energy conservation")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)

    # (c) Momentum conservation
    ax = axes[1, 0]
    ax.plot(times_us, px_arr, label="px")
    ax.plot(times_us, py_arr, label="py")
    ax.plot(times_us, pz_arr, label="pz")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Total momentum [kg m/s]")
    ax.set_title("Momentum conservation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Speed distribution per species: initial vs final
    ax = axes[1, 1]
    d0 = data[steps[0]]
    df = data[steps[-1]]

    is_Dp_0  = d0["type"] == TYPE_Dp
    is_C3p_0 = d0["type"] == TYPE_C3p
    sp_Dp_0  = np.sqrt(d0["vx"][is_Dp_0]**2  + d0["vy"][is_Dp_0]**2  + d0["vz"][is_Dp_0]**2)
    sp_C3p_0 = np.sqrt(d0["vx"][is_C3p_0]**2 + d0["vy"][is_C3p_0]**2 + d0["vz"][is_C3p_0]**2)

    is_Dp_f  = df["type"] == TYPE_Dp
    is_C3p_f = df["type"] == TYPE_C3p
    sp_Dp_f  = np.sqrt(df["vx"][is_Dp_f]**2  + df["vy"][is_Dp_f]**2  + df["vz"][is_Dp_f]**2)
    sp_C3p_f = np.sqrt(df["vx"][is_C3p_f]**2 + df["vy"][is_C3p_f]**2 + df["vz"][is_C3p_f]**2)

    all_speeds = np.concatenate([sp_Dp_0, sp_C3p_0, sp_Dp_f, sp_C3p_f])
    bins = np.linspace(0, np.percentile(all_speeds, 99.5), 60)

    ax.hist(sp_Dp_0, bins=bins, density=True, alpha=0.3, color="r", label="D+ t=0")
    ax.hist(sp_Dp_f, bins=bins, density=True, alpha=0.3, color="darkred",
            label=f"D+ t={times_us[-1]:.0f}us")
    ax.hist(sp_C3p_0, bins=bins, density=True, alpha=0.3, color="b", label="C3+ t=0")
    ax.hist(sp_C3p_f, bins=bins, density=True, alpha=0.3, color="darkblue",
            label=f"C3+ t={times_us[-1]:.0f}us")

#    ax.set_xlabel("Speed [m/s]")
    ax.set_title("Velocity distribution per species")
    ax.set_xlabel(r"velocity (m/s)")
    ax.set_ylabel(r"f(v)")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Nanbu Coulomb collision: D+ / C3+ thermalization", fontsize=14)
    plt.tight_layout()
    plt.savefig("thermalization.png", dpi=150)
    print(f"\nPlot saved to thermalization.png")
    plt.show()


if __name__ == "__main__":
    main()
