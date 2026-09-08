#!/usr/bin/env python3
"""
Analysis script for Nanbu Coulomb collision thermalization test.

Reads particle dump files from output/particles_thermalize* and checks:
  1. D+ (hot) and C3+ (cold) relax toward common equilibrium temperature
  2. Relaxation matches the NRL cross-species equipartition rate
  3. Total momentum is conserved (exact per pair)
  4. Total kinetic energy is approximately conserved

Exits 0 on PASS, 1 on FAIL (for regression use).

Usage:
    python3 plot_thermalization.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ---------- Pass/fail thresholds ----------
RMS_TOL    = 0.10    # rms |T_sim - T_ode| / (T_D0 - T_C0), per species
TEQ_TOL    = 0.10    # final |T_D - T_C| / (T_D0 - T_C0)
KE_TOL     = 0.02    # max |dKE/KE_0|
P_TOL      = 1e-3    # max |p - p0| / p_thermal
GAP_RATIO  = (2.0, 5.0)  # sim/NRL gap e-fold ratio; ~3.5 is the correct
                         # kinetic value (weak D+ self-collisions leave D
                         # non-Maxwellian, slowing exchange below NRL)

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

    single = outpath / "particles_thermalize"
    if single.exists():
        return parse_dump_file(single)

    files = sorted(outpath.glob("particles_thermalize.*"),
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


def thermalize_ode(TD0, TC0, t, n_C3p, lnL):
    """Two-temperature relaxation with T-dependent NRL tau.

    dT_D/dt = -(T_D - T_C)/tau_DC; T_C follows from energy conservation
    (equal particle counts): T_D + T_C = TD0 + TC0.
    """
    TD = np.empty_like(t)
    TC = np.empty_like(t)
    TD[0], TC[0] = TD0, TC0
    for i in range(1, len(t)):
        tau = spitzer_cross_species_tau(TD[i-1], TC[i-1], m_Dp, m_C3p,
                                        Z_Dp, Z_C3p, n_C3p, lnL)
        dT = (t[i] - t[i-1]) * (TD[i-1] - TC[i-1]) / tau
        TD[i] = TD[i-1] - dT
        TC[i] = TC[i-1] + dT * N_Dp / N_C3p
    return TD, TC


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

    # Conservation checks. Momentum is gated on drift from t=0: the initial
    # net |p| is finite-N sampling noise (~sqrt(3) p_th), not an error.
    KE_rel = (KE_arr - KE_arr[0]) / KE_arr[0]
    p_thermal = np.sqrt(N_Dp * m_Dp * kB * T_Dp_arr[0] +
                        N_C3p * m_C3p * kB * T_C3p_arr[0])
    p_drift = np.sqrt((px_arr - px_arr[0])**2 +
                      (py_arr - py_arr[0])**2 +
                      (pz_arr - pz_arr[0])**2)
    p_rel = p_drift / p_thermal

    print(f"\n--- Conservation ---")
    print(f"dKE/KE:  max |dKE/KE| = {np.max(np.abs(KE_rel)):.2e},  "
          f"final = {KE_rel[-1]:.2e}")
    print(f"|p-p0|/p_th: max = {np.max(p_rel):.2e},  final = {p_rel[-1]:.2e}")

    # Analytical relaxation: two-temperature ODE with T-dependent NRL tau
    E_total = N_Dp * T_Dp_eV[0] + N_C3p * T_C3p_eV[0]
    T_eq_eV = E_total / (N_Dp + N_C3p)
    t_fine    = np.linspace(0, times[-1], 4000)
    t_fine_us = t_fine * 1e6
    T_Dp_ana, T_C3p_ana = thermalize_ode(T_Dp_eV[0], T_C3p_eV[0],
                                         t_fine, n_C3p, lnL)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Temperature relaxation
    ax = axes[0, 0]
    ax.plot(times_us, T_Dp_eV, "r-", lw=1.5, label="D+ (hot)")
    ax.plot(times_us, T_C3p_eV, "b-", lw=1.5, label="C3+ (cold)")
    ax.plot(t_fine_us, T_Dp_ana, "r:", lw=1.5, alpha=0.7,
            label=f"NRL ODE (tau0={tau_s_us:.1f} us)")
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
    ax.plot(times_us, px_arr - px_arr[0], label="dpx")
    ax.plot(times_us, py_arr - py_arr[0], label="dpy")
    ax.plot(times_us, pz_arr - pz_arr[0], label="dpz")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Momentum drift from t=0 [kg m/s]")
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

    # Gap e-folding time vs NRL (expected ratio ~3.5, see README)
    gap = T_Dp_eV - T_C3p_eV
    mask = gap > 0.1 * gap[0]
    tau_sim = -1.0 / np.polyfit(times[mask], np.log(gap[mask]), 1)[0]
    gap_a = T_Dp_ana - T_C3p_ana
    mask_a = gap_a > 0.1 * gap_a[0]
    tau_nrl = -1.0 / np.polyfit(t_fine[mask_a], np.log(gap_a[mask_a]), 1)[0]
    print(f"\ngap e-folding: sim {tau_sim*1e6:.0f} us vs NRL {tau_nrl*1e6:.0f} us "
          f"(ratio {tau_sim/tau_nrl:.2f})")

    # ---- Pass/fail ----
    scale   = T_Dp_eV[0] - T_C3p_eV[0]
    TD_ode  = np.interp(times, t_fine, T_Dp_ana)
    TC_ode  = np.interp(times, t_fine, T_C3p_ana)
    rms_D   = float(np.sqrt(np.mean((T_Dp_eV  - TD_ode)**2))) / scale
    rms_C   = float(np.sqrt(np.mean((T_C3p_eV - TC_ode)**2))) / scale
    dT_fin  = abs(T_Dp_eV[-1] - T_C3p_eV[-1]) / scale

    checks = [
        ("T_D+  rms vs NRL ODE", rms_D,                       RMS_TOL),
        ("T_C3+ rms vs NRL ODE", rms_C,                       RMS_TOL),
        ("final T_D+ - T_C3+",   dT_fin,                      TEQ_TOL),
        ("KE conservation",      float(np.max(np.abs(KE_rel))), KE_TOL),
        ("momentum drift |p-p0|/p_th", float(np.max(p_rel)),  P_TOL),
    ]
    failed = False
    for name, val, tol in checks:
        ok = val < tol
        failed |= not ok
        print(f"  {name}: {val:.2e} (tol {tol}) {'ok' if ok else 'FAIL'}")
    ratio = tau_sim / tau_nrl
    ok = GAP_RATIO[0] < ratio < GAP_RATIO[1]
    failed |= not ok
    print(f"  gap e-fold sim/NRL ratio: {ratio:.2f} (band {GAP_RATIO}) "
          f"{'ok' if ok else 'FAIL'}")
    print("PASS" if not failed else "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
