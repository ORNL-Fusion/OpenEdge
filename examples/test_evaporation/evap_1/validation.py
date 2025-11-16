# Create a reusable plotting script for Abdou's droplet validation.
# It supports:
#  - Constant-nu_E tests (gravity + Epstein drag; optional background flow u_g)
#  - Optional piecewise predictor when nu_E varies over time (e.g., evaporation)
#  - Log-slope fit to recover nu_E from data
#  - Multi-run dimensionless collapse (tilde-v vs tilde-t)
#
# Expected input(s):
#  A) Single-run CSV with headers including at least: t or (tstep and dt), vz.
#     Optional columns: vx, vy, z. If 't' is missing but 'tstep' and 'dt' exist,
#     time is built as t = tstep*dt.
#  B) For variable-nu_E runs, a file with 'nuE' over time can be supplied as a
#     CSV with columns t, nuE; we will piecewise-predict using the nearest-sample nuE.
#
# See main() at the bottom for example CLI usage.
#
# This script does not assume any external libs beyond numpy/matplotlib/pandas/scipy.
# (scipy only for linear regression; if unavailable, we fall back to numpy polyfit).
#
# Save path: /mnt/data/epstein_validation_plotter.py

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- physical constants ---
MP      = 1.67262192369e-27         # proton mass [kg]
ECHARGE = 1.602176634e-19           # electron charge [C=J/eV]
PI      = math.pi

def compute_nuE_epstein_series(Ni, Ti_eV, rd_series_m, rho_d,
                               alphaE=1.26, A_background=1.0):
    """
    νE(t) = alphaE * (rho_g * v_th) / (rho_d * r_d(t))
    rho_g = Ni * (A * m_p)
    v_th  = sqrt(8 e Ti / (pi A m_p))
    """
    mi   = A_background * MP
    vth  = np.sqrt(8.0 * (Ti_eV * ECHARGE) / (PI * mi))
    rho_g= Ni * mi
    with np.errstate(divide='ignore', invalid='ignore'):
        nuE_t = alphaE * (rho_g * vth) / (rho_d * np.maximum(rd_series_m, 1e-300))
    return nuE_t


def parse_file(filename):
    timesteps, x_coords, y_coords, z_coords = [], [], [], []
    vx_coords, vy_coords, vz_coords = [], [], []
    mass, temp, radius, ids = [], [], [], []

    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i+1].strip()); i += 2
        elif line == "ITEM: NUMBER OF ATOMS":
            num_atoms = int(lines[i+1].strip()); i += 2
        elif line.startswith("ITEM: ATOMS"):
            # Expect: id type x y z vx vy vz v_pmass temp radius
            for _ in range(num_atoms):
                atom_data = lines[i+1].strip().split()
                # align timestep per row
                timesteps.append(timestep)
                ids.append(int(atom_data[1]))
                x_coords.append(float(atom_data[2]))
                y_coords.append(float(atom_data[3]))
                z_coords.append(float(atom_data[4]))
                vx_coords.append(float(atom_data[5]))
                vy_coords.append(float(atom_data[6]))
                vz_coords.append(float(atom_data[7]))
                mass.append(float(atom_data[8]))
                temp.append(float(atom_data[9]))
                radius.append(float(atom_data[10]))
                i += 1
            i += 1
        else:
            i += 1

    return (np.asarray(timesteps, float),
            np.asarray(x_coords, float),
            np.asarray(y_coords, float),
            np.asarray(z_coords, float),
            np.asarray(vx_coords, float),
            np.asarray(vy_coords, float),
            np.asarray(vz_coords, float),
            np.asarray(mass, float),
            np.asarray(temp, float),
            np.asarray(radius, float),
            np.asarray(ids, int))
            
            

def load_one_id(fname):
        ts,x,y,z,vr,vz_c,vphi,mass,temp,r,ids = parse_file(fname)
        ids = np.asarray(ids); pick = np.unique(ids)[0]
        m = (ids == pick)
        tstep = np.asarray(ts,float)[m]
        z    = np.asarray(z,float)[m]
        vr    = np.asarray(vr,float)[m]
        vz    = np.asarray(vz_c,float)[m]   # this is v_z (cyl)
        vphi  = np.asarray(vphi,float)[m]
        zpos  = np.asarray(z,float)[m]
        m_d   = float(np.asarray(mass,float)[m][0])
        return tstep, z, vr, vz, vphi, zpos, m_d


def load_time_series(path_dump: str, dt: float):
    """
    Returns t[s], vz[m/s], z[m], rd[m] for the FIRST particle id in file.
    """
    tstep, z, vr, vz, vphi, zpos, m_d = load_one_id(path_dump)
    T, X, Y, Z, VX, VY, VZ, MASS, TEMP, RAD, IDS = parse_file(path_dump)
    ids = np.asarray(IDS); pick = np.unique(ids)[0]
    m = (ids == pick)
    rd = np.asarray(RAD, float)[m]
    t  = np.asarray(tstep, float) * float(dt)
    te = np.asarray(TEMP,float)[m]
    return t, np.asarray(vz, float), np.asarray(z, float), rd, te
    
    
import matplotlib as mpl
import matplotlib.pyplot as plt



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from itertools import cycle


def update_droplet_state(R, T, Qs=50.E+06, DT=1.0E-02,
                         AM=1.53E-26, Rho=534.0, Cp=4200., DH=3.158E+03,
                         AN=6.022E+23, Pi=3.141, BK=1.380649E-23):
    """
    Update the state of a Li droplet in plasma for one time step.

    Parameters:
        R (float): Current droplet radius in meters.
        T (float): Current droplet temperature in degrees Celsius.
        Qs (float): Heat flux from plasma to droplet surface in W/m^2.
        DT (float): Integration time step in seconds.
        AM (float): Mass of one Li atom in kg.
        Rho (float): Density of Li in kg/m^3.
        Cp (float): Specific heat of Li in J/kg-K.
        DH (float): Latent heat in J/mol.
        AN (float): Avogadro number.
        Pi (float): Pi number.
        BK (float): Boltzmann constant in J/K.

    Returns:
        R_new (float): Updated droplet radius in meters.
        T_new (float): Updated droplet temperature in degrees Celsius.
        Gevap (float): Evaporation flux in kg/m²/s.
        HF (float): Heat flux entering the droplet in W/m².
    """
    # Convert temperature to Kelvin
    TK = T + 273.15

    # Antoine Equation for vapor pressure
    a1 = 5.055
    b1 = -8023.0
    xm1 = 6.939
    vpres1 = 760.0 * 10.0 ** (a1 + b1 / TK)

    # Evaporation flux
    Gevap = 1.0E+04 * 3.513E+22 * vpres1 / np.sqrt(xm1 * TK)

    # Rate of change of radius
    dRdt = -AM * Gevap / Rho

    # Heat flux
    HF = (Qs - Gevap * DH / AN)

    # Rate of change of temperature
    dTdt = 3.0 / Rho / Cp * HF

    # Update radius and temperature
    R_new = R + dRdt * DT
    T_new = T + dTdt * DT

    return R_new, T_new, Gevap, HF


# --- Typography defaults (consistent everywhere) ---
mpl.rcParams.update({
    "font.size": 14,                 # base size (ticks, legend)
    "axes.titlesize": 14,            # per-subplot titles
    "axes.labelsize": 14,            # x/y labels
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.titlesize": 22,
})


def plot_cases_dual_y_markers(
    cases, out_png=None, title=None, normalize_rd=False, te_units="eV",
    marker_rd="o", marker_te="s", ms_rd=4, ms_te=4
):
    """
    Multi-panel version (vertical layout, twin y-axes)
    Physics unchanged — layout refined for publication.
    """

    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color',
                            ['C0','C1','C2','C3','C4','C5'])
    color_cyc = cycle(default_colors)

    ncases = len(cases)
#    fig, axes = plt.subplots(ncases, 1, figsize=(4.2, 2.5*ncases), dpi=300, sharex=False)
#        fig, axes = plt.subplots(2, len(cases), figsize=(3.5*len(cases), 6), dpi=400, sharex=Tru

    ncases = 3
    fig, axes = plt.subplots(
        ncases, 1,
        figsize=(3.5, 4.2*ncases),   # (width, height)
        dpi=300, sharex=False
    )
#    fig.subplots_adjust(hspace=0.32)

    if ncases == 1:
        axes = [axes]

    for i, spec in enumerate(cases):
        axL = axes[i]
        axR = axL.twinx()

        dump  = spec["dump"]
        dt    = spec["dt"]
        temp0 = spec["temp0"]
        lab   = spec.get("label", Path(dump).stem)
        R0    = float(spec["r_d"])
        T0    = 500.0
        Qs    = 50e6
        DT    = 5.0E-02
        tmax  = float(spec["tmax"])
        color = spec.get("color", next(color_cyc))

        # ----- Analytic ODE -----
        time_steps = int(tmax / DT)
        times_a = np.linspace(0, tmax, time_steps + 1)
        radii_a = np.zeros_like(times_a)
        temps_a = np.zeros_like(times_a)
        R = R0; T = T0
        for n in range(time_steps):
            R, T, Gevap, HF = update_droplet_state(R, T, Qs=Qs, DT=DT)
            radii_a[n + 1] = R
            temps_a[n + 1] = T

        radii_a = np.where(radii_a > 0, radii_a, 0.0)
        radii_a[0] = R0
        y_rd_a = radii_a / R0 if normalize_rd else radii_a
        y_te_a = temps_a + (773.15 - T0)
        y_te_a[0] = float(773.15)

        # ----- Plot analytic (ODE) -----
        axL.plot(times_a, y_rd_a, "-", lw=1.2, color=color, alpha=0.8)
        axR.plot(times_a, y_te_a, "-", lw=1.2, color=color, alpha=0.8)

        # ----- OpenEdge data -----
        t, vz, z, rd, te = load_time_series(dump, dt)
        if "r_d" in spec and (not np.isfinite(rd[0]) or rd[0] == 0.0):
            rd[0] = R0
        te = np.asarray(te, dtype=float)
        te[0] = float(temp0)
        temax = np.nanmax(te[np.isfinite(te)])
        te[(te == 0) | ~np.isfinite(te)] = temax
        y_rd = rd/rd[0] if normalize_rd else rd
        y_te = te

        axL.plot(t[::5], y_rd[::5], linestyle="None", marker=marker_rd, ms=ms_rd, color=color)
        axR.plot(t[::5], y_te[::5], linestyle="None", marker=marker_te, ms=ms_te, mfc="none", color=color)

        # Evaporation lifetime
        evap_idx = np.where(radii_a <= 0)[0]
        if len(evap_idx) > 0:
            tevap = times_a[evap_idx[0]]
            axL.axvline(tevap, color=color, ls='--', alpha=0.4)

        # Formatting
        axL.set_ylabel(r"$r_d/r_{d0}$" if normalize_rd else r"$r_d$ (m)")
        axR.set_ylabel(r"$T_d$ ({})".format(te_units))
        axL.text(0.05, 0.9, f"{lab}", transform=axL.transAxes,
                 fontsize=14, color="black", ha="left", va="top")
        axL.grid(True, linestyle="--", alpha=0.5)

        # Add legend inside the *middle* panel only
        if i == ncases // 2:
            legend_models = [
                Line2D([], [], color="k", lw=1.2, label="analytic"),
                Line2D([], [], color="k", marker="o", linestyle="None", label="simulation")
            ]
            axL.legend(handles=legend_models, loc="lower center",
                       frameon=False, fontsize=14, bbox_to_anchor=(0.45, 1.7))

    # Common x-label on bottom panel
    axes[-1].set_xlabel("time (s)")

    if title:
        fig.suptitle(title, fontsize=14, y=0.99)

    # Reduce gaps between subplots
#    fig.subplots_adjust(hspace=0.1)   smaller value = tighter stacking
    fig.subplots_adjust(hspace=0.1, wspace=0.3)
    if out_png:
        fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig, axes


dt = 1e-5       # your simulation timestep
g  = -9.81

MP = 1.67262192369e-27
ECHARGE = 1.602176634e-19
import numpy as np

def epstein_nuE(Ni, Ti_eV, rd_m, rho_d, alphaE=1.26, A_background=1.0):
    mi  = A_background * MP
    vth = np.sqrt(8.0 * (Ti_eV * ECHARGE) / (np.pi * mi))
    rho_g = Ni * mi
    return alphaE * (rho_g * vth) / (rho_d * rd_m)

nuE_case1 = epstein_nuE(Ni=1.5746e20, Ti_eV=10.0, rd_m=50e-6, rho_d=534.0, alphaE=1.26, A_background=2.0)
nuE_case4 = epstein_nuE(Ni=3.1492e+20, Ti_eV=10.0, rd_m=50e-6, rho_d=534.0, alphaE=1.26, A_background=2.0)
print("nuE =", nuE_case1, "s^-1")

print("nuE_case4 =", nuE_case4, "s^-1")

def error_report(t, vz, v0, g, nuE, u_g):
    v_inf = u_g + g/nuE
    vana = (v0 - v_inf) * np.exp(-nuE * (t - t[0])) + v_inf
    err  = vz - vana
    rmse = np.sqrt(np.mean(err**2))
    nrmse = rmse / max(1e-12, np.ptp(vana))   # normalize by signal span
    tail = max(5, len(vz)//10)
    end_sim = float(np.mean(vz[-tail:]))
    end_ana = float(np.mean(vana[-tail:]))
    return dict(rmse=rmse, nrmse=nrmse, end_sim=end_sim, end_ana=end_ana, maxabs=float(np.max(np.abs(err))))


panel_path = "Figs/oe_ode.png"
cases = [
  dict(dump="case.1", dt=dt, g=-9.81, nuE=nuE_case1, u_g=0.0,
       label="mist", v0z_override=+5.0, r_d=50e-6, temp0=773.15, tmax=3.5),
  dict(dump="case.2", dt=dt, g=-9.81, nuE=nuE_case1, u_g=0.0,
       label="instability", v0z_override=+5.0, r_d=2.5e-3, temp0=773.15, tmax=7),
  dict(dump="case.3", dt=dt, g=-9.81, nuE=nuE_case1, u_g=0.0,
       label="ligament", v0z_override=+5.0, r_d=1e-2, temp0=773.15, tmax=8.5)
]

plot_cases_dual_y_markers(cases, normalize_rd=True,out_png=panel_path,
                          title=None, te_units="°C",
                          marker_rd="o", marker_te="s", ms_rd=4, ms_te=4)
