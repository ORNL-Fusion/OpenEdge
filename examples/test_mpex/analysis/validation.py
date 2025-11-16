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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from itertools import cycle


def analytic_gravity_epstein(t, v0=0.0, z0=0.0, g=-9.81, nuE=10.0, ug=0.0):
    """
    Analytic solution for droplet motion with gravity and constant Epstein drag.

    v(t) = (v0 - v_inf)*exp(-nuE*t) + v_inf
    z(t) = z0 + (v0 - v_inf)/nuE * (1 - exp(-nuE*t)) + v_inf*t
    v_inf = ug + g/nuE
    """
    v_inf = ug + g / nuE
    v = (v0 - v_inf) * np.exp(-nuE * t) + v_inf
    z = z0 + (v0 - v_inf) / nuE * (1 - np.exp(-nuE * t)) + v_inf * t
    return v, z


import numpy as np

def velocity_variable_nuE(t, nuE_t, v0=0.0, g=-9.81, ug=0.0):
    """
    Exact integral solution of dv/dt = g - nuE(t)*(v - ug)
    evaluated numerically on the time grid t.
    Inputs:
      t      : (N,) time array (strictly increasing)
      nuE_t  : (N,) drag rate at each t
    Returns:
      v      : (N,) velocity on t
    """
    t = np.asarray(t)
    nuE_t = np.asarray(nuE_t)
    # cumulative integral I(t) = ∫ nuE dt (trapezoid)
    I = np.zeros_like(t)
    I[1:] = np.cumsum(0.5*(nuE_t[1:]+nuE_t[:-1])*(t[1:]-t[:-1]))
    # kernel K(t,s) = exp(-(I(t)-I(s))); we can compute the convolution iteratively
    v = np.empty_like(t)
    v[0] = v0
    # iterative quadrature keeps O(N) and is stable
    accum = 0.0
    for n in range(1, len(t)):
        dt = t[n] - t[n-1]
        # average source over the step [t_{n-1}, t_n]
        src_n1 = g + nuE_t[n-1]*ug
        src_n  = g + nuE_t[n]*ug
        # Trapezoid for ∫ e^{-(I(tn)-I(s))} * src(s) ds over last step
        w_n1 = np.exp(-(I[n]-I[n-1]))  # factor for left endpoint
        # over the last step, approximate exp term as varying linearly in I
        step = 0.5*(w_n1*src_n1 + 1.0*src_n)*dt
        accum = np.exp(-(I[n]-I[n-1]))*accum + step
        v[n] = v0*np.exp(-I[n]) + accum
    return v
#

# --- Typography defaults (shared across all paper figures) ---
mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.titlesize": 22,
    "lines.linewidth": 1.2,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

def plot_cases_drag_evap(cases, out_png=None, g=-9.81):
    """
    Validation of OpenEdge droplet velocity under gravity + Epstein drag with evaporation.
    Left: velocity vs analytic variable-νE(t).
    Right: r_d/r_d0 and νE(t) evolution.
    """
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    fig, axes = plt.subplots(2, len(cases), figsize=(3.5*len(cases), 6), dpi=400, sharex=True)
    if len(cases) == 1:
        axes = np.array([axes])

    for i, spec in enumerate(cases):
        dump, dt, tmax, R0 = spec["dump"], spec["dt"], spec["tmax"], spec["r_d"]
        nuE0, color = spec["nuE0"], spec.get("color", next(colors))
        label = spec.get("label", Path(dump).stem)
        te0 = spec.get("temp")

        # --- Load trajectory ---
        t, vz, z, rd, te = load_time_series(dump, dt)
        rd = np.maximum(rd, 1e-9); rd[0] = R0
        nuE_t = nuE0 * (R0 / rd)
        v0 = float(vz[0])
        te[0]= te0

        # --- Analytic variable-νE(t) solutions ---
        v_exact = velocity_variable_nuE(t, nuE_t, v0=v0, g=g)
        v_pred  = np.zeros_like(t); v_pred[0] = v0
        for n in range(1, len(t)):
            dt_loc = t[n] - t[n-1]
            v_inf_local = g / nuE_t[n]
            v_pred[n] = (v_pred[n-1] - v_inf_local) * np.exp(-nuE_t[n]*dt_loc) + v_inf_local

        v_inf = g / nuE0

        # --- Left: velocity + inset residual ---
        axv = axes[i, 0]
        axv.plot(t, v_exact, "--", lw=1.2, color=color, label=r"analytic $\nu_E(t)$")
        axv.plot(t[::5], vz[::5], linestyle="None", marker="o", ms=4.0,
                 mfc=color, mec="white", mew=0.6, label="simulation")
        axv.axhline(v_inf, color=color, ls=":", lw=0.8)

        # Residual inset
        ts = t[::5]
        v_ex = np.interp(ts, t, v_exact)
        res = vz[::5] - v_ex
        ins = axv.inset_axes([0.56, 0.62, 0.38, 0.33])
        ins.axhline(0, color="0.7", lw=0.8)
        ins.plot(ts, res, "-", lw=1.0, color=color)
        ins.set_ylabel("Δv (m/s)", fontsize=12)
        ins.tick_params(axis="both", labelsize=10, direction="in")
        ins.grid(ls="--", alpha=0.4)

#        axv.text(0.05, 0.9, label, transform=axv.transAxes,
#                 fontsize=14, color="black", ha="left", va="top")
        axv.set_ylabel("velocity (m/s)")
        axv.grid(ls="--", alpha=0.3)
        if i == 0:
            axv.legend(frameon=False, loc="lower right")

        # --- Right: radius and νE(t) ---
        axr = axes[i, 1]; axr2 = axr.twinx()
        axr.plot(t, rd/R0, "-", lw=1.2, color=color)
        axr2.plot(t, nuE_t, "--", lw=1.2, color=color)
#        axr2.plot(t, te, "--", lw=1.2, color=color)
        axr.set_ylabel(r"$r_d/r_{d0}$")
        axr2.set_ylabel(r"$\nu_E$ (s$^{-1}$)")
        axr.grid(ls="--", alpha=0.5)

        if i == len(cases)-1:
#            axv.set_xlabel("time (s)")
            axr.set_xlabel("time (s)")

    fig.subplots_adjust(hspace=0.1, wspace=0.3)
    if out_png:
        fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    plt.show()


dt = 1e-5       # your simulation timestep
g  = -9.81

MP = 1.67262192369e-27
ECHARGE = 1.602176634e-19


def epstein_nuE(Ni, Ti_eV, rd_m, rho_d, alphaE=1.26, A_background=1.0):
    mi  = A_background * MP
    vth = np.sqrt(8.0 * (Ti_eV * ECHARGE) / (np.pi * mi))
    rho_g = Ni * mi
    return alphaE * (rho_g * vth) / (rho_d * rd_m)

nuE_case1 = epstein_nuE(Ni=1.5746e20, Ti_eV=10.0, rd_m=50e-6, rho_d=534.0, alphaE=1.26, A_background=2.0)

cases = [
    {
        "dump": "case.1",
        "dt": 1e-5,
        "tmax": 5.0,
        "r_d": 50e-6,
        "v0": 0.0,
        "nuE0": nuE_case1,
        "label": "mist",
        "temp":773.15
    },
]
plot_cases_drag_evap(cases, out_png="Figs/evap_2.png")
