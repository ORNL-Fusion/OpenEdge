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
    return t, np.asarray(vz, float), np.asarray(z, float), rd
    
    

def analytic_const_nuE(t, v0, g=-9.81, nuE=1.0, u_g=0.0):
    v_inf = u_g + g/nuE
    return (v0 - v_inf) * np.exp(-nuE * t) + v_inf, v_inf

def piecewise_predictor(t, v, nuE_of_t, g=-9.81, u_g=0.0):
    """Predict v(t) forward one step at a time using local frozen nuE(t_k)."""
    vp = np.zeros_like(v)
    vp[0] = v[0]
    for k in range(len(t)-1):
        dt = t[k+1] - t[k]
        nu = nuE_of_t[k]
        v_inf = u_g + g/nu if nu > 0 else u_g + 0.0
        vp[k+1] = (vp[k] - v_inf)*np.exp(-nu*dt) + v_inf
    # final terminal velocity from last segment
    nu_last = nuE_of_t[-1]
    v_inf_last = u_g + g/nu_last if nu_last > 0 else u_g
    return vp, v_inf_last

def fit_log_slope(t, v, v_inf):
    """Return nuE_fit by linear fit of ln|v - v_inf| vs t (skip values near 0)."""
    y = v - v_inf
    mask = np.isfinite(y) & (np.abs(y) > 1e-10*np.max(np.abs(y)))
    if mask.sum() < 3:
        return np.nan, (np.nan, np.nan)
    T = t[mask]
    Y = np.log(np.abs(y[mask]))
    # Linear fit: Y = a + b*T; b = -nuE
    b, a = np.polyfit(T, Y, 1)
    return -b, (a, b)

def multi_run_collapse(run_specs, out_png):
    """run_specs: list of dicts with keys: csv, v0, g, nuE, u_g (nuE optional if provided in metafile)"""
    plt.figure(figsize=(6,5), dpi=180)
    tt = np.linspace(0, 1.0, 100)
    plt.plot(tt, np.exp(-tt), lw=2, label="exp(-t)")
    for spec in run_specs:
        t, vz, *_ = load_time_series(spec["csv"])
        g = spec.get("g", -9.81)
        u_g = spec.get("u_g", 0.0)
        nuE = spec.get("nuE", None)
        if nuE is None:
            # fit from the data by estimating v_inf at tail
            v_inf = np.median(vz[-max(5, len(vz)//10):])
            nuE_fit, _ = fit_log_slope(t, vz, v_inf)
            nuE = nuE_fit if np.isfinite(nuE_fit) else 1.0
        v_inf = u_g + g/nuE
        ttil = nuE * (t - t[0])
        vtil = (vz - v_inf) / (vz[0] - v_inf + 1e-300)
        plt.plot(ttil, vtil, '.', ms=3, alpha=0.8, label=Path(spec['csv']).stem)
    plt.xlabel(r'$\tilde t = \nu_E (t-t_0)$')
    plt.ylabel(r'$\tilde v = \dfrac{v - v_\infty}{v_0 - v_\infty}$')
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, None)
    plt.grid(True, ls='--', alpha=0.4)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    return out_png

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


def single_run_panels(csv_path, out_png, v0=None, g=-9.81, nuE=None, u_g=0.0,
                      dt=None, Ni=None, Ti_eV=None, rho_d=None,
                      alphaE=1.26, A_background=1.0):
    """
    If nuE is provided -> constant-νE mode (analytic overlay).
    If nuE is None -> variable-νE mode using Epstein νE(t) computed from (Ni, Ti_eV, rho_d, r_d(t)).
    """
    import numpy as np, matplotlib.pyplot as plt

    if dt is None:
        raise ValueError("Please pass dt (seconds) so we can build t from tstep*dt.")

    # load time series + droplet radius
    t, vz, z, rd = load_time_series(csv_path, dt)


    if v0 is None:
        v0 = float(vz[0])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=180)
    ax1, ax2, ax3, ax4 = axes.ravel()
    info = {}

    # ---------------------------
    # CONSTANT νE MODE
    # ---------------------------
    if nuE is not None:
        t_rel = t - t[0]
        vana, v_inf = analytic_const_nuE(t_rel, v0, g=g, nuE=nuE, u_g=u_g)

        ax1.plot(t, vz, '.', ms=3, label='simulation')
        ax1.plot(t, vana, '-', lw=2, label='analytic')
        ax1.set_xlabel('time [s]'); ax1.set_ylabel('$v_z$ [m/s]'); ax1.legend()
        ax1.grid(True, ls='--', alpha=0.4)

        res = vz - vana
        ax2.plot(t, np.abs(res), 'k-', lw=1)
        ax2.set_yscale('log'); ax2.set_xlabel('time [s]')
        ax2.set_ylabel(r'$|v_z - v_{z,\mathrm{analytic}}|$ [m/s]')
        ax2.grid(True, ls='--', alpha=0.4)

        nuE_fit, (a, b) = fit_log_slope(t, vz, v_inf)
        yy = vz - v_inf
        mask = np.isfinite(yy) & (np.abs(yy) > 1e-10*np.max(np.abs(yy)))
        ax3.plot(t[mask], np.log(np.abs(yy[mask])), '.', ms=3)
        ax3.plot(t[mask], (a + b*t[mask]), '-', lw=2,
                 label=rf'slope = {b:.3e}  ⇒  $\nu_E$ = {-b:.3e} s$^{{-1}}$')
        ax3.set_xlabel('time [s]'); ax3.set_ylabel(r'$\ln |v_z - v_\infty|$')
        ax3.legend(); ax3.grid(True, ls='--', alpha=0.4)

        ax4.plot(t, vz - v_inf, '-', lw=1)
        ax4.set_xlabel('time [s]'); ax4.set_ylabel(r'$v_z - v_\infty$ [m/s]')
        ax4.grid(True, ls='--', alpha=0.4)

        info.update({
            "mode": "constant_nuE",
            "nuE_input": float(nuE),
            "nuE_fit": float(-b),
            "v_inf": float(v_inf),
            "max_abs_residual": float(np.max(np.abs(res))),
        })

    # ---------------------------
    # VARIABLE νE(t) MODE (Epstein from dump)
    # ---------------------------
    else:
        for k, name in [(Ni, "Ni"), (Ti_eV, "Ti_eV"), (rho_d, "rho_d")]:
            if k is None:
                raise ValueError(f"variable-νE mode needs {name} supplied.")
        nuE_t = compute_nuE_epstein_series(Ni, Ti_eV, rd, rho_d,
                                           alphaE=alphaE, A_background=A_background)

        vp, v_inf_last = piecewise_predictor(t, vz, nuE_t, g=g, u_g=u_g)

        ax1.plot(t, vz, '.', ms=3, label='simulation')
        ax1.plot(t, vp, '-', lw=2, label='piecewise predictor')
        ax1.set_xlabel('time [s]'); ax1.set_ylabel('$v_z$ [m/s]'); ax1.legend()
        ax1.grid(True, ls='--', alpha=0.4)

        res = vz - vp
        ax2.plot(t, np.abs(res), 'k-', lw=1)
        ax2.set_yscale('log'); ax2.set_xlabel('time [s]')
        ax2.set_ylabel(r'$|v_z - v_{z,\mathrm{pred}}|$ [m/s]')
        ax2.grid(True, ls='--', alpha=0.4)

        ax3.plot(t, nuE_t, '-', lw=2)
        ax3.set_xlabel('time [s]'); ax3.set_ylabel(r'$\nu_E(t)$ [s$^{-1}$]')
        ax3.grid(True, ls='--', alpha=0.4)

        v_inf_t = u_g + g/np.maximum(nuE_t, 1e-300)
        ax4.plot(t, vz - v_inf_t, '-', lw=1)
        ax4.set_xlabel('time [s]'); ax4.set_ylabel(r'$v_z - v_\infty(t)$ [m/s]')
        ax4.grid(True, ls='--', alpha=0.4)

        info.update({
            "mode": "variable_nuE",
            "v_inf_last": float(v_inf_last),
            "nuE_min": float(np.min(nuE_t)),
            "nuE_max": float(np.max(nuE_t)),
            "max_abs_residual": float(np.max(np.abs(res))),
        })

    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    
    return info


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



def analytic_vz_const_nuE(t, v0z, g, nuE, u_g):
    """Analytic vz(t) for constant-νE; returns (vz_analytic, v_inf)."""
    vz_ana, v_inf = analytic_const_nuE(t - t[0], v0z, g=g, nuE=nuE, u_g=u_g)
    return vz_ana, v_inf

def annotate_end_and_asym(ax, t, v0, g, nuE, u_g, vz,
                          box_kwargs=None, fmt="{:.2f}"):
    """
    Annotate a plot with v(t_end), v_infty, and the remaining exp factor.
    v0 is the analytic initial velocity actually used.
    vz is the simulated series (used to show end-of-run median too).
    """
    t0, tend = float(t[0]), float(t[-1])
    v_inf = u_g + g/nuE
    decay = np.exp(-nuE*(tend - t0))
    v_end_ana = (v0 - v_inf)*decay + v_inf
    v_end_sim = float(np.median(vz[-max(5, len(vz)//10):]))

    text = (rf"$v(t_{{end}})$ (sim) = {fmt.format(v_end_sim)} m/s"
            "\n" +
            rf"$v(t_{{end}})$ (ana) = {fmt.format(v_end_ana)} m/s"
            "\n" +
            rf"$v_\infty$ = {fmt.format(v_inf)} m/s")
#            "\n" +
#            rf"$e^{{-\nu_E \Delta t}}$ = {decay:.3f}")

    if box_kwargs is None:
        box_kwargs = dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9)
    ax.text(0.04, 0.06, text, transform=ax.transAxes, fontsize=12, bbox=box_kwargs)


import matplotlib as mpl
import matplotlib.pyplot as plt

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

def annotate_end_and_asym(ax, t, v0, g, nuE, u_g, vz,
                          box_kwargs=None, fmt="{:.2f}", fontsize=13):
    t0, tend = float(t[0]), float(t[-1])
    v_inf = u_g + g/nuE
    decay = np.exp(-nuE*(tend - t0))
    v_end_ana = (v0 - v_inf)*decay + v_inf
    v_end_sim = float(np.median(vz[-max(5, len(vz)//10):]))
    text = (rf"$v(t_{{end}})$ (sim) = {fmt.format(v_end_sim)} m/s" "\n" +
            rf"$v(t_{{end}})$ (ana) = {fmt.format(v_end_ana)} m/s" "\n" +
            rf"$v_\infty$ = {fmt.format(v_inf)} m/s")
    if box_kwargs is None:
        box_kwargs = dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9)
    ax.text(0.04, 0.08, text, transform=ax.transAxes, fontsize=fontsize, bbox=box_kwargs)
#
#def plot_four_cases_panel(cases, out_png):
#    assert len(cases) == 4, "Pass exactly 4 cases for a 2x2 panel."
#    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180, constrained_layout=True)
#    axes = axes.ravel()
#
#    for ax, spec in zip(axes, cases):
#        dump = spec["dump"]; dt = spec["dt"]
#        g    = spec.get("g", -9.81)
#        nuE  = float(spec["nuE"])
#        u_g  = spec.get("u_g", 0.0)
#        lab  = spec.get("label", Path(dump).stem)
#
#        t, vz, z, rd = load_time_series(dump, dt)
#        v0z = spec.get("v0z_override", float(vz[0]))   # <<< use v0z consistently
#
#        vz_ana, v_inf = analytic_vz_const_nuE(t, v0z, g, nuE, u_g)
#
#        ax.plot(t, vz, '.', ms=5, label='simulation')
#        ax.plot(t, vz_ana, '-', lw=2, label='analytic')
#        annotate_end_and_asym(ax, t, v0z, g, nuE, u_g, vz)   # <<< pass v0z
#
#        ax.set_xlabel('time (s)')
#        ax.set_ylabel(r'$v_z$ (m/s)')
#        ax.set_title(lab)
#        ax.grid(True, linestyle='--', alpha=0.6)
#        ax.legend(frameon=True, framealpha=0.9)
#
#    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor='white')
#    plt.show()
#    return out_png

def plot_four_cases_panel(cases, out_png, tag_start="a", tag_fs=16, tag_kwargs=None):
    """Render 2x2 panel; annotate subplots with (a),(b),(c),(d) by default."""
    assert len(cases) == 4, "Pass exactly 4 cases for a 2x2 panel."
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180, constrained_layout=True)
    axes = axes.ravel()

    # panel tags: (a), (b), (c), (d) starting from tag_start
    tags = [f"({chr(ord(tag_start) + i)})" for i in range(len(axes))]
    if tag_kwargs is None:
        tag_kwargs = dict(va="top", ha="left", fontsize=tag_fs) #, fontweight="bold")

    for i, (ax, spec) in enumerate(zip(axes, cases)):
        dump = spec["dump"]; dt = spec["dt"]
        g    = spec.get("g", -9.81)
        nuE  = float(spec["nuE"])
        u_g  = spec.get("u_g", 0.0)
        lab  = spec.get("label", Path(dump).stem)

        t, vz, z, rd = load_time_series(dump, dt)
        v0z = spec.get("v0z_override", float(vz[0]))

        vz_ana, v_inf = analytic_vz_const_nuE(t, v0z, g, nuE, u_g)

        ax.plot(t, vz, '.', ms=5, label='simulation')
        ax.plot(t, vz_ana, '-', lw=2, label='analytic')
        annotate_end_and_asym(ax, t, v0z, g, nuE, u_g, vz)

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$v_z$ (m/s)')
        ax.set_title(lab)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(frameon=True, framealpha=0.9)

        # <-- panel tag in axes coords (top-left, a bit inset)
        ax.text(0.02, 0.98, tags[i], transform=ax.transAxes, **tag_kwargs)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor='white')
    plt.show()
    return out_png


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


#exit()
panel_path = "Figs/validation.png"
cases = [
  dict(dump="case.1", dt=dt, g=-9.81, nuE=nuE_case1, u_g=0.0,
       label="v$\\parallel$g, $u_g=0$, $\\nu_E=0.868$ s$^{-1}$", v0z_override=+5.0),

  dict(dump="case.2", dt=dt, g=-9.81, nuE=nuE_case1, u_g=0.0,
       label="v$\\perp$g, $u_g=0$, $\\nu_E=0.868$ s$^{-1}$", v0z_override=0.0),

  dict(dump="case.3", dt=dt, g=-9.81, nuE=nuE_case1, u_g=1.0,
       label="v$\\parallel$g, $u_g=1$ m/s, $\\nu_E=0.868$ s$^{-1}$", v0z_override=+5.0),

  dict(dump="case.4", dt=dt, g=-9.81, nuE=nuE_case4, u_g=1.0,
       label="v$\\parallel$g,, $u_g=0$, $\\nu_E=1.736$ s$^{-1}$", v0z_override=+0.0),
]

def v_analytic_at_end(t, v0, g, nuE, u_g):
    v_inf = u_g + g/nuE
    return (v0 - v_inf) * np.exp(-nuE * (t[-1] - t[0])) + v_inf

for i, spec in enumerate(cases, 1):
    t, vz, *_ = load_time_series(spec["dump"], spec["dt"])
    v0  = spec.get("v0z_override", float(vz[0]))
    g   = spec.get("g", -9.81); ug = spec.get("u_g", 0.0); nuE = spec["nuE"]
    v_end_ana = v_analytic_at_end(t, v0, g, nuE, ug)
    v_end_sim = float(np.median(vz[-max(5,len(vz)//10):]))
    print(f"T{i}: v_end_sim={v_end_sim:.3f}, v_end_ana={v_end_ana:.3f}  |  v_infty={ug + g/nuE:.3f}")

#exit()
def error_report(t, vz, v0, g, nuE, u_g):
    t_rel = t - t[0]
    v_inf = u_g + g/nuE
    vana  = (v0 - v_inf) * np.exp(-nuE * t_rel) + v_inf
    err   = vz - vana
    rmse  = float(np.sqrt(np.mean(err**2)))
    nrmse = rmse / max(1e-12, np.ptp(vana))
    tail  = max(5, len(vz)//10)
    end_sim = float(np.mean(vz[-tail:]))
    end_ana = float(np.mean(vana[-tail:]))
    return dict(rmse=rmse, nrmse=nrmse, end_sim=end_sim, end_ana=end_ana,
                maxabs=float(np.max(np.abs(err))))
                


# --- when you call it ---
for i, spec in enumerate(cases, 1):
    t, vz, *_ = load_time_series(spec["dump"], spec["dt"])
    v0 = spec["v0z_override"] if "v0z_override" in spec else float(vz[0])
    rep = error_report(t, vz, v0, spec["g"], spec["nuE"], spec.get("u_g", 0.0))
    print(f"RMSE={rep['rmse']:.3e} m/s  NRMSE={rep['nrmse']:.3%}  "
          f"end_sim={rep['end_sim']:.3f}  end_ana={rep['end_ana']:.3f}  "
          f"max|err|={rep['maxabs']:.3e}")
#exit()
plot_four_cases_panel(cases, panel_path)
print("Saved:", panel_path)
