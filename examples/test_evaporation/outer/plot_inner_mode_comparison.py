"""Plot inner-divertor mode overlays and print compact per-mode trajectory/evaporation summaries."""

# Create a reusable plotting script for Abdou's droplet validation.

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import matplotlib.patheffects as pe
from utils import surface


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# --- physical constants ---
MP      = 1.67262192369e-27         # proton mass [kg]
ECHARGE = 1.602176634e-19           # electron charge [C=J/eV]
PI      = math.pi
RHO_LI = 534.0        # kg/m^3 (same as C++)
AM_LI  = 1.53e-26     # kg/atom (microscopic Li mass in your code)



def parse_file(filename):
#    print("filenamen is ", filename)
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
                # ITEM: ATOMS id type x y z vx vy vz v_pmass temp radius
                # column 0 is particle id, column 1 is type.
                ids.append(int(atom_data[0]))
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
            
def atoms_evap(cases, dt, rho=RHO_LI, am_li=AM_LI):
    """
    Loop over a list of cases and attach evaporation diagnostics.

    cases : list of dicts, each with at least {"dump": "filename"}
    dt    : timestep [s]
    """
    results = []
    for case in cases:
        dump = case["dump"]
        try:
            res = total_evap_from_dump(dump, dt, rho=rho, am_li=am_li)
        except ValueError as e:
            # e.g. "Not enough valid timesteps..."
            print(f"Skipping {dump}: {e}")
            continue

        # total_evap_from_dump returns None when file is empty / no atoms
        if res is None:
            print(f"Skipping {dump} (no usable droplet data)")
            continue

        # merge dictionaries: original case info + evap diagnostics
        out = {**case, **res}
        results.append(out)

    return results

import os

def total_evap_from_dump(path_dump, dt, rho=RHO_LI, am_li=AM_LI):
    # skip missing or zero-byte files
    if (not os.path.exists(path_dump)) or (os.path.getsize(path_dump) == 0):
        print(f"  -> {path_dump} is missing or empty, skipping")
        return None

    T, X, Y, Z, VX, VY, VZ, MASS, TEMP, RAD, IDS = parse_file(path_dump)
    
    ids = np.asarray(IDS)
    if ids.size == 0:
        print(f"  -> {path_dump} has no atoms/timesteps, skipping")
        return None

    # pick one droplet id
    pick = np.unique(ids)[0]
    m = (ids == pick)  # select this droplet

    # raw droplet time series
    r      = np.asarray(RAD,  float)[m]
    tsteps = np.asarray(T,    float)[m]

    # geometry: in SPARTA/OpenEdge here X=R, Y=Z, Z=phi (≈0)
    Rd     = np.asarray(X,    float)[m]  # R
    Zd     = np.asarray(Y,    float)[m]  # Z
    # phi   = np.asarray(Z,    float)[m]  # not used for 2D axisym

    # Remove any timesteps where radius is zero or negative (junk)
    valid  = r > 0.0
    r      = r[valid]
    tsteps = tsteps[valid]
    Rpos   = Rd[valid]
    Zpos   = Zd[valid]

    # convert to physical time
    t = tsteps * float(dt)

    if r.size < 2:
        # handled in atoms_evap via try/except
        raise ValueError(f"Not enough valid timesteps for droplet in {path_dump}")

    # --- evaporation diagnostics ---
    r0      = float(np.max(r))
    r_final = float(np.min(r))

    dV = (4.0 / 3.0) * math.pi * (r0**3 - r_final**3)
    dM = max(rho * dV, 0.0)
    N_evap = dM / am_li

    A      = 4.0 * math.pi * r**2
    At_int = np.trapezoid(A, t)
    flux_avg = N_evap / At_int if At_int > 0.0 else 0.0

    # launch position in (R,Z) using corrected mapping
    R_launch = float(Rpos[0])
    Z_launch = float(Zpos[0])

    # geometric penetration along R (same convention as before)
    s = R_launch - Rpos
    pen_geom = float(np.max(s))

    # ablation-weighted penetration
    V = (4.0 / 3.0) * math.pi * r**3
    M = rho * V
    dm = -np.diff(M)
    dm = np.clip(dm, 0.0, None)
    dN = dm / am_li

    s_mid = 0.5 * (s[:-1] + s[1:])
    N_tot = dN.sum()
    if N_tot > 0.0:
        pen_ablate = float((dN * s_mid).sum() / N_tot)
    else:
        pen_ablate = 0.0

    return {
        "N_evap": N_evap,
        "flux_avg": flux_avg,
        "r0": r0,
        "r_final": r_final,
        "t_life": float(t[-1] - t[0]),
        "R_launch": R_launch,      # now true R, not sqrt(R^2+Z^2)
        "Z_launch": Z_launch,      # now from Y-column
        "pen_depth_geom": pen_geom,
        "pen_depth_ablate": pen_ablate,
    }


fname_re = re.compile(
    r"^case\.outer\."
    r"(?P<droplet>[^.]+)\."
    r"site\.(?P<site>\d+)\."
    r"vmag\.(?P<vmag>\d+(?:\.\d+)?)\."
    r"angle\.(?P<angle>\d+(?:\.\d+)?)\."
    r"(?P<tag>.+)$"
)

def site_label(kind: str, i: int) -> str:
    # kind = "I" or "O"
    return rf"$S^{{\mathrm{{{kind}}}}}_{{{i}}}$"

def label_endpoints(ax, R, Z, kind="I", which="both", dx=0.006, dz=0.006):
    """Label only first/last point as S^{kind}_{i} with a white halo."""
    halo = [pe.Stroke(linewidth=2.2, foreground='white'), pe.Normal()]
    N = len(R)
    pick = []
    if which in ("both", "first"):
        pick.append(0)
    if which in ("both", "last"):
        pick.append(N-1)
    for j in pick:
        txt = site_label(kind, j+1)      # j is 0-based, labels are 1-based
        ax.text(R[j]+dx, Z[j]+dz, txt, fontsize=10, weight='bold',
                color='k', zorder=50, clip_on=False, path_effects=halo)

def load_traj_RZ(path_dump):
    """
    Return R(t), Z(t) for the first droplet in a dump file.
    Uses mapping X=R, Y=Z, and keeps only timesteps with r>0.
    """
    T, X, Y, Z, VX, VY, VZ, MASS, TEMP, RAD, IDS = parse_file(path_dump)

    ids = np.asarray(IDS)
    if ids.size == 0:
        print(f"  -> {path_dump} has no atoms/timesteps, skipping trajectory")
        return None, None

    pick = np.unique(ids)[0]
    m = (ids == pick)

    r  = np.asarray(RAD, float)[m]
    Rd = np.asarray(X,   float)[m]  # R
    Zd = np.asarray(Y,   float)[m]  # Z

    valid = r > 0.0
    if not np.any(valid):
        # Trajectory-only runs may not seed radius in all modes.
        # Fallback: keep all timesteps for R,Z trajectory comparison.
        valid = np.ones_like(r, dtype=bool)

    return Rd[valid], Zd[valid]

def subsample_traj(R, Z, n_points=80):
    """
    Return thinner versions of R, Z with at most n_points samples.
    Always keeps first and last points.
    """
    R = np.asarray(R)
    Z = np.asarray(Z)
    n = R.size

    if n <= n_points:
        return R, Z

    # choose indices uniformly from 0..n-1
    idx = np.linspace(0, n - 1, n_points, dtype=int)
    return R[idx], Z[idx]


# --- wall / core geometry ---
wall = surface("wall.surf", "2D")
domain = wall.polygon
core = surface("core.surf", "2D")

rcore, zcore = core.polygon.exterior.xy
Rwall, Zwall = domain.exterior.xy


import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# --- plotting style (set once) ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica"]   # or Arial
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["figure.dpi"] = 200

#
                
# Choose some lifetime contour levels [s]
t_levels = [0.05, 0.2, 0.3]

# Use the same dt as in your SPARTA script
dt = 1.0e-5

t_levels = [0.05, 0.2, 0.3]

# Use the same dt as in your SPARTA script
dt = 1.0e-5

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- style (keep if you like it) ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica"]
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["figure.dpi"] = 200

def trajectories_by_id(path_dump):
    T, X, Y, Z, VX, VY, VZ, MASS, TEMP, RAD, IDS = parse_file(path_dump)
    if IDS.size == 0:
        return {}
    out = {}
    uid = np.unique(IDS)
    for pid in uid:
        m = (IDS == pid) & (RAD > 0.0)
        if not np.any(m):
            # Fallback for modes where radius is not initialized.
            m = (IDS == pid)
            if not np.any(m):
                continue
        t = T[m]
        r = X[m]
        z = Y[m]
        order = np.argsort(t)
        out[int(pid)] = (r[order], z[order])
    return out


mode_files = {
    1: "case.inner.trajcheck.mode.1",
    2: "case.inner.trajcheck.mode.2",
    3: "case.inner.trajcheck.mode.3",
}

missing = [f for f in mode_files.values() if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(
        "Missing mode dump file(s): " + ", ".join(missing) +
        ". Run each mode and save files as case.inner.trajcheck.mode.1/2/3"
    )

traj = {k: trajectories_by_id(v) for k, v in mode_files.items()}
common_ids = sorted(set(traj[1]).intersection(traj[2]).intersection(traj[3]))
if not common_ids:
    raise RuntimeError("No common particle IDs across mode 1/2/3 files.")

print(f"Common particle IDs: {len(common_ids)}")
print("First IDs:", common_ids[:10])

def summarize_mode(path_dump, pid, rho=RHO_LI):
    T, X, Y, Z, VX, VY, VZ, MASS, TEMP, RAD, IDS = parse_file(path_dump)
    m = (IDS == pid)
    if not np.any(m):
        return None
    t = T[m]
    r = RAD[m]
    R = X[m]
    Zp = Y[m]
    o = np.argsort(t)
    t, r, R, Zp = t[o], r[o], R[o], Zp[o]

    # Radius/mass diagnostics from positive-radius samples only.
    rv = r[r > 0.0]
    if rv.size > 0:
        r0 = float(rv[0])
        rf = float(rv[-1])
        m0 = (4.0 / 3.0) * np.pi * rho * r0**3
        mf = (4.0 / 3.0) * np.pi * rho * rf**3
        mloss_pct = 100.0 * (1.0 - mf / m0) if m0 > 0.0 else np.nan
    else:
        r0 = rf = np.nan
        mloss_pct = np.nan

    return {
        "n": int(t.size),
        "r0": r0,
        "rf": rf,
        "mloss_pct": mloss_pct,
        "R_end": float(R[-1]),
        "Z_end": float(Zp[-1]),
    }

# Print compact numeric summary for the first common particle.
pid0 = common_ids[0]
print(f"\nMode summary for particle id={pid0}:")
for mode in [1, 2, 3]:
    s = summarize_mode(mode_files[mode], pid0)
    if s is None:
        print(f"  mode {mode}: missing particle")
        continue
    print(
        f"  mode {mode}: n={s['n']} r0={s['r0']:.6e} rf={s['rf']:.6e} "
        f"mass_loss={s['mloss_pct']:.2f}% end=({s['R_end']:.5f},{s['Z_end']:.5f})"
    )

# Keep plot readable
plot_ids = common_ids[:12]
cmap = plt.get_cmap("tab20", len(plot_ids))
id_color = {pid: cmap(i) for i, pid in enumerate(plot_ids)}

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.plot(Rwall, Zwall, "k-", lw=2.0, label="Wall")
ax.plot(rcore, zcore, color="#2f7d32", lw=2.0, label="Core")

mode_style = {
    1: {"color": "#1f77b4", "label": "Mode 1: Kinematic"},
    2: {"color": "#ff7f0e", "label": "Mode 2: Forces"},
    3: {"color": "#d62728", "label": "Mode 3: Full"},
}


#stats("case.inner.trajcheck.mode.2")
#stats("case.inner.trajcheck.mode.3")
#exit()
for mode in [1, 2, 3]:
    col = mode_style[mode]["color"]
    for i, pid in enumerate(plot_ids):
        r, z = traj[mode][pid]
        rp, zp = subsample_traj(r, z, n_points=200)
        # label only once per mode
        lbl = mode_style[mode]["label"] if i == 0 else None
        ax.plot(rp, zp, "-", lw=1.6, color=col, alpha=0.70, label=lbl)
        ax.plot(rp[::10], zp[::10], "s", ms=2.0, color=col, alpha=0.45)
        ax.plot(rp[0], zp[0], "o", ms=2.5, color=col, alpha=0.8)

ax.set_xlim(2.4,5)
ax.set_ylim(-4,-2)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("R [m]")
ax.set_ylabel("Z [m]")
ax.grid(alpha=0.2, linestyle="--")
ax.legend(loc="upper right", fontsize=9)
fig.suptitle("Inner Trajectory Overlay: Mode Color Comparison", fontsize=12)
fig.tight_layout()
os.makedirs("Figs", exist_ok=True)
fig.savefig("Figs/inner_mode_overlay.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
