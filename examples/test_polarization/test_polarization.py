# Copyright 2025, OpenEdge contributors
# Authors: Abdourahmane Diaw
"""
Polarization-drift test: compare Boris pusher (OpenEdge) vs guiding-center theory
for a single particle in uniform Bz and sinusoidal Ex(t):

    B = B0 * zhat
    Ex(t) = E0 * sin(omega * t)

Theory (guiding center, gyro-averaged):
    v_E(t)   = - Ex(t) / B0
    v_pol(t) = - (m/q) * (1 / B0**2) * dEx/dt
    v_y(t)   = v_E(t) + v_pol(t)

Usage (from test_polarization directory):
    python3 test_polarization.py
"""

import math
import subprocess as sp
import time
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("Missing dependency: numpy. Install with `python3 -m pip install numpy`.", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Missing dependency: matplotlib. Install with `python3 -m pip install matplotlib`.", file=sys.stderr)
    sys.exit(1)

# Number of MPI cores
MPI_cores = '8'

outdir = Path("figs_polarization")
outdir.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")  # nice for papers

def run_openedge_simulation(openedge_lmp, input_file):
    """
    Run a simulation with a particle going through a uniform magnetic field using OpenEdge.

    Parameters:
    ----------
    openedge_lmp: str
        Path to the OpenEdge executable.
    input_file: str
        Path to the input file for OpenEdge.

    """
    args = [openedge_lmp, '-in', input_file]
    ret = sp.run(args, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"OpenEdge run failed with exit code {ret.returncode}")
    time.sleep(1)


# ------------------------------------------------------------------------------
def parse_file(filename):
    """
    Parse SPARTA/OpenEdge dump file of the form:

    ITEM: TIMESTEP
    <timestep>
    ITEM: NUMBER OF ATOMS
    <N>
    ...
    ITEM: ATOMS id type x y z vx vy vz
    <id> <type> x y z vx vy vz
    ...
    """
    timesteps, x, y, z, vx, vy, vz = [], [], [], [], [], [], []

    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == "ITEM: TIMESTEP":
                timestep = int(lines[i + 1].strip())
                i += 2

            elif line == "ITEM: NUMBER OF ATOMS":
                num_atoms = int(lines[i + 1].strip())
                i += 2

            elif line.startswith("ITEM: ATOMS"):
                # assumes ordering: id type x y z vx vy vz
                timesteps.append(timestep)
                for j in range(num_atoms):
                    atom = lines[i + 1 + j].split()
                    x.append(float(atom[2]))
                    y.append(float(atom[3]))
                    z.append(float(atom[4]))
                    vx.append(float(atom[5]))
                    vy.append(float(atom[6]))
                    vz.append(float(atom[7]))
                i += num_atoms + 1
            else:
                i += 1

    return (np.array(timesteps, dtype=int),
            np.array(x), np.array(y), np.array(z),
            np.array(vx), np.array(vy), np.array(vz))


# ------------------------------------------------------------------------------
def moving_average(a, window):
    """Simple 1D moving average for gyro-averaging."""
    if window <= 1:
        return a
    kernel = np.ones(window) / window
    return np.convolve(a, kernel, mode="same")


# ------------------------------------------------------------------------------
#  Main analysis
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Run OpenEdge simulation
    openedge_lmp = '../../src/spa_mpi'
    input_file = 'in.input'
    if Path(openedge_lmp).exists():
        run_openedge_simulation(openedge_lmp, input_file)
    else:
        print(f"Skipping run: executable not found at {openedge_lmp}")
    

    # 1. Simulation parameters (must match in.input, SI units)
    qe = 1.602176634e-19
    mi = 1.67262192369e-27
    B0 = 1.0
    E0 = 50.0
    omegac = qe * B0 / mi
    omega = 0.05 * omegac
    dt_sim = 0.02 / omegac
    dump_stride_steps = None  # we'll infer it from the dump
    m_over_q = mi / qe

    dumpfile = "state"

    # 2. Parse dump
    tstep, x, y, z, vx, vy, vz = parse_file(dumpfile)

    # infer dump stride from the first two timesteps
    if len(tstep) > 1:
        dump_stride_steps = tstep[1] - tstep[0]
    else:
        dump_stride_steps = 1  # fallback

    dt_out = dt_sim * dump_stride_steps    # time between successive records

    # physical time for each record
    t = tstep * dt_sim
   
    # 3. Theory as before
    Ex    = E0 * np.sin(omega * t)
    dEx_dt = E0 * omega * np.cos(omega * t)
    v_E   = -Ex / B0
    v_pol = -m_over_q * dEx_dt / (B0**2)
    v_theory = v_E + v_pol

    # 4. Gyro-averaging
    Omega_c = B0 / m_over_q
    Tgyro   = 2.0 * math.pi / Omega_c

    # window length in *output points*
    Ngyro   = max(1, int(round(Tgyro / dt_out)))

    vy_smooth   = moving_average(vy, Ngyro)
    vth_smooth  = moving_average(v_theory, Ngyro)
    print(f"Using gyro-averaging window Ngyro = {Ngyro} outputs")

    # 5. Guiding-center displacement: use dt_out, not dt_sim
    y_gc_theory = np.cumsum(v_theory) * dt_out


    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })

    # (a) v_y(t): raw vs theory
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300, constrained_layout=False)
    ax.plot(t, vy, label="simulation $v_y$ (raw)", alpha=0.4)
    ax.plot(t, v_theory, label="theory $v_y$ (guiding center)", lw=1.5)
    ax.set_xlabel(r"$\omega t$")
    ax.set_ylabel("$v_y$")
    ax.set_title("Polarization test: $v_y(t)$, simulation vs theory")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "a_vy_raw_vs_theory")


    # (b) v_y(t): gyro-averaged vs theory
    fig2, ax2 = plt.subplots(figsize=(7, 4), dpi=300, constrained_layout=False)
    ax2.plot(t, vy_smooth, label="simulation $v_y$", lw=1.5)
    ax2.plot(t, vth_smooth, label="theory $v_y$", lw=1.5, ls="--")
    ax2.set_xlabel(r"$\omega t$")
    ax2.set_ylabel("$\\langle v_y \\rangle_{\\rm gyro}$")
    ax2.set_title("Polarization test: gyro-averaged $v_y(t)$")
    ax2.minorticks_on()
    ax2.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(frameon=False)
    fig2.tight_layout()
    save(fig2, "b_vy_gyroavg_vs_theory")


    # (c) y(t) displacement vs integrated theory
    fig3, ax3 = plt.subplots(figsize=(7, 4), dpi=300, constrained_layout=False)
    ax3.plot(t, y, label="simulation y(t)", alpha=0.6)
    ax3.plot(t, y_gc_theory, label="theory $y_{gc}(t)$", lw=1.5)
    ax3.set_xlabel(r"$\omega t$")
    ax3.set_ylabel("y")
    ax3.set_title("Guiding-center displacement in y")
    ax3.minorticks_on()
    ax3.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend(frameon=False)
    fig3.tight_layout()
    save(fig3, "c_y_displacement_vs_theory")

    plt.show()
