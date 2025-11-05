# ------------------------------------------------------------------------------
#  OpenEdge Orbit Test Post-Processor
# ------------------------------------------------------------------------------
#  Author : Abdou Diaw (OpenEdge contributors, 2024)
#  License: GPL-2.0
#
#  Description:
#    This script parses a SPARTA/OpenEdge particle "state" file and
#    reconstructs the trajectory of a charged particle in R–Z coordinates.
#    The data are produced by the Boris pusher test (banana/passing orbit).
#
#    The script:
#      1. Reads 'ITEM:' blocks from the dump/state file
#      2. Extracts (x, y, z, vx, vy, vz)
#      3. Computes cylindrical radius R = sqrt(x² + y²)
#      4. Optionally computes kinetic energy and drift
#      5. Plots R–Z trajectory
#
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
#  Function: parse_file
# ------------------------------------------------------------------------------
#  Reads the SPARTA/OpenEdge dump/state file format:
#
#    ITEM: TIMESTEP
#    0
#    ITEM: NUMBER OF ATOMS
#    1
#    ITEM: ATOMS id type x y z vx vy vz
#    1 1 1.09 0.0 0.0 0.0 0.000627 -0.002059
#
#  Returns arrays of timesteps, positions, and velocities.
# ------------------------------------------------------------------------------
def parse_file(filename):
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

            elif line == "ITEM: ATOMS id type x y z vx vy vz":
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

    return np.array(timesteps), np.array(x), np.array(y), np.array(z), \
           np.array(vx), np.array(vy), np.array(vz)


# ------------------------------------------------------------------------------
#  Main analysis
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Define datasets: filename → timestep (normalized)
    datasets = {
        "state": np.pi / 10.0,   # dt
    }

    runs = []
    for fname, dt in datasets.items():
        t, charge, x, y, z, vx, vy, vz = parse_file(fname)

        # If there’s 0 or 1 sample, skip sorting gymnastics
        if t.size <= 1:
            order = np.arange(t.size)
        else:
            order = np.argsort(t)

        t, charge = t[order], charge[order]
        x, y, z   = x[order], y[order], z[order]
        vx, vy, vz= vx[order], vy[order], vz[order]

        R = x
        # Kinetic energy (normalized, m=1)
        vmag = np.sqrt(vx**2 + vy**2 + vz**2)
        K = 0.5 * vmag**2
        K0 = K[0]
        drift = K / K0 - 1.0

        # Normalized time (gyroperiod = 2π)
        tau = t / (2 * np.pi)

        Nsteps = int(round(2 * np.pi / dt))

        runs.append({
            "fname": fname, "dt": dt, "Nsteps": Nsteps,
            "t": t, "tau": tau, "R": R, "z": z,
            "K": K, "drift": drift
        })

    # ------------------------------------------------------------------------------
    #  Plot trajectory in R–Z plane
    # ------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    for run in runs:
        label = f"{run['fname']} (Δt={run['dt']:.3e})"
        ax.plot(run["R"], run["z"],'ko', lw=1.2, label=label, alpha=0.9)

    ax.set_xlabel("X [normalized]")
    ax.set_ylabel("Z [normalized]")
#    ax.set_title("Particle trajectory in R–Z plane")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.show()

