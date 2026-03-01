#!/usr/bin/env python3
"""Compare Boris vs GCA trajectories in R-Z plane."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os


def read_sparta_dump(path):
    """Read SPARTA particle dump file, return dict of arrays."""
    timesteps, x, y, z, vx, vy, vz = [], [], [], [], [], [], []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" in line:
                ts = int(f.readline().strip())
                f.readline()  # NUMBER OF ATOMS
                natoms = int(f.readline().strip())
                f.readline()  # ITEM: BOX BOUNDS ...
                for _ in range(3):  # x/y/z bounds lines
                    f.readline()
                f.readline()  # ITEM: ATOMS header
                if natoms == 0:
                    continue
                # Read first (only) particle
                dline = f.readline()
                if not dline or not dline.strip():
                    continue
                parts = dline.split()
                try:
                    timesteps.append(ts)
                    x.append(float(parts[2]))
                    y.append(float(parts[3]))
                    z.append(float(parts[4]))
                    vx.append(float(parts[5]))
                    vy.append(float(parts[6]))
                    vz.append(float(parts[7]))
                except (ValueError, IndexError):
                    timesteps.pop() if len(timesteps) > len(x) else None
                    continue
                # Skip remaining atoms
                for _ in range(natoms - 1):
                    f.readline()
    return {
        "t": np.array(timesteps),
        "x": np.array(x), "y": np.array(y), "z": np.array(z),
        "vx": np.array(vx), "vy": np.array(vy), "vz": np.array(vz),
    }


def fmt_range(arr):
    """Format [min, max] for non-empty arrays and mark empty arrays clearly."""
    if arr.size == 0:
        return "[empty]"
    return f"[{arr.min():.4f}, {arr.max():.4f}]"


def main():
    outdir = os.path.join(os.path.dirname(__file__), "output")
    boris_file = os.path.join(outdir, "traj.boris")
    gca_file = os.path.join(outdir, "traj.gca")

    boris = read_sparta_dump(boris_file)
    gca = read_sparta_dump(gca_file)

    # R = sqrt(x^2 + y^2), Z = z
    R_boris = np.sqrt(boris["x"]**2 + boris["y"]**2)
    Z_boris = boris["z"]
    R_gca = np.sqrt(gca["x"]**2 + gca["y"]**2)
    Z_gca = gca["z"]

    # Speed
    v_boris = np.sqrt(boris["vx"]**2 + boris["vy"]**2 + boris["vz"]**2)
    v_gca = np.sqrt(gca["vx"]**2 + gca["vy"]**2 + gca["vz"]**2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- R-Z trajectory ---
    ax = axes[0, 0]
    ax.plot(R_boris, Z_boris, "b-", lw=0.5, alpha=0.7, label="Boris (500 sub)")
    ax.plot(R_gca, Z_gca, "r--", lw=1.0, alpha=0.8, label="GCA hybrid")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("R-Z Trajectory")
    ax.legend()
    ax.set_aspect("equal")

    # --- R vs time ---
    ax = axes[0, 1]
    dt = 1e-9  # from input
    t_boris = boris["t"] * dt * 1e6  # microseconds
    t_gca = gca["t"] * dt * 1e6
    ax.plot(t_boris, R_boris, "b-", lw=0.5, alpha=0.7, label="Boris")
    ax.plot(t_gca, R_gca, "r--", lw=1.0, alpha=0.8, label="GCA")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("R [m]")
    ax.set_title("R vs Time")
    ax.legend()

    # --- Z vs time ---
    ax = axes[1, 0]
    ax.plot(t_boris, Z_boris, "b-", lw=0.5, alpha=0.7, label="Boris")
    ax.plot(t_gca, Z_gca, "r--", lw=1.0, alpha=0.8, label="GCA")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Z vs Time")
    ax.legend()

    # --- Speed vs time ---
    ax = axes[1, 1]
    ax.plot(t_boris, v_boris / 1e3, "b-", lw=0.5, alpha=0.7, label="Boris")
    ax.plot(t_gca, v_gca / 1e3, "r--", lw=1.0, alpha=0.8, label="GCA")
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("|v| [km/s]")
    ax.set_title("Speed vs Time")
    ax.legend()

    fig.suptitle("Boris vs GCA Hybrid Pusher Comparison", fontsize=14)
    fig.tight_layout()

    out_png = os.path.join(outdir, "boris_vs_gca.png")
    fig.savefig(out_png, dpi=180)
    print(f"Wrote {out_png}")
    print(f"  Boris: {len(boris['t'])} frames, R range {fmt_range(R_boris)}")
    print(f"  GCA:   {len(gca['t'])} frames, R range {fmt_range(R_gca)}")


if __name__ == "__main__":
    main()

