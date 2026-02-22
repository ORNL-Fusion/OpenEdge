#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

#from __future__ import annotations
from pathlib import Path
import numpy as np

def read_last_grid_block(path: Path) -> np.ndarray:
    blocks = []
    current = []
    in_atoms = False

    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("ITEM: TIMESTEP"):
                if current:
                    blocks.append(np.asarray(current, dtype=float))
                    current = []
                in_atoms = False
                continue
            if line.startswith("ITEM: NUMBER OF CELLS"):
                in_atoms = True
                continue
            if line.startswith("ITEM:"):
                in_atoms = False
                continue
            if not in_atoms or not line:
                continue
            vals = line.split()
            if len(vals) >= 7:
                try:
                    current.append([float(v) for v in vals[:7]])
                except ValueError:
                    continue

    if current:
        blocks.append(np.asarray(current, dtype=float))
    if not blocks:
        raise RuntimeError(f"No grid data blocks found in {path}")
    return blocks[-1]


def read_last_grid_block(path: Path) -> np.ndarray:
    blocks = []
    current = []
    in_data = False

    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("ITEM: TIMESTEP"):
                if current:
                    blocks.append(np.asarray(current, dtype=float))
                    current = []
                in_data = False
                continue

            # Accept both cell dumps and atom dumps
            if line.startswith("ITEM: CELLS") or line.startswith("ITEM: ATOMS"):
                in_data = True
                continue

            if line.startswith("ITEM:"):
                in_data = False
                continue

            if not in_data or not line:
                continue

            vals = line.split()
            # id, xc, yc, plus at least 4 fields = 7 columns minimum
            if len(vals) >= 7:
                try:
                    current.append([float(v) for v in vals[:7]])
                except ValueError:
                    continue

    if current:
        blocks.append(np.asarray(current, dtype=float))
    if not blocks:
        raise RuntimeError(f"No grid data blocks found in {path}")
    return blocks[-1]



def parse_sparta_cells_by_timestep(
    filename: str | Path,
    *,
    start_col: int,
    end_col: int,
    require_complete: bool = True,
) -> dict[int, list[list[float]]]:
    """
    Parse a SPARTA dump grid/cells file with repeated TIMESTEP blocks.

    Returns:
      results[timestep] = [xcs, ycs] + [col(start_col), col(start_col+1), ..., col(end_col)]

    Notes:
      - Column indices are 0-based on the split line.
        For: id xc yc f_fml_I[1] f_fml_I[2] f_fml_O[1] f_fml_O[2]
        indices are: 0  1  2      3          4          5          6
      - If require_complete=True, a timestep block is only stored if we read exactly NUMBER OF CELLS rows.
    """
    filename = Path(filename)

    results: dict[int, list[list[float]]] = {}

    timestep = None
    n_cells = None
    in_cells = False

    xcs: list[float] = []
    ycs: list[float] = []
    cols: list[list[float]] = []

    def reset_block():
        nonlocal xcs, ycs, cols, in_cells
        xcs = []
        ycs = []
        cols = [[] for _ in range(start_col, end_col + 1)]
        in_cells = False

    def finalize_block():
        nonlocal timestep, n_cells, xcs, ycs, cols
        if timestep is None or n_cells is None:
            return
        if require_complete and len(xcs) != n_cells:
            # Skip incomplete/truncated blocks
            return
        results[int(timestep)] = [xcs, ycs] + cols

    reset_block()

    with filename.open("r") as f:
        it = iter(f)
        for raw in it:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("ITEM: TIMESTEP"):
                # close previous
                finalize_block()

                # start new
                timestep_line = next(it).strip()
                timestep = int(timestep_line)
                n_cells = None
                reset_block()
                continue

            if line.startswith("ITEM: NUMBER OF CELLS"):
                n_cells_line = next(it).strip()
                n_cells = int(n_cells_line)
                continue

            if line.startswith("ITEM: CELLS"):
                in_cells = True
                continue

            if line.startswith("ITEM:"):
                in_cells = False
                continue

            if not in_cells:
                continue

            parts = line.split()
            # Need xc,yc and the requested columns
            if len(parts) <= max(2, end_col):
                continue

            try:
                xc = float(parts[1])
                yc = float(parts[2])
                vals = [float(parts[k]) for k in range(start_col, end_col + 1)]
            except ValueError:
                continue

            xcs.append(xc)
            ycs.append(yc)
            for idx, v in enumerate(vals):
                cols[idx].append(v)

            # Optional early stop if we already read n_cells rows
            if n_cells is not None and len(xcs) >= n_cells:
                in_cells = False

    # finalize last block at EOF
    finalize_block()

    if not results:
        raise RuntimeError(f"No timestep blocks parsed from {filename}")

    return results


def main():
    p = argparse.ArgumentParser(description="Plot 2D mass loss/source map from SPARTA mass_loss dump.")
    p.add_argument("--infile", default="mass_loss.txt", help="Input grid dump from dump grid.")
    p.add_argument("--dt", type=float, default=1e-5, help="SPARTA timestep [s].")
    p.add_argument("--nr", type=int, default=100, help="SPARTA grid cells in R.")
    p.add_argument("--nz", type=int, default=100, help="SPARTA grid cells in Z.")
    p.add_argument("--rmin", type=float, default=2.0, help="R min of create_box.")
    p.add_argument("--rmax", type=float, default=6.0, help="R max of create_box.")
    p.add_argument("--zmin", type=float, default=-4.0, help="Z min of create_box.")
    p.add_argument("--zmax", type=float, default=3.8, help="Z max of create_box.")
    p.add_argument("--mode", choices=["dn_step", "source"], default="source",
                   help="dn_step=atoms/cell/step, source=atoms/m^3/s")
    p.add_argument("--out", default="Figs/mass_loss_2d.png", help="Output image path.")
    p.add_argument("--evolution-out", default="Figs/mass_loss_evolution.png",
               help="Output PNG for total loss vs time.")
    p.add_argument("--which", choices=["last", "all", "timestep"], default="last",
                   help="Which 2D snapshot to plot: last, all (makes many files), or a specific timestep.")
    p.add_argument("--timestep", type=int, default=None, help="Timestep to plot when --which=timestep.")


    args = p.parse_args()

    infile = Path(args.infile)

    res = parse_sparta_cells_by_timestep(infile, start_col=3, end_col=6)
    print("timesteps:", sorted(res.keys())[:5], "...", sorted(res.keys())[-5:])

    last_ts = max(res.keys())
    print("Using timestep:", last_ts)

    R, Z, fI1, fI2, fO1, fO2 = res[last_ts]

    # Convert lists -> numpy arrays (float)
    R   = np.asarray(R, dtype=float)
    Z   = np.asarray(Z, dtype=float)
    fI2 = np.asarray(fI2, dtype=float)
    fO2 = np.asarray(fO2, dtype=float)

    dn = fI2 + fO2   # now elementwise


    if args.mode == "dn_step":
        field = dn
        label = "Li loss [atoms/cell/step]"
    else:
        dR = (args.rmax - args.rmin) / args.nr
        dZ = (args.zmax - args.zmin) / args.nz
        vol = 2.0 * np.pi * R * dR * dZ  # tokamak toroidal cell volume
        field = np.where(vol > 0.0, dn / (args.dt * vol), 0.0)
        label = "Li source [atoms/m^3/s]"


    N = int(10/1e-05/20000)
    T0 = 779.757  # K
    r0 = 0.0025*1000    # m
    
    ts_list = sorted(res.keys())
    times_s = np.array(ts_list, dtype=float) * args.dt * N  # physical time in seconds

    total_atoms_per_step = []
    total_atoms_per_s = []

    for ts in ts_list:
        R, Z, fI1, fI2, fO1, fO2 = res[ts]

        # numpy arrays
        fI2 = np.asarray(fI2, dtype=float)
        fO2 = np.asarray(fO2, dtype=float)

        dn_step = fI2  + fO2  # atoms/cell/step (assuming that's what f_fml_* represent)

        # robust sum over finite values
        dn_step = dn_step[np.isfinite(dn_step)]
        total = dn_step.sum()

        total_atoms_per_step.append(total)
        total_atoms_per_s.append(total / args.dt)

    total_atoms_per_step = np.array(total_atoms_per_step)
    total_atoms_per_s = np.array(total_atoms_per_s)
    
    cumulative_atoms = np.trapz(total_atoms_per_s, times_s)
    print("Approx cumulative atoms lost over dumped window:", cumulative_atoms)

   # droplet initial temp and radius 779.757 0.0025
       
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, ax = plt.subplots(
        1, 1,
        figsize=(6, 4),
        dpi=300,
        gridspec_kw={"wspace": 0.0015}   # <--- reduce spacing
    )


    ax.plot(
        times_s,
        total_atoms_per_s,
        "-o",
        color="black",
        lw=2.5,
        markersize=4,
        label=fr"Droplet: $T_0={T0:.1f}\,$K, $r_0={r0:.1f}\,$mm"
    )

    ax.legend()


    eout = Path(args.evolution_out)
    eout.parent.mkdir(parents=True, exist_ok=True)
#    fig.savefig(eout, dpi=300, bbox_inches="tight")

    ax.tick_params(axis="both", labelsize=10, direction="in")
    ax.set_ylabel(r"Droplet loss rate (atoms/s)", fontsize=14)
    ax.set_xlabel(r"time (s)", fontsize=14)
    ax.grid(ls="--", alpha=0.3)

    ax.minorticks_on()
    ax.tick_params(axis="both", which="both",
                   direction="in", top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("Figs/droplet_losses_rate.png",  dpi=400, bbox_inches="tight", facecolor="white")
    plt.show()

#    plt.close(fig)



if __name__ == "__main__":
    main()
