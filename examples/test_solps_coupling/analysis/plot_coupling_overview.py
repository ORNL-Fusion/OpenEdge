#!/usr/bin/env python3
"""Plot 4-panel overview of OpenEdge-SOLPS two-way coupling results.

Panels:
  1. Droplet trajectories (colored by mass)
  2-4. SOLPS Li density at first, mid, and last coupling steps (log scale)

Usage:
    python plot_coupling_overview.py [--run-dir PATH] [--out-dir PATH]
                                     [--steps 1 11 22] [--nx 170] [--ny 36]

Defaults:
    --run-dir  ../coupled_test
    --out-dir  ../coupled_test/figures
"""

import argparse
import re
from pathlib import Path

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import LogNorm


# ──────────────────────────────────────────────────────────────────────
# b2fgmtry parser
# ──────────────────────────────────────────────────────────────────────

def parse_b2fgmtry(path, nx, ny):
    nxg, nyg = nx + 2, ny + 2
    num_cells = nxg * nyg
    with open(path) as fh:
        lines = fh.readlines()

    float_re = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")
    header_idx = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("*cf:")]
    datasets = {}
    for k, i0 in enumerate(header_idx):
        i1 = header_idx[k + 1] if (k + 1) < len(header_idx) else len(lines)
        parts = lines[i0].split()
        try:
            dtype_s, count, name = parts[1].lower(), int(parts[2]), parts[3]
        except (IndexError, ValueError):
            continue
        if dtype_s not in ("real", "int"):
            continue
        vals = []
        for ln in lines[i0 + 1: i1]:
            for tok in float_re.findall(ln):
                try:
                    vals.append(float(tok.replace("D", "E").replace("d", "e")))
                except ValueError:
                    continue
        if len(vals) < count:
            continue
        arr = np.asarray(vals[:count], dtype=float)
        if count and count % num_cells == 0:
            arr = arr.reshape((nxg, nyg, count // num_cells), order="F")
        datasets[name] = arr
    return datasets, nxg, nyg


def get_separatrix(datasets, nxg):
    crx, cry = datasets["crx"], datasets["cry"]
    nncut = int(np.ravel(datasets["nncut"])[0])
    topcut = np.ravel(datasets["topcut"]).astype(int)

    if nncut == 2:
        jsep = int(np.min(topcut)) - 1
        iy_sep = jsep + 3
        left2 = int(np.ravel(datasets["leftcut"])[1])
        right2 = int(np.ravel(datasets["rightcut"])[1])
        idx = np.r_[0:left2 + 1, right2 + 1:nxg]
        sep = {"r": crx[idx, iy_sep, 0], "z": cry[idx, iy_sep, 0]}
        iy2 = int(np.max(topcut)) + 1
        sep["r2"] = crx[:, iy2, 0]
        sep["z2"] = cry[:, iy2, 0]
    else:
        jsep = int(topcut[0]) - 1
        iy_sep = jsep + 3
        sep = {"r": crx[:, iy_sep, 0], "z": cry[:, iy_sep, 0], "r2": None}
    return sep


# ──────────────────────────────────────────────────────────────────────
# Wall surface reader
# ──────────────────────────────────────────────────────────────────────

def read_wall(path):
    with open(path) as f:
        raw = f.readlines()
    r, z = [], []
    for line in raw:
        parts = line.split()
        if len(parts) == 3:
            try:
                int(parts[0])
                r.append(float(parts[1]))
                z.append(float(parts[2]))
            except ValueError:
                pass
    r.append(r[0])
    z.append(z[0])
    return r, z


# ──────────────────────────────────────────────────────────────────────
# Trajectory reader
# ──────────────────────────────────────────────────────────────────────

def read_trajectories(run_dir, subsample=10):
    """Read particle positions from step_*/openedge/particles.txt."""
    traj = {}
    for sd in sorted(run_dir.glob("step_*")):
        pf = sd / "openedge" / "particles.txt"
        if not pf.exists():
            continue
        with open(pf) as f:
            pl = f.readlines()
        i = 0
        while i < len(pl):
            if pl[i].strip() == "ITEM: TIMESTEP":
                na = int(pl[i + 3].strip())
                for j in range(na):
                    d = i + 9 + j
                    if d >= len(pl):
                        break
                    p = pl[d].split()
                    if len(p) < 9:
                        continue
                    pid = int(p[0])
                    r, z, mass = float(p[2]), float(p[3]), float(p[8])
                    traj.setdefault(pid, []).append((r, z, mass))
                i = i + 9 + na
            else:
                i += 1

    segs, smass = [], []
    for pid in sorted(traj.keys()):
        pts = traj[pid]
        for k in range(0, len(pts) - 1, subsample):
            r0, z0, m0 = pts[k]
            k1 = min(k + subsample, len(pts) - 1)
            r1, z1, m1 = pts[k1]
            segs.append([(r0, z0), (r1, z1)])
            smass.append(0.5 * (m0 + m1))
    return segs, np.array(smass), len(traj)


# ──────────────────────────────────────────────────────────────────────
# Li density from balance.nc
# ──────────────────────────────────────────────────────────────────────

def get_li_density(bal_path, guard):
    ds = nc.Dataset(str(bal_path))
    na, am = ds.variables["na"][:], ds.variables["am"][:]
    ds.close()
    li_idx = [i for i in range(len(am)) if abs(am[i] - 7) < 0.2]
    li = np.sum(na[li_idx, :, :], axis=0).T
    li[guard] = np.nan
    return li


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "coupled_test")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, nargs=3, default=[1, 11, 22],
                        help="Three coupling step numbers to plot (default: 1 11 22)")
    parser.add_argument("--nx", type=int, default=170)
    parser.add_argument("--ny", type=int, default=36)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.run_dir / "figures"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    nxg, nyg = args.nx + 2, args.ny + 2

    # Geometry
    b2fgmtry_path = args.run_dir / "solps" / "baserun" / "b2fgmtry"
    if not b2fgmtry_path.exists():
        b2fgmtry_path = args.run_dir / "solps" / "v0_1" / "b2fgmtry"
    datasets, nxg, nyg = parse_b2fgmtry(b2fgmtry_path, args.nx, args.ny)
    crx, cry = datasets["crx"], datasets["cry"]
    sep = get_separatrix(datasets, nxg)

    guard = np.zeros((nxg, nyg), dtype=bool)
    guard[0, :] = True; guard[-1, :] = True
    guard[:, 0] = True; guard[:, -1] = True

    # Wall
    wall_path = args.run_dir.parent / "input" / "wall.surf"
    if not wall_path.exists():
        wall_path = args.run_dir / "openedge" / "wall.surf"
    wall_r, wall_z = read_wall(wall_path)

    # Trajectories
    print("Reading trajectories...")
    segs, smass, n_particles = read_trajectories(args.run_dir, subsample=10)
    print(f"  {n_particles} particles, {len(segs)} segments")

    # Li density at requested steps
    step_labels = [f"Step {s}" for s in args.steps]
    li_fields = []
    for s in args.steps:
        bp = args.run_dir / f"step_{s:03d}" / "solps" / "balance.nc"
        li = get_li_density(bp, guard)
        li_fields.append(li)
        pos = li[li > 0]
        print(f"  step_{s:03d}: Li [{pos.min():.2e}, {pos.max():.2e}]")

    # Precompute cell polygons
    verts_list, flat_indices = [], []
    for ix in range(nxg):
        for iy in range(nyg):
            if guard[ix, iy]:
                continue
            verts_list.append(
                np.array([[crx[ix, iy, c], cry[ix, iy, c]] for c in [0, 1, 3, 2]])
            )
            flat_indices.append(ix * nyg + iy)
    flat_indices = np.array(flat_indices)

    # Color range
    all_pos = np.concatenate([li[li > 0].ravel() for li in li_fields])
    vmin_log, vmax_log = 1e12, all_pos.max()
    norm = LogNorm(vmin=vmin_log, vmax=vmax_log)

    x_lim, y_lim = [2.34, 4.0], [-3.8, -2.1]

    # ── Figure ──
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(22, 8))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.08)

    # Panel 1: trajectories
    ax0 = fig.add_subplot(gs[0, 0])
    if segs:
        lc = LineCollection(segs, cmap="turbo", linewidths=0.8)
        lc.set_array(smass)
        ax0.add_collection(lc)
        cb0 = fig.colorbar(lc, ax=ax0, shrink=0.7, pad=0.02)
        cb0.set_label("Droplet mass (kg)", fontsize=12)
        cb0.ax.tick_params(labelsize=10)
    ax0.plot(wall_r, wall_z, "k-", lw=2)
    ax0.plot(sep["r"], sep["z"], "r-", lw=2)
    if sep["r2"] is not None:
        ax0.plot(sep["r2"], sep["z2"], "r-", lw=2)
    ax0.set_xlim(x_lim); ax0.set_ylim(y_lim); ax0.set_aspect("equal")
    ax0.set_xlabel("R (m)", fontsize=14, fontweight="bold")
    ax0.set_ylabel("Z (m)", fontsize=14, fontweight="bold")
    ax0.set_title("Droplet Trajectories", fontsize=16, fontweight="bold")
    ax0.tick_params(labelsize=12)

    # Panels 2-4: Li density
    last_pc = None
    for i in range(3):
        ax = fig.add_subplot(gs[0, i + 1])
        vals_flat = li_fields[i].ravel()
        cell_vals = vals_flat[flat_indices]
        good = np.isfinite(cell_vals) & (cell_vals > 0)
        vg = [verts_list[j] for j in np.where(good)[0]]
        vv = cell_vals[good]
        pc = PolyCollection(vg, cmap="turbo", edgecolors="face",
                            linewidths=0.1, norm=norm)
        pc.set_array(vv)
        ax.add_collection(pc)
        ax.plot(wall_r, wall_z, "k-", lw=2)
        ax.plot(sep["r"], sep["z"], "r-", lw=2)
        if sep["r2"] is not None:
            ax.plot(sep["r2"], sep["z2"], "r-", lw=2)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_aspect("equal")
        ax.set_xlabel("R (m)", fontsize=14, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Z (m)", fontsize=14, fontweight="bold")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"SOLPS-ITER \u2014 {step_labels[i]}", fontsize=16,
                     fontweight="bold")
        ax.tick_params(labelsize=12)
        last_pc = pc

    cax = fig.add_subplot(gs[0, 4])
    cb = fig.colorbar(last_pc, cax=cax)
    cb.set_label(r"$n_{\mathrm{Li,tot}}$ (m$^{-3}$)", fontsize=14)
    cb.ax.tick_params(labelsize=11)

    out_base = "openedge_solps_li_overview"
    fig.savefig(args.out_dir / f"{out_base}.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out_dir / f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out_dir / out_base}.png")


if __name__ == "__main__":
    main()
