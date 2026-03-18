#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from utils import surface


AMU_KG = 1.66053906660e-27
LI_AMU = 6.94
LI_ATOM_MASS_KG = LI_AMU * AMU_KG


def parse_dump(filepath: Path):
    rows = {k: [] for k in ["timestep", "id", "x", "y"]}
    with filepath.open() as f:
        lines = f.read().splitlines()

    i = 0
    timestep = None
    n_atoms = None
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2
            continue
        if line == "ITEM: NUMBER OF ATOMS":
            n_atoms = int(lines[i + 1].strip())
            i += 2
            continue
        if line.startswith("ITEM: BOX BOUNDS"):
            i += 4
            continue
        if line.startswith("ITEM: ATOMS"):
            headers = lines[i].split()[2:]
            idx = {name: j for j, name in enumerate(headers)}
            if timestep is None or n_atoms is None:
                raise RuntimeError("Incomplete dump block before ATOMS.")
            for k in range(n_atoms):
                vals = lines[i + 1 + k].split()
                rows["timestep"].append(timestep)
                rows["id"].append(int(vals[idx["id"]]))
                rows["x"].append(float(vals[idx["x"]]))
                rows["y"].append(float(vals[idx["y"]]))
            i += 1 + n_atoms
            continue
        i += 1

    for key in rows:
        rows[key] = np.asarray(rows[key])
    return rows


def parse_mass_loss(path: Path):
    blocks = []
    timestep = None
    n_cells = None
    i = 0
    with path.open() as f:
        lines = f.read().splitlines()

    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2
            continue
        if line == "ITEM: NUMBER OF CELLS":
            n_cells = int(lines[i + 1].strip())
            i += 2
            continue
        if line.startswith("ITEM: BOX BOUNDS"):
            i += 4
            continue
        if line.startswith("ITEM: CELLS"):
            headers = lines[i].split()[2:]
            if timestep is None or n_cells is None:
                raise RuntimeError("Incomplete mass_loss block before CELLS.")
            rows = []
            for k in range(n_cells):
                vals = lines[i + 1 + k].split()
                if len(vals) < len(headers):
                    continue
                rows.append([float(vals[j]) for j in range(len(headers))])
            blocks.append({"timestep": timestep, "headers": headers, "data": np.asarray(rows, dtype=float)})
            i += 1 + n_cells
            continue
        i += 1

    if not blocks:
        raise RuntimeError(f"No timestep blocks found in {path}")
    return blocks


def extract_total_li_rate_atoms_per_s(block, dt):
    headers = block["headers"]
    data = block["data"]
    dm_idx = [j for j, name in enumerate(headers) if ("[1]" in name) or ("dm" in name.lower())]
    dn_idx = [j for j, name in enumerate(headers) if ("[2]" in name) or ("dn" in name.lower())]

    if dm_idx:
        dm_kg_step = np.zeros(len(data), dtype=float)
        for j in dm_idx:
            dm_kg_step += data[:, j]
        dm_kg_step = np.where(np.isfinite(dm_kg_step) & (dm_kg_step > 0.0), dm_kg_step, 0.0)
        atoms_step = dm_kg_step / LI_ATOM_MASS_KG
        method = "mass_to_atoms"
    elif dn_idx:
        atoms_step = np.zeros(len(data), dtype=float)
        for j in dn_idx:
            atoms_step += data[:, j]
        atoms_step = np.where(np.isfinite(atoms_step) & (atoms_step > 0.0), atoms_step, 0.0)
        method = "atoms_direct"
    else:
        raise RuntimeError("No recognizable Li loss column found in mass_loss (need [1]/dm or [2]/dn).")

    return float(np.sum(atoms_step) / dt), method


def parse_b2fgmtry(path: Path, nx: int, ny: int):
    nxg, nyg = nx + 2, ny + 2

    with path.open() as fh:
        lines = fh.readlines()

    float_re = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")
    header_idx = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("*cf:")]
    datasets = {}
    for k, i0 in enumerate(header_idx):
        i1 = header_idx[k + 1] if (k + 1) < len(header_idx) else len(lines)
        parts = lines[i0].split()
        try:
            dtype = parts[1].lower()
            count = int(parts[2])
            name = parts[3]
        except (IndexError, ValueError):
            continue

        if dtype not in ("real", "int"):
            continue

        vals = []
        for ln in lines[i0 + 1 : i1]:
            for tok in float_re.findall(ln):
                try:
                    vals.append(float(tok.replace("D", "E").replace("d", "e")))
                except ValueError:
                    continue
        if len(vals) < count:
            raise RuntimeError(
                f"Dataset parse failed for {name} in {path}: needed {count}, got {len(vals)}."
            )
        datasets[name] = np.asarray(vals[:count], dtype=float)

    for req in ["crx", "cry", "vol"]:
        if req not in datasets:
            raise KeyError(f"{req} not found in {path}")

    crx = datasets["crx"].reshape(4, nxg, nyg, order="F")
    cry = datasets["cry"].reshape(4, nxg, nyg, order="F")
    r_s = crx.mean(axis=0).T
    z_s = cry.mean(axis=0).T
    vol = datasets["vol"].reshape(nxg, nyg, order="F").T
    return r_s, z_s, vol


def build_segments(data, ml_ts, ml_values):
    ids = np.unique(data["id"])
    segs = []
    seg_ts = []

    for pid in ids:
        m = data["id"] == pid
        if np.count_nonzero(m) < 2:
            continue
        t = data["timestep"][m]
        x = data["x"][m]
        y = data["y"][m]
        order = np.argsort(t)
        t = t[order]
        x = x[order]
        y = y[order]
        p0 = np.column_stack((x[:-1], y[:-1]))
        p1 = np.column_stack((x[1:], y[1:]))
        segs.append(np.stack((p0, p1), axis=1))
        seg_ts.append(t[:-1])

    if not segs:
        raise RuntimeError("No trajectory segments found in dump file.")

    segs = np.concatenate(segs, axis=0)
    seg_ts = np.concatenate(seg_ts, axis=0)

    idx = np.searchsorted(ml_ts, seg_ts, side="right") - 1
    idx = np.clip(idx, 0, len(ml_ts) - 1)
    seg_values = ml_values[idx]
    return segs, seg_ts, seg_values


def main():
    p = argparse.ArgumentParser(
        description="Panel1: trajectory lines colored by Li evaporation rate. Panel2: SOLPS R-Z grid."
    )
    p.add_argument("--dump", default="case.dump", help="Particle dump file.")
    p.add_argument("--mass-loss", default="mass_loss.txt", help="SPARTA mass loss dump.")
    p.add_argument("--dt", type=float, default=1.0e-5, help="SPARTA timestep [s].")
    p.add_argument(
        "--b2fgmtry",
        default="/Users/42d/ORNL Dropbox/Abdou Diaw/addLi/fnacore=6.00e22_pheat=90.00MW_cont_dt=1e-6_te_up/b2fgmtry",
        help="SOLPS b2fgmtry path.",
    )
    p.add_argument("--nx", type=int, default=170)
    p.add_argument("--ny", type=int, default=36)
    p.add_argument("--xlim", nargs=2, type=float, default=[3.0, 5.0])
    p.add_argument("--ylim", nargs=2, type=float, default=[-4.0, -1.5])
    p.add_argument("--out", default="Figs/solps_coupling_overview.png")
    args = p.parse_args()

    base = Path(__file__).resolve().parent
    dump_path = base / args.dump
    ml_path = base / args.mass_loss
    b2_path = Path(args.b2fgmtry).expanduser()
    wall = surface(str(base / "wall.surf"), "2D")
    core = surface(str(base / "core.surf"), "2D")
    r_wall, z_wall = wall.polygon.exterior.xy
    r_core, z_core = core.polygon.exterior.xy

    data = parse_dump(dump_path)
    blocks = parse_mass_loss(ml_path)
    ml_ts = np.array([int(b["timestep"]) for b in blocks], dtype=int)
    ml_times_s = ml_ts.astype(float) * args.dt
    ml_rates = []
    method = None
    for b in blocks:
        rate, method = extract_total_li_rate_atoms_per_s(b, args.dt)
        ml_rates.append(rate)
    ml_rates = np.asarray(ml_rates, dtype=float)

    # Use cumulative evaporated atoms so the color metric is monotonic in time.
    dt_dump = np.diff(np.r_[ml_times_s[0], ml_times_s])
    dt_dump[0] = 0.0
    ml_cum_atoms = np.cumsum(ml_rates * dt_dump)

    segs, seg_ts, seg_cum_atoms = build_segments(data, ml_ts, ml_cum_atoms)
    seg_time_s = seg_ts.astype(float) * args.dt

    r_s, z_s, _ = parse_b2fgmtry(b2_path, args.nx, args.ny)

    vmin = float(max(np.nanmin(seg_cum_atoms), 0.0))
    vmax = float(np.nanmax(seg_cum_atoms))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    fig, ax0 = plt.subplots(1, 1, figsize=(7.5, 5.4), dpi=220, constrained_layout=True)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segs, cmap="viridis", norm=norm, linewidth=2.1, alpha=0.95)
    lc.set_array(seg_cum_atoms)
    ax0.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax0, pad=0.01)
    cbar.set_label("Cumulative evaporated Li (atoms)")
#    ax0.set_xlim(args.xlim)
#    ax0.set_ylim(args.ylim)
    ax0.plot(r_wall, z_wall, color="k", lw=2.6, zorder=5, label="Wall")
    ax0.plot(r_core, z_core, color="forestgreen", lw=2.6, zorder=5, label="Core")
    ax0.set_xlabel("R (m)")
    ax0.set_ylabel("Z (m)")
    ax0.grid(True, linestyle="--", alpha=0.3)
#    ax0.set_title(f"Full trajectories colored by cumulative Li\nfinal t={seg_time_s.max():.5f} s")
    ax0.legend(loc="lower right")



    out_path = base / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")

    print(f"[Li method] {method}")
    print(f"[rate] first={ml_rates[0]:.3e} atoms/s last={ml_rates[-1]:.3e} atoms/s")
    print(f"[cumulative] final={ml_cum_atoms[-1]:.3e} atoms")
    print(f"[b2fgmtry] {b2_path}")
    print(f"Saved {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
#python3 trajectories.py \
#    --dump case.dump \
#    --mass-loss mass_loss.txt \
#    --b2fgmtry "/Users/42d/ORNL Dropbox/Abdou Diaw/addLi/fnacore=6.00e22_pheat=90.00MW_cont_dt=1e-6_te_up/b2fgmtry" \
#    --out Figs/solps_coupling_overview.png

