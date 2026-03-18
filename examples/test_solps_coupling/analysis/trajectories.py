#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from utils import surface


def mass_from_radius(radius, rho):
    radius = np.asarray(radius, dtype=float)
    mass = np.full(radius.shape, np.nan, dtype=float)
    valid = np.isfinite(radius) & (radius > 0.0)
    mass[valid] = (4.0 / 3.0) * np.pi * rho * radius[valid] ** 3
    return mass


def parse_dump(filepath: Path):
    rows = {k: [] for k in ["timestep", "id", "type", "x", "y", "z", "radius", "pmass"]}
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
                raise RuntimeError("Incomplete dump block before ATOMS")
            for k in range(n_atoms):
                vals = lines[i + 1 + k].split()
                rows["timestep"].append(timestep)
                rows["id"].append(int(vals[idx["id"]]))
                type_idx = idx.get("type")
                type_val = int(vals[type_idx]) if type_idx is not None else -1
                rows["type"].append(type_val)
                rows["x"].append(float(vals[idx["x"]]))
                rows["y"].append(float(vals[idx["y"]]))
                rows["z"].append(float(vals[idx["z"]]))
                radius_idx = idx.get("radius")
                radius_val = float(vals[radius_idx]) if radius_idx is not None else np.nan
                rows["radius"].append(radius_val)
                pmass_idx = idx.get("v_pmass")
                pmass_val = float(vals[pmass_idx]) if pmass_idx is not None else np.nan
                rows["pmass"].append(pmass_val)
            i += 1 + n_atoms
            continue
        i += 1

    for key in rows:
        rows[key] = np.asarray(rows[key])
    return rows


def relative_metric(series, mode="percent_loss"):
    vals = np.asarray(series, dtype=float)
    valid = np.isfinite(vals) & (vals > 0.0)
    if not np.any(valid):
        return np.full(vals.shape, np.nan)
    v0 = vals[np.where(valid)[0][0]]
    if mode == "percent_loss":
        return 100.0 * (1.0 - vals / v0)
    if mode == "ratio":
        return vals / v0
    if mode == "absolute":
        return vals
    raise ValueError("mode must be absolute, ratio, or percent_loss")


def add_traj_colored(ax, R, Z, values, cmap, vmin, vmax):
    if len(R) < 2:
        return
    vals = np.asarray(values, dtype=float)
    valid = np.isfinite(vals)
    if np.count_nonzero(valid) < 2:
        return
    points = np.array([R, Z]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    keep = np.isfinite(vals[:-1]) & np.isfinite(vals[1:])
    if not np.any(keep):
        return
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segs[keep], cmap=cmap, norm=norm, linewidth=2.4, alpha=0.95)
    lc.set_array(vals[:-1][keep])
    ax.add_collection(lc)


def select_mass_series(data, ids, rho, source):
    pmass = np.asarray(data["pmass"], dtype=float)
    radius_mass = mass_from_radius(data["radius"], rho)

    if source == "dump":
        print("Using dumped v_pmass for coloring.")
        return pmass, "mass"
    if source == "radius":
        print(f"Using mass derived from radius with rho={rho:g} kg/m^3 for coloring.")
        return radius_mass, "mass"

    for pid in ids:
        mask = data["id"] == pid
        vals = pmass[mask]
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size >= 2:
            spread = np.nanmax(vals) - np.nanmin(vals)
            if spread > max(1.0e-18, 1.0e-6 * np.nanmax(np.abs(vals))):
                print("Using dumped v_pmass for coloring.")
                return pmass, "mass"

    if np.any(np.isfinite(radius_mass)):
        print(f"Using mass derived from radius with rho={rho:g} kg/m^3 because dumped v_pmass is constant.")
        return radius_mass, "mass"

    print("Using dumped v_pmass for coloring.")
    return pmass, "mass"


def main():
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    parser = argparse.ArgumentParser(description="Plot cleaned droplet trajectories.")
    parser.add_argument("--dump", default="case.dump", help="Dump file (Lammps format).")
    parser.add_argument(
        "--quantity",
        choices=["mass", "radius"],
        default="mass",
        help="Particle property used for coloring.",
    )
    parser.add_argument(
        "--color-mode",
        choices=["percent_loss", "ratio", "absolute"],
        default="percent_loss",
        help="How the selected quantity is transformed for coloring.",
    )
    parser.add_argument(
        "--annotate-types",
        choices=["yes", "no"],
        default="yes",
        help="Annotate representative trajectories by particle type.",
    )
    parser.add_argument("--out", default="Figs/trajs_clean.png", help="Output PNG path.")
    parser.add_argument("--marker-size", type=float, default=0.0, help="Scatter marker size for trajectory points.")
    parser.add_argument("--rho", type=float, default=534.0, help="Droplet density [kg/m^3] used to derive mass from radius.")
    parser.add_argument(
        "--mass-source",
        choices=["auto", "dump", "radius"],
        default="auto",
        help="How mass is obtained when --quantity mass is used.",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    dump_path = base / args.dump
    if not dump_path.exists():
        raise FileNotFoundError(f"Missing dump: {dump_path}")
    data = parse_dump(dump_path)

    wall = surface(str(base / "wall.surf"), "2D")
    core = surface(str(base / "core.surf"), "2D")
    Rwall, Zwall = wall.polygon.exterior.xy
    Rcore, Zcore = core.polygon.exterior.xy

    ids = np.unique(data["id"])
    metrics = []
    mode_used = args.color_mode
    if args.quantity == "mass":
        quantity_values, quantity_used = select_mass_series(data, ids, args.rho, args.mass_source)
    else:
        quantity_values = data["radius"]
        quantity_used = "radius"
    for pid in ids:
        mask = data["id"] == pid
        metric = relative_metric(quantity_values[mask], mode=mode_used)
        finite = metric[np.isfinite(metric)]
        if finite.size:
            metrics.append(finite)
    if not metrics:
        fallback_metric = np.asarray(data["radius"], dtype=float)
        fallback_finite = fallback_metric[np.isfinite(fallback_metric)]
        if fallback_finite.size == 0:
            raise RuntimeError("No valid mass/radius data available for coloring.")
        metrics = [fallback_finite]
        mode_used = "absolute"
        quantity_used = "radius"
        print("Warning: selected quantity/mode has no finite data; falling back to absolute radius.")
    metrics = np.concatenate(metrics)
    vmin = float(np.nanmin(metrics))
    vmax = float(np.nanmax(metrics))
    if mode_used == "percent_loss":
        vmin = 0.0
        vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(6.8, 5.4),
        dpi=400,
        gridspec_kw={"wspace": 0.0015},
    )
    cmap = "viridis"
    type_reps = {}
    type_final_values = {}
    marker_map = {1: "o", 2: "s", 3: "^"}
    for pid in ids:
        mask = data["id"] == pid
        if not np.any(mask):
            continue
        metric = relative_metric(quantity_values[mask], mode=mode_used)
        add_traj_colored(
            ax,
            data["x"][mask],
            data["y"][mask],
            metric,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if args.marker_size > 0:
            pts = np.asarray([data["x"][mask], data["y"][mask]], dtype=float).T
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=args.marker_size**4,
                c="k",
                alpha=0.3,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
        ptype = int(data["type"][mask][0]) if np.any(mask) else -1
        finite_metric = metric[np.isfinite(metric)]
        if ptype > 0 and finite_metric.size > 0:
            type_final_values.setdefault(ptype, []).append(float(finite_metric[-1]))
        if ptype not in type_reps:
            xtraj = data["x"][mask]
            ytraj = data["y"][mask]
            rep_idx = int(np.argmax(xtraj))
            type_reps[ptype] = (float(xtraj[rep_idx]), float(ytraj[rep_idx]))

    ax.plot(Rwall, Zwall, color="k", lw=3.0, label="Wall")
    ax.plot(Rcore, Zcore, color="forestgreen", lw=3.0, label="Core")
    ax.set_xlim(2.3,5)
    ax.set_ylim(-4,-1.5)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    quantity_name = "Mass" if quantity_used == "mass" else "Radius"
    if mode_used == "percent_loss":
        cbar.set_label(f"{quantity_name} loss (%)")
    elif mode_used == "ratio":
        cbar.set_label(f"{quantity_name} / {quantity_name}0")
    else:
        unit = "kg" if quantity_used == "mass" else "m"
        cbar.set_label(f"{quantity_name} ({unit})")

    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.3)
#    ax.set_aspect("equal", adjustable="box")
    handles = [
        Line2D([0], [0], color="k", lw=3.0, label="Wall"),
        Line2D([0], [0], color="forestgreen", lw=3.0, label="Core"),
    ]
    if mode_used == "percent_loss":
        handles.append(Line2D([0], [0], linestyle="None", label="Mass loss (%)"))
    for ptype in sorted(type_reps.keys()):
        if ptype <= 0:
            continue
        type_label = f"Type {ptype}"
        if mode_used == "percent_loss" and ptype in type_final_values:
            vals = np.asarray(type_final_values[ptype], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                type_label = f"Type {ptype} ({np.nanmean(vals):.2f}%)"
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker_map.get(ptype, "D"),
                markersize=7,
                markerfacecolor="white",
                markeredgecolor="k",
                linestyle="None",
                label=type_label,
            )
        )
    ax.legend(handles=handles, loc="lower right")

    if args.annotate_types == "yes":
        offset_map = {1: (6, 6), 2: (8, 6), 3: (4, 8)}
        for ptype in sorted(type_reps.keys()):
            if ptype <= 0:
                continue
            xr, zr = type_reps[ptype]
            ax.scatter(
                [xr],
                [zr],
                marker=marker_map.get(ptype, "D"),
                s=60,
                facecolors="white",
                edgecolors="k",
                linewidths=1.0,
                zorder=6,
            )
            ax.annotate(
                f"type {ptype}",
                (xr, zr),
                xytext=offset_map.get(ptype, (6, 6)),
                textcoords="offset points",
                fontsize=9,
                color="k",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    out_path = base / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
#    exit()
