#!/usr/bin/env python3
"""
Compare rocket-force trajectories for eta = 0.0, 0.5, 1.0.

Reads SPARTA particle dump files and plots:
  1. Trajectories in the (R, Z) plane with wall/core geometry
  2. Radial velocity history v_R(t)
  3. Normalized radius history r/r0

Usage:
    python3 compare_rocket.py
    python3 compare_rocket.py --show
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
except ModuleNotFoundError:
    plt = None


DT = 1.0e-5
ETA_ORDER = [0.0, 0.5, 1.0]
ETA_STYLE = {
    0.0: {"label": r"$\eta = 0$", "color": "steelblue", "linestyle": "-"},
    0.5: {"label": r"$\eta = 0.5$", "color": "darkorange", "linestyle": "--"},
    1.0: {"label": r"$\eta = 1.0$", "color": "crimson", "linestyle": ":"},
}
TRAJ_PAD_R = 0.08
TRAJ_PAD_Z = 0.10
TRAJ_MIN_SPAN_R = 0.55
TRAJ_MIN_SPAN_Z = 0.95


def eta_tag(eta: float) -> str:
    return f"{eta:.1f}".replace(".", "p")


def read_surf(path: Path):
    """Read a SPARTA 2D surface file and return ordered (R, Z) arrays."""
    pts = {}
    edges = []
    mode = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "Points":
            mode = "points"
            continue
        if line == "Lines":
            mode = "lines"
            continue
        parts = line.split()
        if mode == "points" and len(parts) >= 3:
            try:
                pts[int(parts[0])] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue
        elif mode == "lines" and len(parts) >= 3:
            try:
                edges.append((int(parts[1]), int(parts[2])))
            except ValueError:
                continue

    if not edges:
        return np.array([]), np.array([])

    ordered = [edges[0][0], edges[0][1]]
    used = {0}
    for _ in range(len(edges) - 1):
        tail = ordered[-1]
        for idx, (a, b) in enumerate(edges):
            if idx in used:
                continue
            if a == tail:
                ordered.append(b)
                used.add(idx)
                break
            if b == tail:
                ordered.append(a)
                used.add(idx)
                break

    r = np.array([pts[i][0] for i in ordered if i in pts], dtype=float)
    z = np.array([pts[i][1] for i in ordered if i in pts], dtype=float)
    return r, z


def read_particle_dump(path: Path):
    """Read a SPARTA particle dump sequence or single dump file."""
    file_paths = sorted(glob.glob(f"{path}.*"))
    if not file_paths:
        if path.exists():
            file_paths = [str(path)]
        else:
            print(f"  Missing dump: {path}")
            return None

    rows = {k: [] for k in ["time", "x", "y", "vx", "vy", "temp", "radius"]}

    for file_path in file_paths:
        lines = Path(file_path).read_text().splitlines()
        step = None
        natoms = 0
        idx = None
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "ITEM: TIMESTEP":
                step = int(lines[i + 1].strip())
                i += 2
                continue
            if line == "ITEM: NUMBER OF ATOMS":
                natoms = int(lines[i + 1].strip())
                i += 2
                continue
            if line.startswith("ITEM: BOX BOUNDS"):
                i += 4
                continue
            if line.startswith("ITEM: ATOMS") or line.startswith("ITEM: PARTICLES"):
                headers = line.split()[2:]
                idx = {name: j for j, name in enumerate(headers)}
                required = {"x", "y", "vx", "vy", "temp", "radius"}
                if not required.issubset(idx):
                    raise RuntimeError(f"{file_path}: missing required columns {sorted(required - set(idx))}")
                for offset in range(natoms):
                    parts = lines[i + 1 + offset].split()
                    if len(parts) < len(headers):
                        continue
                    rows["time"].append(step * DT)
                    rows["x"].append(float(parts[idx["x"]]))
                    rows["y"].append(float(parts[idx["y"]]))
                    rows["vx"].append(float(parts[idx["vx"]]))
                    rows["vy"].append(float(parts[idx["vy"]]))
                    rows["temp"].append(float(parts[idx["temp"]]))
                    rows["radius"].append(float(parts[idx["radius"]]))
                i += 1 + natoms
                continue
            i += 1

    if not rows["time"]:
        return None

    return {key: np.asarray(values, dtype=float) for key, values in rows.items()}


def radius_ratio(radius):
    values = np.asarray(radius, dtype=float)
    ratio = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0.0)
    if not np.any(valid):
        return ratio, np.nan
    r0 = values[np.where(valid)[0][0]]
    ratio[valid] = values[valid] / r0
    return ratio, r0


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 2.4,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def style_axis(ax, grid=True):
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.28)


def plot_geometry(ax, wall_path: Path):
    if wall_path.exists():
        rw, zw = read_surf(wall_path)
        if rw.size:
            ax.plot(rw, zw, color="black", lw=2.2, zorder=1)


def set_local_traj_limits(ax, data: dict[float, dict[str, np.ndarray]], wall_path: Path):
    """Zoom the trajectory panel to the local launch/divertor neighborhood."""
    r_values = [run["x"] for run in data.values() if len(run["x"])]
    z_values = [run["y"] for run in data.values() if len(run["y"])]
    if not r_values or not z_values:
        return

    r_all = np.concatenate(r_values)
    z_all = np.concatenate(z_values)
    rmin = float(np.min(r_all))
    rmax = float(np.max(r_all))
    zmin = float(np.min(z_all))
    zmax = float(np.max(z_all))

    # Pull in only the nearby wall segment so the local geometry stays visible.
    if wall_path.exists():
        rw, zw = read_surf(wall_path)
        if rw.size:
            mask = (
                (rw >= rmin - 0.40)
                & (rw <= rmax + 0.40)
                & (zw >= zmin - 0.40)
                & (zw <= zmax + 0.40)
            )
            if np.any(mask):
                rmin = min(rmin, float(np.min(rw[mask])))
                rmax = max(rmax, float(np.max(rw[mask])))
                zmin = min(zmin, float(np.min(zw[mask])))
                zmax = max(zmax, float(np.max(zw[mask])))

    r_span = max(rmax - rmin, TRAJ_MIN_SPAN_R)
    z_span = max(zmax - zmin, TRAJ_MIN_SPAN_Z)
    r_mid = 0.5 * (rmin + rmax)
    z_mid = 0.5 * (zmin + zmax)

    ax.set_xlim(r_mid - 0.5 * r_span - TRAJ_PAD_R, r_mid + 0.5 * r_span + TRAJ_PAD_R)
    ax.set_ylim(z_mid - 0.5 * z_span - TRAJ_PAD_Z, z_mid + 0.5 * z_span + TRAJ_PAD_Z)


def build_figure(base: Path, data: dict[float, dict[str, np.ndarray]]):
    configure_plot_style()

    fig = plt.figure(figsize=(12.6, 4.5), dpi=300)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], wspace=0.2)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_vr = fig.add_subplot(gs[0, 1])
    ax_rr = fig.add_subplot(gs[0, 2])

    for ax in (ax_traj, ax_vr, ax_rr):
        ax.set_box_aspect(1.0)

    wall_path = base / "wall.surf"
    plot_geometry(ax_traj, wall_path)

    eta_handles = []
    first_eta = next(iter(data))
    start_xy = (data[first_eta]["x"][0], data[first_eta]["y"][0])

    all_rr = []
    for eta in ETA_ORDER:
        if eta not in data:
            continue
        run = data[eta]
        style = ETA_STYLE[eta]
        time_ms = 1.0e3 * run["time"]
        rr, _ = radius_ratio(run["radius"])
        finite_rr = rr[np.isfinite(rr)]
        if finite_rr.size:
            all_rr.append(finite_rr)

        ax_traj.plot(
            run["x"],
            run["y"],
            color=style["color"],
            linestyle=style["linestyle"],
            zorder=3,
        )
        ax_traj.scatter(
            run["x"][-1],
            run["y"][-1],
            s=38,
            facecolors="white",
            edgecolors=style["color"],
            linewidths=1.5,
            zorder=4,
        )
        ax_vr.plot(time_ms, run["vx"], color=style["color"], linestyle=style["linestyle"])
        ax_rr.plot(time_ms, rr, color=style["color"], linestyle=style["linestyle"])
        eta_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.4,
                label=style["label"],
            )
        )

    ax_traj.scatter(
        start_xy[0],
        start_xy[1],
        s=36,
        color="black",
        linewidths=0,
        zorder=5,
    )

    geometry_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=6, label="Launch"),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="None",
            markersize=6,
            label="Final",
        ),
    ]
    geom_legend = ax_traj.legend(handles=geometry_handles, loc="upper right", frameon=True)
    ax_traj.add_artist(geom_legend)
    ax_traj.legend(handles=eta_handles, loc="lower right", frameon=False)

#    ax_traj.set_title("Trajectory map")
    ax_traj.set_xlabel("R (m)")
    ax_traj.set_ylabel("Z (m)")
    ax_traj.set_aspect("equal", adjustable="datalim")
    set_local_traj_limits(ax_traj, data, wall_path)
    style_axis(ax_traj, grid=True)

#    ax_vr.set_title(r"Radial velocity")
    ax_vr.set_xlabel("Time (ms)")
    ax_vr.set_ylabel(r"$v_R$ (m/s)")
    ax_vr.axhline(0.0, color="0.65", lw=1.0, linestyle=":")
    style_axis(ax_vr, grid=True)

#    ax_rr.set_title(r"Evaporation history")
    ax_rr.set_xlabel("Time (ms)")
    ax_rr.set_ylabel(r"$r / r_0$", labelpad=3.0)
    ax_rr.yaxis.set_label_coords(-0.14, 0.5)
    style_axis(ax_rr, grid=True)
    if all_rr:
        merged = np.concatenate(all_rr)
        ymin = float(np.nanmin(merged))
        ymax = float(np.nanmax(merged))
        span = max(ymax - ymin, 0.002)
        ax_rr.set_ylim(ymin - 0.08 * span, min(1.002, ymax + 0.08 * span))

    for panel, ax in zip(["(a)", "(b)", "(c)"], [ax_traj, ax_vr, ax_rr]):
        ax.text(0.03, 0.97, panel, transform=ax.transAxes, ha="left", va="top", fontweight="bold")

    fig.subplots_adjust(left=0.055, right=0.992, bottom=0.14, top=0.96, wspace=0.14)
    return fig


def print_summary(data: dict[float, dict[str, np.ndarray]]):
    print("\n--- Summary ---")
    common_t_end = min(run["time"][-1] for run in data.values())
    print(f"  common_t_end = {common_t_end:.4f} s")
    for eta in ETA_ORDER:
        if eta not in data:
            continue
        run = data[eta]
        idx = np.searchsorted(run["time"], common_t_end, side="right") - 1
        idx = max(idx, 0)
        rr, r0 = radius_ratio(run["radius"])
        delta_r = run["x"][idx] - run["x"][0]
        vr_common = run["vx"][idx]
        vr_min = float(np.min(run["vx"]))
        rr_common = rr[idx] if np.isfinite(rr[idx]) else np.nan
        print(
            f"  eta={eta:.1f}: deltaR = {delta_r:.4f} m, "
            f"vR(t_common) = {vr_common:.4f} m/s, "
            f"vR_min = {vr_min:.4f} m/s, "
            f"r_final/r0 = {rr_common:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Display the figure after saving.")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    data = {}
    for eta in ETA_ORDER:
        dump_path = base / f"traj_eta_{eta_tag(eta)}"
        run = read_particle_dump(dump_path)
        if run is None:
            print(f"  eta={eta:.1f}: NO DATA")
            continue
        data[eta] = run
        print(
            f"  eta={eta:.1f}: {len(run['time'])} frames, "
            f"t=[{run['time'].min():.3f}, {run['time'].max():.3f}] s"
        )

    if not data:
        raise SystemExit("No trajectory data found. Run the eta cases first.")

    if plt is None:
        print("\nmatplotlib is not available; skipping figure generation.")
        print_summary(data)
        return

    fig = build_figure(base, data)
    for suffix in ("png", "pdf"):
        out_path = base / f"rocket_force_comparison.{suffix}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print("\nSaved rocket_force_comparison.png and rocket_force_comparison.pdf")

    print_summary(data)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
