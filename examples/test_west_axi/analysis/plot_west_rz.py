#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np


def parse_dump(path: Path):
    lines = path.read_text().splitlines()
    i = 0
    tracks = defaultdict(list)
    bounds = None

    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        step = int(lines[i + 1].strip())
        i += 2

        if i >= len(lines) or not lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            raise RuntimeError("Bad dump format: missing NUMBER OF ATOMS")
        nat = int(lines[i + 1].strip())
        i += 2

        if i >= len(lines) or not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError("Bad dump format: missing BOX BOUNDS")
        i += 1

        local_bounds = []
        while i < len(lines) and not lines[i].startswith("ITEM: ATOMS"):
            parts = lines[i].split()
            if len(parts) >= 2:
                try:
                    local_bounds.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
            i += 1
        if bounds is None and len(local_bounds) >= 2:
            bounds = (
                local_bounds[0][0],
                local_bounds[0][1],
                local_bounds[1][0],
                local_bounds[1][1],
            )

        if i >= len(lines) or not lines[i].startswith("ITEM: ATOMS"):
            raise RuntimeError("Bad dump format: missing ATOMS header")
        cols = lines[i].split()[2:]
        i += 1
        colmap = {name: idx for idx, name in enumerate(cols)}
        required = ("id", "x", "y")
        missing = [name for name in required if name not in colmap]
        if missing:
            raise RuntimeError(f"ATOMS header missing required columns: {', '.join(missing)}")

        for _ in range(nat):
            if i >= len(lines) or lines[i].startswith("ITEM:"):
                raise RuntimeError("Bad dump format: atom count does not match NUMBER OF ATOMS")
            parts = lines[i].split()
            pid = int(parts[colmap["id"]])
            r = float(parts[colmap["x"]])
            z = float(parts[colmap["y"]])
            tracks[pid].append((step, r, z))
            i += 1

    return tracks, bounds


def parse_surface_segments(path: Path):
    lines = path.read_text().splitlines()
    points = {}
    segments = []
    mode = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        lower = line.lower()
        if lower == "points":
            mode = "points"
            continue
        if lower == "lines":
            mode = "lines"
            continue

        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue

        if mode == "points" and len(parts) >= 3:
            points[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif mode == "lines" and len(parts) >= 3:
            p1 = int(parts[1])
            p2 = int(parts[2])
            if p1 in points and p2 in points:
                segments.append((points[p1], points[p2]))

    return segments


def detect_surface_files(case_dir: Path, input_file: Path):
    if not input_file.exists():
        return []

    surfaces = []
    for raw in input_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("read_surf"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        surf_path = Path(parts[1])
        if not surf_path.is_absolute():
            surf_path = case_dir / surf_path
        if surf_path.exists():
            surfaces.append(surf_path)
    return surfaces


def split_segments(track, bounds):
    if not track:
        return []
    if bounds is None:
        jump_r = jump_z = 1.0
    else:
        rlo, rhi, zlo, zhi = bounds
        jump_r = 0.25 * abs(rhi - rlo)
        jump_z = 0.25 * abs(zhi - zlo)

    out = []
    cur = [track[0]]
    for prev, curpt in zip(track[:-1], track[1:]):
        step_reset = curpt[0] <= prev[0]
        jump = abs(curpt[1] - prev[1]) > jump_r or abs(curpt[2] - prev[2]) > jump_z
        if step_reset or jump:
            if len(cur) > 1:
                out.append(cur)
            cur = [curpt]
        else:
            cur.append(curpt)
    if len(cur) > 1:
        out.append(cur)
    return out


def add_surface_plot(ax, surface_paths):
    colors = ["0.2", "0.45", "0.6"]
    for idx, surf_path in enumerate(surface_paths):
        segments = parse_surface_segments(surf_path)
        if not segments:
            continue
        lc = LineCollection(
            segments,
            colors=colors[idx % len(colors)],
            linewidths=1.0 if idx else 1.3,
            alpha=0.85,
            zorder=1,
        )
        ax.add_collection(lc)


def plot_single_track(ax, track):
    steps = np.asarray([p[0] for p in track], dtype=float)
    rz = np.asarray([[p[1], p[2]] for p in track], dtype=float)

    if len(rz) > 1:
        segments = np.stack([rz[:-1], rz[1:]], axis=1)
        lc = LineCollection(
            segments,
            cmap="viridis",
            norm=Normalize(vmin=steps.min(), vmax=steps.max()),
            linewidths=2.0,
            zorder=3,
        )
        lc.set_array(steps[:-1])
        ax.add_collection(lc)
        return lc, rz

    ax.plot(rz[:, 0], rz[:, 1], color="tab:blue", linewidth=2.0, zorder=3)
    return None, rz


def plot_many_tracks(ax, tracks, bounds):
    ntracks = len(tracks)
    for idx, (pid, track) in enumerate(sorted(tracks.items())):
        for iseg, seg in enumerate(split_segments(track, bounds)):
            rr = [p[1] for p in seg]
            zz = [p[2] for p in seg]
            ax.plot(
                rr,
                zz,
                linewidth=1.1,
                alpha=0.9,
                label=f"id {pid}" if (ntracks <= 8 and iseg == 0) else None,
                zorder=3,
            )


def main():
    parser = argparse.ArgumentParser(description="Plot WEST particle trajectories in the R-Z plane.")
    case_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--dump", type=Path, default=case_dir / "output" / "state.west.o_sputter",
                        help="Particle dump file to read.")
    parser.add_argument("--input", type=Path, default=case_dir / "in.test",
                        help="Input file used to auto-detect read_surf geometry.")
    parser.add_argument("--wall-file", type=Path, action="append", default=None,
                        help="Explicit surface geometry file(s) to overlay. Can be passed multiple times.")
    parser.add_argument("--particle-id", type=int, default=None,
                        help="Plot only one particle ID. Default: all particles, or the only ID present.")
    parser.add_argument("--output", type=Path, default=case_dir / "output" / "trajectory.west.rz.png",
                        help="Output image path.")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("RMIN", "RMAX"),
                        help="Optional R-axis limits.")
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("ZMIN", "ZMAX"),
                        help="Optional Z-axis limits.")
    parser.add_argument("--tight", action="store_true",
                        help="Zoom to the selected trajectory extent instead of the full dump bounds.")
    parser.add_argument("--pad", type=float, default=2.0e-4,
                        help="Padding in meters used with --tight (default: 2e-4).")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively.")
    args = parser.parse_args()

    dump_path = args.dump if args.dump.is_absolute() else case_dir / args.dump
    input_path = args.input if args.input.is_absolute() else case_dir / args.input
    output_path = args.output if args.output.is_absolute() else case_dir / args.output

    tracks, bounds = parse_dump(dump_path)
    if not tracks:
        raise RuntimeError(f"No particle trajectory found in {dump_path}")

    if args.wall_file:
        surface_paths = []
        for path in args.wall_file:
            surface_paths.append(path if path.is_absolute() else case_dir / path)
    else:
        surface_paths = detect_surface_files(case_dir, input_path)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=220, constrained_layout=True)
    add_surface_plot(ax, [p for p in surface_paths if p.exists()])

    chosen_pid = args.particle_id
    if chosen_pid is None and len(tracks) == 1:
        chosen_pid = next(iter(tracks))

    lc = None
    rz = None
    if chosen_pid is not None:
        if chosen_pid not in tracks:
            raise RuntimeError(f"Particle ID {chosen_pid} not found in {dump_path}")
        track = tracks[chosen_pid]
        lc, rz = plot_single_track(ax, track)
        ax.scatter(rz[0, 0], rz[0, 1], s=36, c="tab:green", marker="o", label="start", zorder=4)
        ax.scatter(rz[-1, 0], rz[-1, 1], s=42, c="tab:red", marker="x", label="end", zorder=4)
        ax.set_title(f"WEST trajectory in R-Z for particle {chosen_pid}")
    else:
        plot_many_tracks(ax, tracks, bounds)
        ax.set_title(f"WEST trajectories in R-Z, N={len(tracks)}")

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.set_aspect("equal", adjustable="box")

    if bounds is not None:
        rlo, rhi, zlo, zhi = bounds
        ax.set_xlim(rlo, rhi)
        ax.set_ylim(zlo, zhi)

    if args.xlim is not None:
        ax.set_xlim(*args.xlim)
    if args.ylim is not None:
        ax.set_ylim(*args.ylim)
    if args.tight and rz is not None and len(rz) > 0:
        rmin = float(np.min(rz[:, 0]))
        rmax = float(np.max(rz[:, 0]))
        zmin = float(np.min(rz[:, 1]))
        zmax = float(np.max(rz[:, 1]))
        rpad = max(args.pad, 0.05 * max(rmax - rmin, 1.0e-12))
        zpad = max(args.pad, 0.05 * max(zmax - zmin, 1.0e-12))
        ax.set_xlim(rmin - rpad, rmax + rpad)
        ax.set_ylim(zmin - zpad, zmax + zpad)

    if lc is not None:
        cbar = fig.colorbar(lc, ax=ax, pad=0.02)
        cbar.set_label("Step")
    elif len(tracks) <= 8:
        ax.legend(frameon=True, framealpha=0.9, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    print(f"Wrote: {output_path}")
    print(f"Plotted trajectories: {len(tracks)}")
    if chosen_pid is not None:
        print(f"Particle ID: {chosen_pid}")
    if surface_paths:
        print("Surface overlays:")
        for surf_path in surface_paths:
            print(f"  {surf_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
