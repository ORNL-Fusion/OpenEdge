#!/usr/bin/env python3
"""Plot transfer matrix with segment map + inner/outer injection distributions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

WALL_REGIONS = {
    "lfs_wall": list(range(1, 50)) + list(range(151, 157)),
    "up_outer_div": list(range(50, 64)),
    "crown": list(range(64, 83)),
    "up_inner_div": list(range(83, 91)),
    "hfs_wall": list(range(91, 110)),
    "lo_inner_div": list(range(110, 118)),
    "dome": list(range(118, 137)),
    "lo_outer_div": list(range(137, 151)),
}

DEST_ORDER = [
    "lfs_wall",
    "up_outer_div",
    "crown",
    "up_inner_div",
    "hfs_wall",
    "lo_inner_div",
    "dome",
    "lo_outer_div",
    "core",
]

_surf_to_region = {sid: name for name, ids in WALL_REGIONS.items() for sid in ids}


def surf_id_to_region(surf_id: int, n_wall: int = 156) -> str:
    if surf_id > n_wall:
        return "core"
    return _surf_to_region.get(surf_id, "unknown")


def infer_last_ts(events_csv: Path) -> int:
    if not events_csv.exists():
        return 0
    df = pd.read_csv(events_csv)
    if df.empty or "timestep" not in df.columns:
        return 0
    return int(df["timestep"].max())


def read_points_and_lines(surf_path: Path):
    pts = {}
    lines = []
    mode = None
    with surf_path.open() as fh:
        for raw in fh:
            s = raw.strip()
            if not s:
                continue
            if s == "Points":
                mode = "points"
                continue
            if s == "Lines":
                mode = "lines"
                continue
            tok = s.split()
            if mode == "points":
                pid = int(tok[0])
                pts[pid] = (float(tok[1]), float(tok[2]))
            elif mode == "lines":
                lid = int(tok[0])
                p1 = int(tok[1])
                p2 = int(tok[2])
                lines.append((lid, p1, p2))
    return pts, lines


def point_segment_distance(px, py, x1, y1, x2, y2):
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        dx = px - x1
        dy = py - y1
        return dx * dx + dy * dy
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        dx = px - x2
        dy = py - y2
        return dx * dx + dy * dy
    t = c1 / c2
    projx = x1 + t * vx
    projy = y1 + t * vy
    dx = px - projx
    dy = py - projy
    return dx * dx + dy * dy


def build_wall_segments(wall_path: Path):
    pts, lines = read_points_and_lines(wall_path)
    segs = []
    for lid, a, b in lines:
        if a not in pts or b not in pts:
            continue
        x1, y1 = pts[a]
        x2, y2 = pts[b]
        tx = x2 - x1
        ty = y2 - y1
        n1x, n1y = -ty, tx
        n2x, n2y = ty, -tx
        n1n = max((n1x * n1x + n1y * n1y) ** 0.5, 1.0e-16)
        n2n = max((n2x * n2x + n2y * n2y) ** 0.5, 1.0e-16)
        segs.append(
            {
                "lid": lid,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "n1": (n1x / n1n, n1y / n1n),
                "n2": (n2x / n2n, n2y / n2n),
            }
        )
    return segs


def parse_first_appearance_particles(path: Path):
    cols = None
    first = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("ITEM: ATOMS"):
            cols = s.split()[2:]
            i += 1
            continue
        if s.startswith("ITEM:"):
            i += 1
            continue
        if cols is None or not s:
            i += 1
            continue
        parts = s.split()
        if len(parts) != len(cols):
            i += 1
            continue
        row = {c: parts[k] for k, c in enumerate(cols)}
        pid = int(float(row.get("id", "nan")))
        if pid not in first:
            first[pid] = row
        i += 1
    return first


def incidence_df(case_path: Path, wall_path: Path, label: str):
    if not case_path.exists():
        return pd.DataFrame(columns=["id", "angle_deg", "vnormal_mps", "label"])

    segs = build_wall_segments(wall_path)
    first = parse_first_appearance_particles(case_path)
    rows = []

    for pid, row in first.items():
        x = float(row.get("x", 0.0))
        y = float(row.get("y", 0.0))
        vx = float(row.get("vx", 0.0))
        vy = float(row.get("vy", 0.0))
        vz = float(row.get("vz", 0.0))
        vnorm = max((vx * vx + vy * vy + vz * vz) ** 0.5, 1.0e-30)

        best = None
        best_d2 = 1.0e300
        for seg in segs:
            d2 = point_segment_distance(x, y, seg["x1"], seg["y1"], seg["x2"], seg["y2"])
            if d2 < best_d2:
                best_d2 = d2
                best = seg
        if best is None:
            continue

        n1 = best["n1"]
        n2 = best["n2"]
        d1 = abs(vx * n1[0] + vy * n1[1])
        d2 = abs(vx * n2[0] + vy * n2[1])
        nx, ny = n1 if d1 >= d2 else n2

        vnormal = abs(vx * nx + vy * ny)
        cosang = vnormal / vnorm
        cosang = max(-1.0, min(1.0, cosang))
        angle_deg = float(np.degrees(np.arccos(cosang)))

        rows.append({"id": pid, "angle_deg": angle_deg, "vnormal_mps": vnormal, "label": label})

    return pd.DataFrame(rows)


def plot_segment_panel(
    ax,
    wall_path: Path,
    core_path: Path,
    wall_label_every: int = 10,
    core_label_every: int = 20,
    label_fontsize: float = 8.0,
    show_legend: bool = True,
):
    wall_pts, wall_lines = read_points_and_lines(wall_path)
    n_wall = len(wall_lines)

    cmap = plt.colormaps.get_cmap("tab10")
    region_color = {
        name: cmap(i / max(len(DEST_ORDER) - 1, 1))
        for i, name in enumerate(DEST_ORDER)
    }

    for sid, a, b in wall_lines:
        if a not in wall_pts or b not in wall_pts:
            continue
        p1 = wall_pts[a]
        p2 = wall_pts[b]
        region = surf_id_to_region(sid, n_wall=156)
        color = region_color.get(region, "gray")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=2.0, solid_capstyle="round")

        if wall_label_every > 0 and (sid % wall_label_every) == 1:
            mR = 0.5 * (p1[0] + p2[0])
            mZ = 0.5 * (p1[1] + p2[1])
            dR, dZ = mR - 3.5, mZ
            norm = max((dR * dR + dZ * dZ) ** 0.5, 0.01)
            ax.text(
                mR + 0.06 * dR / norm,
                mZ + 0.06 * dZ / norm,
                str(sid),
                fontsize=label_fontsize,
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    if core_path.exists():
        core_pts, core_lines = read_points_and_lines(core_path)
        for lid, a, b in core_lines:
            if a not in core_pts or b not in core_pts:
                continue
            p1 = core_pts[a]
            p2 = core_pts[b]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=region_color["core"], lw=1.4, ls="--")

            if core_label_every > 0 and (lid % core_label_every) == 1:
                shifted_id = n_wall + lid
                mR = 0.5 * (p1[0] + p2[0])
                mZ = 0.5 * (p1[1] + p2[1])
                dR, dZ = mR - 3.5, mZ
                norm = max((dR * dR + dZ * dZ) ** 0.5, 0.01)
                ax.text(
                    mR + 0.05 * dR / norm,
                    mZ + 0.05 * dZ / norm,
                    str(shifted_id),
                    fontsize=label_fontsize,
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

    if show_legend:
        wall_handles = [Patch(color=region_color[r], label=r) for r in DEST_ORDER if r != "core"]
        core_handle = Line2D([0], [0], color=region_color["core"], lw=1.8, ls="--", label="core")
        leg = ax.legend(
            handles=wall_handles + [core_handle],
            loc="center",
            bbox_to_anchor=(0.47, 0.48),
            bbox_transform=ax.transAxes,
            framealpha=0.9,
            fontsize=7,
            ncol=1,
        )
        leg.get_frame().set_edgecolor("0.7")

#    ax.set_title("Segment Map")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts",
        default="transfer_matrix.mode1.nemit200.counts.csv",
        help="Path to transfer matrix counts CSV",
    )
    parser.add_argument(
        "--events",
        default="transfer_matrix.mode1.nemit200.events.csv",
        help="Path to transfer matrix events CSV (for timestep in title)",
    )
    parser.add_argument("--launched-I", type=int, default=200, help="Launched droplets for I row")
    parser.add_argument("--launched-O", type=int, default=0, help="Launched droplets for O row")
    parser.add_argument("--source-region-I", type=str, default="lo_inner_div")
    parser.add_argument("--source-region-O", type=str, default="lo_outer_div")
    parser.add_argument(
        "--subtract-emission",
        action="store_true",
        default=False,
        help="Subtract source-surface emission counts before percentage normalization",
    )
    parser.add_argument(
        "--no-subtract-emission",
        dest="subtract_emission",
        action="store_false",
        help="Disable source-emission subtraction",
    )
    parser.add_argument("--out", default="Figs/transfer_matrix_percentage1.png", help="Output PNG")
    parser.add_argument("--inner", default="case.inner", help="Inner source particle dump")
    parser.add_argument("--outer", default="case.outer", help="Outer source particle dump")
    parser.add_argument("--wall", default="wall.surf", help="Wall surface file")
    parser.add_argument("--core", default="core.surf", help="Core surface file")
    parser.add_argument("--wall-label-every", type=int, default=4)
    parser.add_argument("--core-label-every", type=int, default=20)
    parser.add_argument("--label-fontsize", type=float, default=5.0)
    parser.add_argument("--no-segment-legend", action="store_true", help="Disable segment map legend")
    args = parser.parse_args()

    print("Region -> surf ID ranges:")
    for name, ids in WALL_REGIONS.items():
        print(f"  {name:20s}: {min(ids):3d} - {max(ids):3d}  ({len(ids)} elements)")
    print(f"  Total wall elements: {sum(len(v) for v in WALL_REGIONS.values())}")

    counts_path = Path(args.counts)
    if not counts_path.exists():
        raise SystemExit(f"Missing counts CSV: {counts_path}")

    raw = pd.read_csv(counts_path).set_index("source")

    numeric_surf_cols = []
    for col in raw.columns:
        try:
            int(str(col))
            numeric_surf_cols.append(col)
        except ValueError:
            pass

    if numeric_surf_cols:
        matrix = pd.DataFrame(0, index=raw.index, columns=DEST_ORDER, dtype=float)
        for col in raw.columns:
            if str(col) == "evaporated":
                continue
            try:
                sid = int(str(col))
            except ValueError:
                continue
            region = surf_id_to_region(sid, n_wall=156)
            if region not in matrix.columns:
                matrix[region] = 0.0
            matrix[region] += raw[col].astype(float)
        if "evaporated" in raw.columns:
            matrix["evaporated"] = raw["evaporated"].astype(float)
    else:
        keep = [c for c in DEST_ORDER if c in raw.columns]
        if "evaporated" in raw.columns:
            keep = keep + ["evaporated"]
        matrix = raw[keep].copy().astype(float)

    if args.subtract_emission:
        if "I" in matrix.index and args.source_region_I in matrix.columns:
            matrix.at["I", args.source_region_I] = max(
                0.0, float(matrix.at["I", args.source_region_I]) - float(args.launched_I)
            )
        if "O" in matrix.index and args.source_region_O in matrix.columns:
            matrix.at["O", args.source_region_O] = max(
                0.0, float(matrix.at["O", args.source_region_O]) - float(args.launched_O)
            )

        if "evaporated" in matrix.columns:
            for row in matrix.index:
                launched = float({"I": args.launched_I, "O": args.launched_O}.get(row, 0))
                hits = float(matrix.loc[row].drop(labels=["evaporated"], errors="ignore").sum())
                matrix.at[row, "evaporated"] = max(0.0, launched - hits)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    N_LAUNCHED = {"I": args.launched_I, "O": args.launched_O}
    matrix_pct = matrix.copy().astype(float)
    for row in matrix_pct.index:
        launched = float(N_LAUNCHED.get(row, 0))
        if launched > 0:
            matrix_pct.loc[row] = 100.0 * matrix_pct.loc[row] / launched
        else:
            matrix_pct.loc[row] = 0.0

    annot_fmt = np.vectorize(lambda x: f"{x:.1f}")

    df_inner = incidence_df(Path(args.inner), Path(args.wall), "Inner")
    df_outer = incidence_df(Path(args.outer), Path(args.wall), "Outer")

    fig = plt.figure(figsize=(13.6, 9), dpi=150)
#    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.25], hspace=0.28)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1.15], hspace=0.28)

    top = gs[0].subgridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.16)
    ax_seg = fig.add_subplot(top[0, 0])
    ax_i = fig.add_subplot(top[0, 1])
    ax_o = fig.add_subplot(top[0, 2], sharey=ax_i)

    # Remove gap between inner/outer distribution panels.
    pos_i = ax_i.get_position()
    pos_o = ax_o.get_position()
    ax_o.set_position([pos_i.x1, pos_o.y0, pos_o.width, pos_o.height])

    plot_segment_panel(
        ax_seg,
        Path(args.wall),
        Path(args.core),
        wall_label_every=max(0, int(args.wall_label_every)),
        core_label_every=max(0, int(args.core_label_every)),
        label_fontsize=float(args.label_fontsize),
        show_legend=not args.no_segment_legend,
    )

    if not df_inner.empty:
        ax_i.scatter(df_inner["angle_deg"], df_inner["vnormal_mps"], s=14, alpha=0.7, edgecolors="none")
    ax_i.set_title("Inner")
    ax_i.set_xlabel(r"Angle $\theta$ (deg)")
    ax_i.set_ylabel(r"Speed $|v \cdot n|$ (m/s)")
    ax_i.minorticks_on()
    ax_i.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax_i.grid(True, linestyle="--", alpha=0.3)

    if not df_outer.empty:
        ax_o.scatter(df_outer["angle_deg"], df_outer["vnormal_mps"], s=14, alpha=0.7, edgecolors="none")
    ax_o.set_title("Outer")
    ax_o.set_xlabel(r"Angle $\theta$ (deg)")
    ax_o.tick_params(axis="y", labelleft=False)
    ax_o.minorticks_on()
    ax_o.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax_o.grid(True, linestyle="--", alpha=0.3)

    ax_h = fig.add_subplot(gs[1, 0])
    sns.heatmap(
        matrix_pct,
        ax=ax_h,
        annot=annot_fmt(matrix_pct.values),
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar_kws={"label": "perc. injected droplets (%)"},
    )

    # Show physical source region names instead of internal I/O row keys.
    ylabels = []
    for src in matrix_pct.index:
        if src == "I":
            ylabels.append(args.source_region_I)
        elif src == "O":
            ylabels.append(args.source_region_O)
        else:
            ylabels.append(str(src))
    ax_h.set_yticklabels(ylabels, rotation=0)

    last_ts = infer_last_ts(Path(args.events))
#    ax_h.set_title(f"Transfer matrix - particle outcomes (T = {last_ts:,} steps)")
    ax_h.set_xlabel("destination region")
    ax_h.set_ylabel("source")
    ax_h.tick_params(axis="x", rotation=32)
    ax_h.tick_params(axis="y", rotation=0)

    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
