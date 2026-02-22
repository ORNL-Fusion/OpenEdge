
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from utils import surface

WALL_REGIONS = {
    "lfs_wall":         list(range(1, 50)) + list(range(151, 157)),
    "up_outer_div":  list(range(50, 64)),
    "crown":            list(range(64, 83)),
    "up_inner_div":  list(range(83, 91)),
    "hfs_wall":         list(range(91, 110)),
    "lo_inner_div":  list(range(110, 118)),
    "dome":             list(range(118, 137)),
    "lo_outer_div":  list(range(137, 151)),
}

_surf_to_region = {sid: name for name, ids in WALL_REGIONS.items() for sid in ids}

def surf_id_to_region(surf_id, n_wall=156):
    if surf_id > n_wall:
        return "core"
    return _surf_to_region.get(surf_id, "unknown")

DEST_ORDER = [
    "lfs_wall", "up_outer_div", "crown",
    "up_inner_div", "hfs_wall",
    "lo_inner_div", "dome", "lo_outer_div",
    "core",
]

WALL_SURF = "/Users/42d/OpenedgeGPU/examples/test_droplet/wall.surf"
CORE_SURF = "/Users/42d/OpenedgeGPU/examples/test_droplet/core.surf"
FIG_OUT   = "/Users/42d/OpenedgeGPU/examples/test_droplet/Figs/wall_surf_ids.png"


def read_points_and_lines(surf_path):
    """
    Parse a .surf file with sections:
      Points
        pid R Z ...
      Lines
        lid p1 p2 ...
    Returns:
      pts: dict[int] -> (R,Z)
      lines: list[tuple(lid, p1, p2)]
    """
    pts = {}
    lines = []

    mode = None
    with open(surf_path) as fh:
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
                # pid R Z  (ignore possible extra columns)
                pid = int(tok[0])
                pts[pid] = (float(tok[1]), float(tok[2]))
            elif mode == "lines":
                # lid p1 p2 (ignore possible extra columns)
                lid = int(tok[0])
                p1  = int(tok[1])
                p2  = int(tok[2])
                lines.append((lid, p1, p2))

    return pts, lines


def plot_wall_ids():
    # --- wall points (keep your simple parsing, or use helper) ---
    wall_pts = {}
    with open(WALL_SURF) as fh:
        in_pts = False
        for line in fh:
            s = line.strip()
            if s == "Points":  in_pts = True;  continue
            if s == "Lines":   break
            if in_pts and s:
                tok = s.split()
                wall_pts[int(tok[0])] = (float(tok[1]), float(tok[2]))
    n_wall = len(wall_pts)

    # --- core points + lines (this is the important part) ---
    core_pts, core_lines = read_points_and_lines(CORE_SURF)
    n_core_lines = len(core_lines)

    region_list = DEST_ORDER[:]  # ordered legend/colormap

    cmap_reg = plt.colormaps.get_cmap("tab10")
    region_color = {
        name: cmap_reg(i / max(len(region_list) - 1, 1))
        for i, name in enumerate(region_list)
    }

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, ax = plt.subplots(figsize=(4, 6), dpi=150)

    # ---------------- wall segments ----------------
    for sid in range(1, n_wall + 1):
        p1 = wall_pts[sid]
        p2 = wall_pts[(sid % n_wall) + 1]

        region = surf_id_to_region(sid, n_wall)
        color  = region_color.get(region, "gray")

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, lw=2.5, solid_capstyle="round")

        # label every other wall surf ID
        WALL_LABEL_EVERY = 4
#        if sid % 2 == 1:
        if (sid % WALL_LABEL_EVERY) == 1:
            mR = 0.5 * (p1[0] + p2[0])
            mZ = 0.5 * (p1[1] + p2[1])
            dR, dZ = mR - 3.5, mZ
            norm   = max((dR**2 + dZ**2)**0.5, 0.01)
            ax.text(mR + 0.07 * dR / norm, mZ + 0.07 * dZ / norm, str(sid),
                    fontsize=5, ha="center", va="center",
                    color="black", fontweight="bold")

    # ---------------- core segments (shifted IDs) ----------------
    core_color = region_color["core"]

    # how often to label core segments (296 labels will be chaos)
    CORE_LABEL_EVERY = 20  # tweak: 5, 10, 20...

    for (lid, a, b) in core_lines:
        if a not in core_pts or b not in core_pts:
            continue
        p1 = core_pts[a]
        p2 = core_pts[b]

        shifted_id = n_wall + lid  # <-- the key shift

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=core_color, lw=2.5, ls="--", solid_capstyle="round")

        if (lid % CORE_LABEL_EVERY) == 1:
            mR = 0.5 * (p1[0] + p2[0])
            mZ = 0.5 * (p1[1] + p2[1])
            dR, dZ = mR - 3.5, mZ
            norm   = max((dR**2 + dZ**2)**0.5, 0.01)
            ax.text(mR + 0.05 * dR / norm, mZ + 0.05 * dZ / norm, str(shifted_id),
                    fontsize=5, ha="center", va="center",
                    color="black", fontweight="bold")

#    # legend: patches for wall regions + a dashed handle for core
#    wall_handles = [mpatches.Patch(color=region_color[r], label=r) for r in DEST_ORDER if r != "core"]
#    core_handle  = Line2D([0], [0], color=core_color, lw=2.5, ls="--", label="core")
#    ax.legend(handles=wall_handles + [core_handle],
#              fontsize=8, loc="center", bbox_to_anchor=(1.02, 1.0),
#              borderaxespad=0, framealpha=0.9)

    # legend: patches for wall regions + dashed handle for core
    wall_handles = [mpatches.Patch(color=region_color[r], label=r)
                    for r in DEST_ORDER if r != "core"]
    core_handle  = Line2D([0], [0], color=core_color, lw=2.5, ls="--", label="core")

    # Put legend inside the axes (core area)
    leg = ax.legend(
        handles=wall_handles + [core_handle],
        loc="center",
        bbox_to_anchor=(0.48, 0.48),   # <--- tweak these two numbers
        bbox_transform=ax.transAxes,   # interpret anchor in axes coords
        framealpha=0.90,
        fontsize=8,
        ncol=1
    )
    leg.get_frame().set_edgecolor("0.7")

    ax.set_xlabel(r"R (m)", fontsize=14)
    ax.set_ylabel(r"Z (m)", fontsize=14)

    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved → {FIG_OUT}")
    print(f"Wall segments: {n_wall}")
    print(f"Core segments: {n_core_lines}")
    print(f"Core surf-id range (shifted): {n_wall+1} .. {n_wall+n_core_lines}")


if __name__ == "__main__":
    print("Region → surf ID ranges:")
    for name, ids in WALL_REGIONS.items():
        print(f"  {name:25s}: {min(ids):3d} – {max(ids):3d}  ({len(ids)} elements)")
    print(f"  Total wall: {sum(len(v) for v in WALL_REGIONS.values())}")
    plot_wall_ids()
