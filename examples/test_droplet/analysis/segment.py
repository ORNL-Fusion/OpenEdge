"""
segment.py — wall region definitions and diagnostic surf-ID plot.

8 physics-based regions covering all 156 wall surf IDs (no gaps):
  outer_divertor_lower  outer_divertor_upper
  inner_divertor_lower  inner_divertor_upper
  dome  crown  lfs_wall  hfs_wall

"dome" = private flux region floor between inner and outer lower divertors.
outer_divertor_lower wraps the polygon start/end (IDs 1-11 and 139-156).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import surface

# ── Region definitions ────────────────────────────────────────────────────
WALL_REGIONS = {
    "lfs_wall":             list(range(1,  50))+list(range(151,  157)),
    "upper_outer_div": list(range(50, 64)),
     "crown":                list(range(64,  83)),
     "upper_inner_div": list(range(83, 91)),
     "hfs_wall":             list(range(91, 110)),
     "lower_inner_div": list(range(110, 118)),
     "dome":                 list(range(118, 137)),
     "lower_outer_div": list(range(137, 151)),
}

_surf_to_region = {sid: name for name, ids in WALL_REGIONS.items() for sid in ids}

def surf_id_to_region(surf_id, n_wall=156):
    """Return region name for a wall surf element, 'core' if surf_id > n_wall."""
    if surf_id > n_wall:
        return "core"
    return _surf_to_region.get(surf_id, "unknown")

DEST_ORDER = [
    "lfs_wall", "upper_outer_div", "crown",
    "upper_inner_div", "hfs_wall",
    "lower_inner_div", "dome", "lower_outer_div", "core",
]

# ── Diagnostic plot ───────────────────────────────────────────────────────
WALL_SURF = "/Users/42d/OpenedgeGPU/examples/test_droplet/wall.surf"
CORE_SURF = "/Users/42d/OpenedgeGPU/examples/test_droplet/core.surf"
FIG_OUT   = "/Users/42d/OpenedgeGPU/examples/test_droplet/Figs/wall_surf_ids.png"

def plot_wall_ids():
    # read wall points
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

    region_list = list(WALL_REGIONS.keys())
    cmap_reg    = plt.colormaps.get_cmap("tab10")
    region_color = {name: cmap_reg(i / max(len(region_list) - 1, 1))
                    for i, name in enumerate(region_list)}


    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })


#    fig, ax = plt.subplots(figsize=(6, 8), dpi=150)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(4, 6),
        dpi=150,
        gridspec_kw={"wspace": 0.0015}   # <--- reduce spacing
    )


    for sid in range(1, n_wall + 1):
        p1 = wall_pts[sid]
        p2 = wall_pts[(sid % n_wall) + 1]
        region = surf_id_to_region(sid, n_wall)
        color  = region_color.get(region, "gray")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=2.5,
                solid_capstyle="round")

        # label every other surf ID, nudged outward from the vessel centroid
        if sid % 2 == 1:
            mR = 0.5 * (p1[0] + p2[0])
            mZ = 0.5 * (p1[1] + p2[1])
            dR, dZ = mR - 3.5, mZ
            norm   = max((dR**2 + dZ**2)**0.5, 0.01)
            ax.text(mR + 0.07 * dR / norm, mZ + 0.07 * dZ / norm, str(sid),
                    fontsize=5, ha="center", va="center",
                    color="black", fontweight="bold")

    # core boundary
    core_s = surface(CORE_SURF, "2D")
    cR, cZ = core_s.polygon.exterior.xy
    ax.plot(cR, cZ, color="green", lw=2.5, ls="--")


    patches = [mpatches.Patch(color=region_color[r], label=r) for r in region_list]
    ax.legend(handles=patches, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0, framealpha=0.9)

    ax.set_xlabel(r"R (m)", fontsize=14)
    ax.set_ylabel(r"Z (m)", fontsize=14)
#    ax.set_title("Wall surf IDs (odd labels)\nRegion colouring", fontsize=11)
#    ax.set_aspect("equal")
#    ax.grid(True, ls=":", alpha=0.4)
    
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both",
                   direction="in", top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.3)


    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {FIG_OUT}")


if __name__ == "__main__":
    print("Region → surf ID ranges:")
    for name, ids in WALL_REGIONS.items():
        print(f"  {name:25s}: {min(ids):3d} – {max(ids):3d}  ({len(ids)} elements)")
    print(f"  Total: {sum(len(v) for v in WALL_REGIONS.values())}")
    plot_wall_ids()
