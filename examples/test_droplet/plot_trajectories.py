#!/usr/bin/env python3
"""Plot droplet trajectories from case.outer over the axi wall.surf outline."""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def parse_dump(path):
    blocks = {"t": [], "id": [], "x": [], "y": [], "temp": [], "radius": []}
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    t = None
    while i < len(lines):
        s = lines[i].strip()
        if s == "ITEM: TIMESTEP":
            t = int(lines[i+1]); i += 2; continue
        if s == "ITEM: NUMBER OF ATOMS":
            n = int(lines[i+1]); i += 2; continue
        if s.startswith("ITEM: BOX BOUNDS"):
            i += 4; continue
        if s.startswith("ITEM: ATOMS"):
            hdr = s.split()[2:]
            col = {k: j for j, k in enumerate(hdr)}
            for k in range(n):
                v = lines[i+1+k].split()
                blocks["t"].append(t)
                blocks["id"].append(int(v[col["id"]]))
                blocks["x"].append(float(v[col["x"]]))
                blocks["y"].append(float(v[col["y"]]))
                blocks["temp"].append(float(v[col["temp"]]))
                blocks["radius"].append(float(v[col["radius"]]))
            i += 1 + n; continue
        i += 1
    return {k: np.asarray(v) for k, v in blocks.items()}


def parse_wall(path):
    with open(path) as f:
        txt = f.read()
    pts = []
    in_pts = False
    for line in txt.splitlines():
        s = line.strip()
        if s == "Points": in_pts = True; continue
        if s == "Lines": in_pts = False; continue
        if in_pts and s:
            parts = s.split()
            if len(parts) >= 3:
                pts.append((float(parts[1]), float(parts[2])))
    return np.asarray(pts)


def main():
    base = Path(__file__).resolve().parent
    data = parse_dump(base / "case.outer")
    wall = parse_wall(base / "wall.surf")

    # axi slot layout: x=Z (axial), y=R (radial)
    Z_dr = data["x"]; R_dr = data["y"]
    Z_w = wall[:, 0]; R_w = wall[:, 1]

    fig = plt.figure(figsize=(12, 5.2), dpi=140)
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.05, wspace=0.3)
    ax  = fig.add_subplot(gs[:, 0])
    axR = fig.add_subplot(gs[0, 1])
    axT = fig.add_subplot(gs[1, 1], sharex=axR)

    ax.plot(R_w, Z_w, "-", lw=1.0, color="0.4", label="vessel wall")
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title("Droplet trajectories in (R, Z)")
    ax.set_aspect("equal")
    ax.set_xlim(2.5, 5.0)
    z_hi = float(np.nanmax(Z_dr)) + 0.3
    ax.set_ylim(-4.0, max(0.5, z_hi))

    ids = np.unique(data["id"])
    cmap = plt.get_cmap("plasma")
    # Sort by launch radius so "drop 1/2/3" labels match the size ordering.
    r0_per_id = {}
    for pid in ids:
        m = data["id"] == pid
        pos = data["radius"][m]
        pos = pos[pos > 0]
        r0_per_id[pid] = pos[0] if pos.size else 0.0
    ids = sorted(ids, key=lambda p: r0_per_id[p])

    for k, pid in enumerate(ids):
        m = data["id"] == pid
        r_series = data["radius"][m]
        r0 = r0_per_id[pid]
        frac = np.where(r_series > 0, r_series / r0, np.nan)
        R_seg = R_dr[m]; Z_seg = Z_dr[m]
        pts = np.column_stack([R_seg, Z_seg]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 1), lw=2.0)
        lc.set_array(frac[:-1])
        ax.add_collection(lc)

    # Launch marker (shared — all three droplets start at the same point).
    pid0 = ids[0]
    m0 = data["id"] == pid0
    ax.plot(R_dr[m0][0], Z_dr[m0][0], "o", ms=11,
            color="white", mec="k", mew=1.5, zorder=6)
    ax.annotate("launch", (R_dr[m0][0], Z_dr[m0][0]),
                xytext=(10, -10), textcoords="offset points", fontsize=9)

    # End markers with labels.
    legend_lines = []
    for k, pid in enumerate(ids):
        m = data["id"] == pid
        r0 = r0_per_id[pid]
        R_end = R_dr[m][-1]; Z_end = Z_dr[m][-1]
        r_end = data["radius"][m][-1]
        ax.plot(R_end, Z_end, "*", ms=18,
                color="red", mec="white", mew=1.5, zorder=7)
        dx = 0.18 if k % 2 == 0 else -0.05
        dy = 0.10 * (1 if k < 2 else -1)
        ax.annotate(f"drop {k+1} END", (R_end, Z_end),
                    xytext=(R_end + dx, Z_end + dy),
                    fontsize=9, color="darkred",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="0.4"))
        legend_lines.append(f"drop {k+1}: {r0*1e3:.1f} mm → {r_end*1e3:.2f} mm")

    cb = fig.colorbar(lc, ax=ax, shrink=0.85)
    cb.set_label("r(t) / r₀")
    # Size summary box — bottom-right corner (away from trajectories).
    ax.text(0.98, 0.02, "\n".join(legend_lines),
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9),
            verticalalignment="bottom", horizontalalignment="right")

    # Right panels — radius (top) and temperature (bottom) vs. timestep.
    # Each droplet gets a consistent color across the two panels. ids is
    # already sorted by launch radius, so drop 1 = smallest.
    drop_colors = plt.get_cmap("tab10")
    for k, pid in enumerate(ids):
        m = data["id"] == pid
        c = drop_colors(k)
        r_mm = data["radius"][m] * 1e3
        axR.plot(data["t"][m], np.where(r_mm > 0, r_mm, np.nan),
                 lw=1.6, color=c, label=f"drop {k+1}")
        axT.plot(data["t"][m],
                 np.where(data["temp"][m] > 0, data["temp"][m], np.nan),
                 lw=1.6, color=c)
    axR.set_ylabel("radius [mm]")
    axR.grid(alpha=0.3)
    axR.legend(fontsize=9, loc="upper right")
    plt.setp(axR.get_xticklabels(), visible=False)

    axT.set_xlabel("timestep")
    axT.set_ylabel("T [K]")
    axT.grid(alpha=0.3)

    fig.tight_layout()
    out = base / "trajs.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
