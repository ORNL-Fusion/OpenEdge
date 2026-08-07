#!/usr/bin/env python3
"""Plot droplet trajectories from output/case.outer in the (R, Z) plane.

Exits 0 on PASS, 1 on FAIL (for regression use).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

N_DROPLETS = 3
R0_MM = (1.5, 2.5, 3.5)        # launch radii from droplet.species
BOX = (-3.8, 3.8, 0.0, 5.7)    # xlo xhi ylo yhi


def parse_dump(path):
    blocks = {"t": [], "id": [], "x": [], "y": [], "radius": []}
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
    data = parse_dump(base / "output" / "case.outer")
    wall = parse_wall(base / "wall.surf")

    # axi slot layout: x = Z (axial), y = R (radial)
    Z_dr = data["x"]; R_dr = data["y"]
    Z_w = wall[:, 0]; R_w = wall[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    ax.plot(R_w, Z_w, "-", lw=1.0, color="0.4", label="vessel wall")
    ax.set_xlabel(r"$R$ (m)")
    ax.set_ylabel(r"$Z$ (m)")
    ax.set_title("Droplet trajectories in $(R, Z)$")
    ax.set_aspect("equal")

    # Auto-zoom around the trajectory extent with 25 % padding so short arcs
    # don't disappear into a vessel-wide axis. Falls back to vessel bounds if
    # only a single point of trajectory exists.
    R_tr = R_dr[np.isfinite(R_dr)]; Z_tr = Z_dr[np.isfinite(Z_dr)]
    if R_tr.size > 1 and Z_tr.size > 1:
        Rmin, Rmax = float(R_tr.min()), float(R_tr.max())
        Zmin, Zmax = float(Z_tr.min()), float(Z_tr.max())
        spanR = max(Rmax - Rmin, 0.02)
        spanZ = max(Zmax - Zmin, 0.02)
        span  = max(spanR, spanZ)
        Rc = 0.5 * (Rmin + Rmax); Zc = 0.5 * (Zmin + Zmax)
        ax.set_xlim(Rc - 0.75 * span, Rc + 0.75 * span)
        ax.set_ylim(Zc - 0.75 * span, Zc + 0.75 * span)
    else:
        ax.set_xlim(2.5, 5.0); ax.set_ylim(-4.0, 0.5)

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

    # color scale stretched over the actual r/r0 range so small mass loss
    # is still visible (falls back to (0, 1) for full evaporation)
    frac_by_id = {}
    fmin = 1.0
    for pid in ids:
        m = data["id"] == pid
        r_series = data["radius"][m]
        frac = np.where(r_series > 0, r_series / r0_per_id[pid], np.nan)
        frac_by_id[pid] = frac
        fmin = min(fmin, np.nanmin(frac))
    span = max(1.0 - fmin, 1e-6)
    norm = plt.Normalize(fmin - 0.05 * span, 1.0)

    for pid in ids:
        m = data["id"] == pid
        frac = frac_by_id[pid]
        R_seg = R_dr[m]; Z_seg = Z_dr[m]
        pts = np.column_stack([R_seg, Z_seg]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, lw=2.0)
        lc.set_array(frac[:-1])
        ax.add_collection(lc)

    legend_lines = []
    for k, pid in enumerate(ids):
        m = data["id"] == pid
        r0 = r0_per_id[pid]
        r_end = data["radius"][m][-1]
        dm = (r_end / r0) ** 3 - 1.0
        legend_lines.append(f"drop {k+1}: {r0*1e3:.3f} $\\rightarrow$ {r_end*1e3:.4f} mm "
                            f"($\\Delta m$ {dm*100:+.2f}%)")

    cb = fig.colorbar(lc, ax=ax, shrink=0.85)
    cb.set_label(r"$r(t) / r_0$")

    # Vessel-overview inset — small panel showing where the zoomed view sits.
    if R_tr.size > 1 and Z_tr.size > 1:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axin = inset_axes(ax, width="28%", height="28%", loc="upper right",
                          borderpad=1.0)
        axin.plot(R_w, Z_w, "-", lw=0.8, color="0.4")
        for pid in ids:
            m = data["id"] == pid
            axin.plot(R_dr[m], Z_dr[m], "-", lw=1.2, color="C3")
        rect_x = ax.get_xlim(); rect_y = ax.get_ylim()
        axin.add_patch(plt.Rectangle((rect_x[0], rect_y[0]),
                                     rect_x[1] - rect_x[0],
                                     rect_y[1] - rect_y[0],
                                     fill=False, ec="C0", lw=1.2))
        axin.set_aspect("equal")
        axin.set_xticks([]); axin.set_yticks([])
        axin.set_title("vessel", fontsize=8)
    ax.text(0.98, 0.02, "\n".join(legend_lines),
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9),
            verticalalignment="bottom", horizontalalignment="right")

    out = base / "output" / "trajs.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

    # pass/fail: 3 droplets with the expected launch radii, shrinking
    # monotonically, staying inside the domain box
    failed = False
    r0_sim = sorted(r0_per_id[p] * 1e3 for p in ids)
    ok = len(ids) == N_DROPLETS and np.allclose(r0_sim, R0_MM, rtol=1e-6)
    failed |= not ok
    print(f"  launch radii {[f'{r:.2f}' for r in r0_sim]} mm "
          f"(expect {R0_MM}) {'ok' if ok else 'FAIL'}")
    for k, pid in enumerate(ids):
        m = data["id"] == pid
        r = data["radius"][m]
        ok = bool(np.all(np.diff(r) <= 1e-12))
        failed |= not ok
        print(f"  drop {k+1} radius monotonic non-increasing "
              f"({r[0]*1e3:.3f} -> {r[-1]*1e3:.4f} mm) {'ok' if ok else 'FAIL'}")
    inside = (np.all(Z_dr >= BOX[0]) and np.all(Z_dr <= BOX[1])
              and np.all(R_dr >= BOX[2]) and np.all(R_dr <= BOX[3]))
    failed |= not inside
    print(f"  all positions inside domain {'ok' if inside else 'FAIL'}")
    print("PASS" if not failed else "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
