#!/usr/bin/env python3
"""Afonin-style figure for the WEST boron drop (cf. Afonin NF 63 (2023)
fig. 3): grain trajectories colored by the local evaporation flux, over
the WEST wall and core contours, plus the B/B+ cloud at the last frame.

Axi layout: dumps store (x, y) = (Z, R); plotted in physical (R, Z).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
WEST = ROOT / ".." / "test_west_cases" / "axi_1p5MW_plasma_bkg" / "input"

# boron sublimation flux (matches grain_material.cpp: log10 p[atm] = a+b/T)
A_ANT, B_ANT, M_AMU = 8.64, -32030.0, 10.81
def gevap(T):
    T = np.maximum(T, 1.0)
    p_mmHg = 760.0 * 10.0**(A_ANT + B_ANT / T)
    return 1.0e4 * 3.513e22 * p_mmHg / np.sqrt(M_AMU * T)   # atoms/m^2/s


def surf_loop(path):
    pts = {}; mode = None
    for s in Path(path).read_text().splitlines():
        t = s.split()
        if not t: continue
        if t[0] == "Points": mode = "p"; continue
        if t[0] == "Lines": mode = None; continue
        if mode == "p" and t[0].isdigit():
            pts[int(t[0])] = (float(t[2]), float(t[1]))     # (R, Z)
    xy = [pts[i] for i in sorted(pts)] + [pts[1]]
    return np.array(xy)


def snapshots(path):
    L = Path(path).read_text().splitlines()
    out = []
    for k, l in enumerate(L):
        if not l.startswith("ITEM: TIMESTEP"): continue
        step = int(L[k + 1]); n = int(L[k + 3])
        hdr = L[k + 8].split()[2:]
        rows = [L[k + 9 + j].split() for j in range(n)]
        out.append((step, hdr, rows))
    return out


def main():
    W = surf_loop(WEST / "wall_fine.surf")
    C = surf_loop(WEST / "core.surf")

    snaps = snapshots(ROOT / "output" / "state")
    hdr = snaps[0][1]
    ii, ti = hdr.index("id"), hdr.index("type")
    xi, yi = hdr.index("x"), hdr.index("y")
    ri, Ti = hdr.index("radius"), hdr.index("temp")

    # grain trajectories: id -> list of (R, Z, Gevap)
    traj = {}
    for step, _, rows in snaps:
        for r in rows:
            if int(r[ti]) != 1: continue
            gid = int(r[ii])
            traj.setdefault(gid, []).append(
                (float(r[yi]), float(r[xi]), gevap(float(r[Ti]))))

    step, _, rows = snaps[-1]
    Bn  = np.array([[float(r[yi]), float(r[xi])] for r in rows if int(r[ti]) == 2])
    Bp  = np.array([[float(r[yi]), float(r[xi])] for r in rows if int(r[ti]) == 3])

    fig, ax = plt.subplots(figsize=(6.5, 9))
    ax.plot(W[:, 0], W[:, 1], "-", color="0.2", lw=1.3)
    ax.plot(C[:, 0], C[:, 1], "--", color="royalblue", lw=1.0)
    if len(Bp): ax.scatter(Bp[:, 0], Bp[:, 1], s=1.5, color="limegreen",
                           alpha=0.4, label=f"B$^+$ ({len(Bp)})")
    if len(Bn): ax.scatter(Bn[:, 0], Bn[:, 1], s=3, color="red",
                           alpha=0.5, label=f"B ({len(Bn)})")

    vmax = max(max(g for _, _, g in t) for t in traj.values())
    vmax = max(vmax, 1.0)
    for t in traj.values():
        a = np.array(t)
        sc = ax.scatter(a[:, 0], a[:, 1], c=a[:, 2] / 1e21, s=4,
                        cmap="viridis", vmin=0.0, vmax=vmax / 1e21, zorder=5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(r"$\Gamma_{evap}$ [$10^{21}$ atoms m$^{-2}$ s$^{-1}$]")

    ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(f"WEST boron drop — step {step}\n"
                 "grain trajectories colored by evaporation flux")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    out = ROOT / "b_west_drop.png"
    plt.savefig(out, dpi=130)
    print("wrote", out, f"| {len(traj)} grains, B={len(Bn)}, B+={len(Bp)}")


if __name__ == "__main__":
    main()
