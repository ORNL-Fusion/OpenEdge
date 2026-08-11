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
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent
WEST = ROOT / ".." / "test_west_cases" / "input"

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
    vmax = max(vmax, 1e21)
    norm = LogNorm(vmin=vmax * 1e-8, vmax=vmax)
    for t in traj.values():
        a = np.array(t)
        sc = ax.scatter(a[:, 0], a[:, 1], c=np.maximum(a[:, 2], vmax * 1e-8),
                        s=4, cmap="viridis", norm=norm, zorder=5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(r"$\Gamma_{evap}$ [atoms m$^{-2}$ s$^{-1}$]")

    ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(f"WEST boron drop — step {step}\n"
                 "grain trajectories colored by evaporation flux")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    out = ROOT / "b_west_drop.png"
    plt.savefig(out, dpi=130)
    print("wrote", out, f"| {len(traj)} grains, B={len(Bn)}, B+={len(Bp)}")


def plot_density():
    """2D map of the time-averaged total-B density from output/b_dens."""
    L = (ROOT / "output" / "b_dens").read_text().splitlines()
    idx = [i for i, l in enumerate(L) if l.startswith("ITEM: TIMESTEP")]
    i = idx[-1]
    step = int(L[i + 1])
    a = next(m for m in range(i, len(L)) if L[m].startswith("ITEM: CELLS"))
    end = next((m for m in range(a + 1, len(L)) if L[m].startswith("ITEM:")), len(L))
    rows = [r.split() for r in L[a + 1:end] if r.strip()]
    Zc = np.array([float(r[1]) for r in rows])
    Rc = np.array([float(r[2]) for r in rows])
    v  = np.array([float(r[3]) for r in rows])

    # deck grid: 200 x 150 over Z [-1, 0.85], R [0, 3.25]
    nx, ny = 200, 150
    ix = np.clip(((Zc + 1.0) / (1.85 / nx)).astype(int), 0, nx - 1)
    iy = np.clip((Rc / (3.25 / ny)).astype(int), 0, ny - 1)
    img = np.full((ny, nx), np.nan); cnt = np.zeros((ny, nx))
    for a_, b_, val in zip(iy, ix, v):
        img[a_, b_] = (0.0 if np.isnan(img[a_, b_]) else img[a_, b_]) + val
        cnt[a_, b_] += 1
    img = np.where(cnt > 0, img / np.maximum(cnt, 1), np.nan)

    W = surf_loop(WEST / "wall_fine.surf")
    C = surf_loop(WEST / "core.surf")
    Zg = np.linspace(-1.0, 0.85, nx + 1)
    Rg = np.linspace(0.0, 3.25, ny + 1)
    vmax = np.nanmax(img)
    fig, ax = plt.subplots(figsize=(6.0, 8.5))
    pm = ax.pcolormesh(0.5*(Rg[:-1]+Rg[1:]), 0.5*(Zg[:-1]+Zg[1:]), img.T,
                       norm=LogNorm(vmin=max(vmax*1e-4, 1e8), vmax=vmax),
                       cmap="viridis", shading="auto")
    ax.plot(W[:, 0], W[:, 1], "-", color="0.2", lw=1.3)
    ax.plot(C[:, 0], C[:, 1], "--", color="w", lw=1.0)
    ax.set_xlim(W[:, 0].min() - 0.05, W[:, 0].max() + 0.05)
    ax.set_ylim(W[:, 1].min() - 0.05, W[:, 1].max() + 0.05)
    ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(f"Total B density — step {step}", fontsize=11)
    fig.colorbar(pm, ax=ax, label=r"$n_B$ [m$^{-3}$]")
    plt.tight_layout()
    out = ROOT / "b_west_density.png"
    plt.savefig(out, dpi=130)
    print("wrote", out, f"| step {step} | n_B max = {vmax:.3e} m^-3")


if __name__ == "__main__":
    main()
    plot_density()
