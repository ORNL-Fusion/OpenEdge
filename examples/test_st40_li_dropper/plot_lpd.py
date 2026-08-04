#!/usr/bin/env python3
"""Plots for the ST40 Li-powder-dropper run (axisymmetric layout: surf files
and dumps store (x, y) = (Z, R); everything is swapped to (R, Z) here).

main()          — camera-style picture: grains, Li vapor, Li+/Li2+/Li3+.
plot_density()  — 2D map of the time-averaged total-Li density n_Li(R, Z)
                  from the grid dump (cf. Afonin NF 2023 impurity maps).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent

# wall line ids of the Li dropper segments (printed by build_geometry.py)
DROPPER_IDS = (15, 16)

SPECIES = {1: "grain", 2: "Li", 3: "Li$^+$", 4: "Li$^{2+}$", 5: "Li$^{3+}$"}


def read_surf_pts(path):
    pts = {}; mode = None
    for l in open(path):
        s = l.strip()
        if s.lower() == "points": mode = "p"; continue
        if s.lower() == "lines":  mode = "l"; continue
        if mode == "p" and s and s[0].isdigit():
            a = s.split()
            pts[int(a[0])] = (float(a[2]), float(a[1]))   # (Z, R) -> (R, Z)
    return pts


def surf_loop(path):
    pts = read_surf_pts(path)
    xy = [pts[i] for i in sorted(pts)] + [pts[1]]
    return [p[0] for p in xy], [p[1] for p in xy]


def snapshots(path):
    L = Path(path).read_text().splitlines()
    idx = [k for k, l in enumerate(L) if l.startswith("ITEM: TIMESTEP")]
    out = []
    for k in idx:
        a = next(m for m in range(k, len(L))
                 if L[m].startswith(("ITEM: ATOMS", "ITEM: CELLS")))
        end = next((m for m in range(a + 1, len(L)) if L[m].startswith("ITEM:")), len(L))
        hdr = L[a].split()[2:]
        rows = [r.split() for r in L[a + 1:end] if r.strip()]
        out.append((int(L[k + 1]), hdr, rows))
    return out


def draw_vessel(ax):
    wx, wy = surf_loop(ROOT / "input" / "st40_wall_axi.surf")
    cx, cy = surf_loop(ROOT / "input" / "st40_core.surf")
    ax.plot(wx, wy, "-", color="0.2", lw=1.4, zorder=5)
    ax.plot(cx, cy, "--", color="royalblue", lw=1.0, zorder=4)


def main():
    wall = read_surf_pts(ROOT / "input" / "st40_wall_axi.surf")
    drp = [wall[i] for i in range(DROPPER_IDS[0], DROPPER_IDS[1] + 2)]

    step, hdr, rows = snapshots(ROOT / "output" / "state")[-1]   # last frame
    ti, xi, yi = hdr.index("type"), hdr.index("x"), hdr.index("y")
    P = {t: np.array([[float(r[yi]), float(r[xi])] for r in rows if int(r[ti]) == t])
         for t in SPECIES}

    fig, ax = plt.subplots(figsize=(4.2, 6.5))
    draw_vessel(ax)
    ax.plot([p[0] for p in drp], [p[1] for p in drp], "-",
            color="black", lw=3, zorder=6, label="Li dropper")
    style = {3: dict(s=1.5, color="limegreen", alpha=0.35),
             4: dict(s=1.5, color="seagreen", alpha=0.5),
             5: dict(s=2.0, color="darkgreen", alpha=0.7),
             2: dict(s=8, color="red", alpha=0.8),
             1: dict(s=10, color="silver", edgecolor="0.3", lw=0.3)}
    for t in (3, 4, 5, 2, 1):
        if len(P[t]):
            ax.scatter(P[t][:, 0], P[t][:, 1], zorder=2 + 0.1 * t,
                       label=SPECIES[t] + f" ({len(P[t])})", **style[t])
    ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(f"ST40 Li powder dropper — step {step}", fontsize=10)
    ax.legend(loc="lower left", fontsize=6, framealpha=0.9)
    plt.tight_layout()
    out = ROOT / "st40_lpd_chain.png"
    plt.savefig(out, dpi=120)
    counts = {SPECIES[t]: len(P[t]) for t in SPECIES}
    print("wrote", out, "| last step", step, "|", counts)


def plot_density():
    """2D map of the time-averaged total-Li density from output/li_dens."""
    step, hdr, rows = snapshots(ROOT / "output" / "li_dens")[-1]
    xc, yc, dv = hdr.index("xc"), hdr.index("yc"), len(hdr) - 1
    Zc = np.array([float(r[xc]) for r in rows])
    Rc = np.array([float(r[yc]) for r in rows])
    n = np.array([float(r[dv]) for r in rows])

    # uniform 130 x 72 grid over Z [-1, 1], R [0, 1.05]
    nx, ny = 130, 72
    ix = np.clip(((Zc + 1.0) / (2.0 / nx)).astype(int), 0, nx - 1)
    iy = np.clip((Rc / (1.05 / ny)).astype(int), 0, ny - 1)
    img = np.full((ny, nx), np.nan)
    cnt = np.zeros((ny, nx))
    for i, j, v in zip(iy, ix, n):
        img[i, j] = (0.0 if np.isnan(img[i, j]) else img[i, j]) + v
        cnt[i, j] += 1
    img = np.where(cnt > 0, img / np.maximum(cnt, 1), np.nan)

    Zg = np.linspace(-1.0, 1.0, nx + 1)
    Rg = np.linspace(0.0, 1.05, ny + 1)
    vmax = np.nanmax(img)
    fig, ax = plt.subplots(figsize=(4.2, 6.5))
    pm = ax.pcolormesh(0.5 * (Rg[:-1] + Rg[1:]), 0.5 * (Zg[:-1] + Zg[1:]), img.T,
                       norm=LogNorm(vmin=max(vmax * 1e-4, 1e10), vmax=vmax),
                       cmap="viridis", shading="auto")
    draw_vessel(ax)
    ax.set_aspect("equal"); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(f"Total Li density — step {step}", fontsize=10)
    fig.colorbar(pm, ax=ax, label=r"$n_{Li}$ [m$^{-3}$]")
    plt.tight_layout()
    out = ROOT / "st40_li_density.png"
    plt.savefig(out, dpi=120)
    print("wrote", out, f"| step {step} | n_Li max = {vmax:.3e} m^-3")


if __name__ == "__main__":
    main()
    plot_density()
