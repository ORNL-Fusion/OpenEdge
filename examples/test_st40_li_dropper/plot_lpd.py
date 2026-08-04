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


def plot_density(step=None):
    """Per-species Li density maps from output/li_dens.

    New-format dumps carry 4 columns (Li, Li+, Li2+, Li3+) -> 5 panels
    including the total; legacy single-column dumps plot the total only.
    step=None picks the frame with the highest total Li inventory (the
    pulse peak) — the last frame is usually the post-pulse decay.
    """
    snaps = snapshots(ROOT / "output" / "li_dens")
    ncols = len(snaps[0][1]) - 3                      # id xc yc + data cols
    labels = (["Li", "Li$^+$", "Li$^{2+}$", "Li$^{3+}$"][:ncols]
              if ncols > 1 else [])
    totals = [(s, sum(sum(float(v) for v in r[3:]) for r in rows))
              for s, hdr, rows in snaps]
    if step is None:
        step = max(totals, key=lambda t: t[1])[0]
        print("frames (step, total):",
              [(s, f"{t:.2e}") for s, t in totals[-8:]], "-> plotting", step)
    step, hdr, rows = next(s for s in snaps if s[0] == step)

    Zc = np.array([float(r[1]) for r in rows])
    Rc = np.array([float(r[2]) for r in rows])
    data = np.array([[float(v) for v in r[3:]] for r in rows])  # (ncell, ncols)

    nx, ny = 130, 72
    ix = np.clip(((Zc + 1.0) / (2.0 / nx)).astype(int), 0, nx - 1)
    iy = np.clip((Rc / (1.05 / ny)).astype(int), 0, ny - 1)

    def grid(v):
        img = np.full((ny, nx), np.nan)
        cnt = np.zeros((ny, nx))
        for i, j, val in zip(iy, ix, v):
            img[i, j] = (0.0 if np.isnan(img[i, j]) else img[i, j]) + val
            cnt[i, j] += 1
        return np.where(cnt > 0, img / np.maximum(cnt, 1), np.nan)

    fields = [(lab, grid(data[:, k])) for k, lab in enumerate(labels)]
    fields.append(("total Li", grid(data.sum(axis=1))))

    vmax = max(np.nanmax(img) for _, img in fields)
    norm = LogNorm(vmin=max(vmax * 1e-4, 1e10), vmax=vmax)  # shared scale
    Rg = np.linspace(0.0, 1.05, ny + 1)
    Zg = np.linspace(-1.0, 1.0, nx + 1)
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 6.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (lab, img) in zip(axes, fields):
        pm = ax.pcolormesh(0.5 * (Rg[:-1] + Rg[1:]), 0.5 * (Zg[:-1] + Zg[1:]),
                           img.T, norm=norm, cmap="viridis", shading="auto")
        draw_vessel(ax)
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        ax.set_title(lab, fontsize=10)
    axes[0].set_ylabel("Z [m]")
    shot_ms = step * 2.0e-6 * 1e3 - 200.0
    fig.suptitle(f"Li density — step {step} (shot {shot_ms:+.0f} ms)",
                 fontsize=11, y=0.98)
    cb = fig.colorbar(pm, ax=list(axes), shrink=0.8, pad=0.02)
    cb.set_label(r"$n$ [m$^{-3}$]")
    out = ROOT / "st40_li_density.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out, f"| step {step} | n max = {vmax:.3e} m^-3")


def plot_history(logfile=None):
    """Per-species inventory vs time from the stats block in log.openedge.

    Needs stats_style step cpu np c_cnt[1..5] (grain, Li, Li+, Li2+, Li3+).
    Time axis is shot time (matches Erin's IPD plots): sim starts at
    shot -200 ms, the powder's in-vessel flight time — Li appears near 0.
    """
    path = Path(logfile) if logfile else ROOT / "log.openedge"
    rows = []
    for l in path.read_text().splitlines():
        p = l.split()
        if len(p) == 8:
            try:
                rows.append([float(v) for v in p])
            except ValueError:
                pass
    if not rows:
        print(f"no 8-column stats lines in {path} — run with the c_cnt "
              "stats_style first")
        return
    a = np.array(rows)
    t = a[:, 0] * 2.0e-6 * 1e3 - 200.0        # shot time [ms]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
    ax1.plot(t, a[:, 3], color="0.35", lw=1.4, label="grain")
    ax1.set_ylabel("grain macroparticles")
    ax1.legend(fontsize=8)
    for lab, y, c in [("Li", a[:, 4], "red"), ("Li$^+$", a[:, 5], "limegreen"),
                      ("Li$^{2+}$", a[:, 6], "seagreen"),
                      ("Li$^{3+}$", a[:, 7], "darkgreen")]:
        ax2.plot(t, y, color=c, lw=1.2, label=lab)
    ax2.set_ylabel("Li macroparticles")
    ax2.set_xlabel("shot time [ms]")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.axvline(0.0, color="0.6", lw=0.8, ls="--")
    ax1.text(0.0, ax1.get_ylim()[1] * 0.95, "  first Li at plasma",
             fontsize=7, color="0.4", va="top")
    ax1.set_title("species inventory vs time  (t < 0: powder in flight)",
                  fontsize=11)
    plt.tight_layout()
    out = ROOT / "st40_li_history.png"
    plt.savefig(out, dpi=120)
    print("wrote", out, f"| {len(rows)} stats samples")


if __name__ == "__main__":
    main()
    plot_density()
