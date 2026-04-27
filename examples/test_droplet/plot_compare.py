#!/usr/bin/env python3
"""Overlay droplet trajectory vs. pure-gravity prediction.

Reads a SPARTA particle dump (default `case_slow.outer`), extracts each
droplet's (R, Z, vR, vZ, vphi) time series in the axi slot convention
(x_slot = Z, y_slot = R, z_slot = phi), and overlays:

  - left: (R, Z) trajectory observed (solid) vs. gravity-only parabola
    (dashed), with the wall outline.
  - right (4 panels): vZ(t), vR(t), vphi(t), |v_total|(t) observed vs.
    gravity-only prediction.

Deviation between solid and dashed = net non-gravity force (drag, rocket,
charging, sheath).

Usage:  python3 plot_compare.py [dump_path]
        python3 plot_compare.py case.outer
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

GRAVITY_Z = -9.8  # m/s^2 along Z slot (axi: gravity input "0 -9.8 0" -> slot[0])
DT_PER_STEP = 1.0e-5


def parse_dump(path):
    """Return dict id -> {t, Z, R, vZ, vR, vphi, T, r} arrays."""
    with open(path) as f:
        lines = f.read().splitlines()
    per_id = {}
    i, t, n, col = 0, None, 0, None
    while i < len(lines):
        s = lines[i].strip()
        if s == "ITEM: TIMESTEP":
            t = int(lines[i + 1]); i += 2; continue
        if s == "ITEM: NUMBER OF ATOMS":
            n = int(lines[i + 1]); i += 2; continue
        if s.startswith("ITEM: BOX BOUNDS"):
            i += 4; continue
        if s.startswith("ITEM: ATOMS"):
            hdr = s.split()[2:]
            col = {k: j for j, k in enumerate(hdr)}
            for k in range(n):
                tok = lines[i + 1 + k].split()
                pid = int(tok[col["id"]])
                row = per_id.setdefault(
                    pid, dict(t=[], Z=[], R=[], vZ=[], vR=[], vphi=[], T=[], r=[]))
                row["t"].append(t * DT_PER_STEP)
                row["Z"].append(float(tok[col["x"]]))
                row["R"].append(float(tok[col["y"]]))
                row["vZ"].append(float(tok[col["vx"]]))
                row["vR"].append(float(tok[col["vy"]]))
                row["vphi"].append(float(tok[col["vz"]]))
                row["T"].append(float(tok[col["temp"]]) if "temp" in col else 0.0)
                row["r"].append(float(tok[col["radius"]]) if "radius" in col else 0.0)
            i += 1 + n; continue
        i += 1
    return {pid: {k: np.asarray(v) for k, v in row.items()}
            for pid, row in per_id.items()}


def parse_wall(path):
    """Return Nx2 array of (Z, R) wall vertices for plotting."""
    pts, in_pts = [], False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s == "Points":
                in_pts = True; continue
            if s == "Lines":
                in_pts = False; continue
            if in_pts and s:
                parts = s.split()
                if len(parts) >= 3:
                    pts.append((float(parts[1]), float(parts[2])))
    return np.asarray(pts) if pts else np.empty((0, 2))


def gravity_prediction(t, Z0, R0, vZ0, vR0, vph0):
    """Pure-gravity ballistic trajectory at -g along Z slot."""
    return dict(
        Z=Z0 + vZ0 * t + 0.5 * GRAVITY_Z * t * t,
        R=R0 + vR0 * t,
        vZ=vZ0 + GRAVITY_Z * t,
        vR=np.full_like(t, vR0),
        vphi=np.full_like(t, vph0),
    )


def main():
    base = Path(__file__).resolve().parent
    dump_path = base / (sys.argv[1] if len(sys.argv) > 1 else "case_slow.outer")
    if not dump_path.exists():
        sys.exit(f"no dump at {dump_path} — run in.droplet_slow first")

    drops = parse_dump(dump_path)
    if not drops:
        sys.exit(f"{dump_path} has no particle records")

    wall = parse_wall(base / "wall.surf")

    fig = plt.figure(figsize=(13, 7), dpi=140)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 1], hspace=0.4, wspace=0.28)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_vZ = fig.add_subplot(gs[0, 1])
    ax_vR = fig.add_subplot(gs[1, 1], sharex=ax_vZ)
    ax_vp = fig.add_subplot(gs[2, 1], sharex=ax_vZ)
    ax_vm = fig.add_subplot(gs[3, 1], sharex=ax_vZ)

    if wall.size:
        ax_xy.plot(wall[:, 1], wall[:, 0], "-", lw=1.0, color="0.4", label="wall")
    ax_xy.set_xlabel(r"$R$ (m)")
    ax_xy.set_ylabel(r"$Z$ (m)")
    ax_xy.set_title(f"observed vs gravity-only (dashed)  —  {dump_path.name}")
    ax_xy.set_aspect("equal")
    ax_xy.grid(alpha=0.3)

    pids_sorted = sorted(drops.keys(), key=lambda p: drops[p]["r"].max())
    cmap = plt.get_cmap("tab10")
    for k, pid in enumerate(pids_sorted):
        d = drops[pid]
        c = cmap(k)
        label = f"drop {k+1}  r$_0$={d['r'].max()*1e3:.1f} mm"

        ax_xy.plot(d["R"], d["Z"], "-", lw=1.8, color=c, label=label)
        ax_xy.plot(d["R"][0], d["Z"][0], "o", ms=8, color=c, mec="k")
        ax_xy.plot(d["R"][-1], d["Z"][-1], "*", ms=14, color=c, mec="k")

        pred = gravity_prediction(d["t"] - d["t"][0],
                                  d["Z"][0], d["R"][0],
                                  d["vZ"][0], d["vR"][0], d["vphi"][0])
        ax_xy.plot(pred["R"], pred["Z"], "--", lw=1.2, color=c, alpha=0.7)

        ax_vZ.plot(d["t"], d["vZ"], "-", lw=1.6, color=c, label=label)
        ax_vZ.plot(d["t"], pred["vZ"], "--", lw=1.0, color=c, alpha=0.7)
        ax_vR.plot(d["t"], d["vR"], "-", lw=1.6, color=c)
        ax_vR.plot(d["t"], pred["vR"], "--", lw=1.0, color=c, alpha=0.7)
        ax_vp.plot(d["t"], d["vphi"], "-", lw=1.6, color=c)
        ax_vp.plot(d["t"], pred["vphi"], "--", lw=1.0, color=c, alpha=0.7)

        v_obs = np.sqrt(d["vZ"]**2 + d["vR"]**2 + d["vphi"]**2)
        v_pre = np.sqrt(pred["vZ"]**2 + pred["vR"]**2 + pred["vphi"]**2)
        ax_vm.plot(d["t"], v_obs, "-", lw=1.6, color=c)
        ax_vm.plot(d["t"], v_pre, "--", lw=1.0, color=c, alpha=0.7)

    ax_xy.legend(fontsize=9, loc="best")

    ax_vZ.set_ylabel(r"$v_Z$ (m/s)"); ax_vZ.grid(alpha=0.3)
    ax_vZ.legend(fontsize=8, loc="best")
    plt.setp(ax_vZ.get_xticklabels(), visible=False)
    ax_vR.set_ylabel(r"$v_R$ (m/s)"); ax_vR.grid(alpha=0.3)
    plt.setp(ax_vR.get_xticklabels(), visible=False)
    ax_vp.set_ylabel(r"$v_\phi$ (m/s)"); ax_vp.grid(alpha=0.3)
    plt.setp(ax_vp.get_xticklabels(), visible=False)
    ax_vm.set_ylabel(r"$|v|$ (m/s)"); ax_vm.grid(alpha=0.3)
    ax_vm.set_xlabel(r"$t$ (s)")

    fig.suptitle(
        "droplet trajectory: observed (solid) vs gravity-only ballistic (dashed)",
        fontsize=12, y=0.995)
    out = base / f"compare_{dump_path.stem}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

    print()
    print("Final-frame deviation from gravity-only (force-driven bend):")
    for k, pid in enumerate(pids_sorted):
        d = drops[pid]
        pred = gravity_prediction(d["t"] - d["t"][0],
                                  d["Z"][0], d["R"][0],
                                  d["vZ"][0], d["vR"][0], d["vphi"][0])
        dZ  = d["Z"][-1]    - pred["Z"][-1]
        dR  = d["R"][-1]    - pred["R"][-1]
        dvZ = d["vZ"][-1]   - pred["vZ"][-1]
        dvR = d["vR"][-1]   - pred["vR"][-1]
        dvp = d["vphi"][-1] - pred["vphi"][-1]
        print(f"  drop {k+1} (r0 = {d['r'].max()*1e3:.1f} mm)  "
              f"t = {d['t'][-1]:.4f} s")
        print(f"    Δ position:  ΔZ = {dZ:+.4f} m   ΔR = {dR:+.4f} m")
        print(f"    Δ velocity:  ΔvZ = {dvZ:+.3f}   ΔvR = {dvR:+.3f}   "
              f"Δvphi = {dvp:+.3f}  m/s")


if __name__ == "__main__":
    main()
