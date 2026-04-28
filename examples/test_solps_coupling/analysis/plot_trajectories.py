#!/usr/bin/env python3
"""Plot droplet trajectories from a SPARTA particle dump in (R, Z).

Default dump: ../coupled_test/openedge/output/particles.txt
Usage:        python3 plot_trajectories.py [dump_path]

Slot order: axi (x = Z, y = R) — auto-detected from ITEM: BOX BOUNDS.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def parse_dump(path):
    blocks = {"t": [], "id": [], "x": [], "y": [], "radius": []}
    is_axi = None
    with open(path) as f:
        lines = f.read().splitlines()
    i, t = 0, None
    while i < len(lines):
        s = lines[i].strip()
        if s == "ITEM: TIMESTEP":
            t = int(lines[i+1]); i += 2; continue
        if s == "ITEM: NUMBER OF ATOMS":
            n = int(lines[i+1]); i += 2; continue
        if s.startswith("ITEM: BOX BOUNDS"):
            tokens = s.split()[3:]
            if is_axi is None and len(tokens) >= 2:
                is_axi = "a" in tokens[1]
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
    out = {k: np.asarray(v) for k, v in blocks.items()}
    out["axi"] = bool(is_axi)
    return out


def parse_wall(path):
    pts, in_pts = [], False
    if not path.exists():
        return np.empty((0, 2))
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s == "Points":
                in_pts = True; continue
            if s == "Lines":
                in_pts = False; continue
            if in_pts and s and s[0].isdigit():
                parts = s.split()
                if len(parts) >= 3:
                    pts.append((float(parts[1]), float(parts[2])))
    return np.asarray(pts)


def main():
    base = Path(__file__).resolve().parent
    project = base.parent
    dump_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        project / "coupled_test" / "openedge" / "output" / "particles.txt"
    if not dump_path.exists():
        sys.exit(f"no dump at {dump_path}")

    data = parse_dump(dump_path)
    if data["axi"]:
        # axi slot: x = Z, y = R
        Z = data["x"]; R = data["y"]
    else:
        # 2D Cart slot: x = R, y = Z
        R = data["x"]; Z = data["y"]

    wall = parse_wall(project / "input" / "wall.surf")
    if wall.size:
        # wall.surf in axi format stores (Z, R); same swap rule applies.
        if data["axi"]:
            wall_R = wall[:, 1]; wall_Z = wall[:, 0]
        else:
            wall_R = wall[:, 0]; wall_Z = wall[:, 1]
    else:
        wall_R = wall_Z = None

    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    if wall_R is not None:
        ax.plot(wall_R, wall_Z, "-", lw=1.0, color="0.4", label="vessel wall")
    ax.set_xlabel(r"$R$ (m)")
    ax.set_ylabel(r"$Z$ (m)")
    ax.set_title(f"Droplet trajectories  —  {dump_path.name}")
    ax.set_aspect("equal")

    R_tr = R[np.isfinite(R)]; Z_tr = Z[np.isfinite(Z)]
    if R_tr.size > 1:
        Rmin, Rmax = R_tr.min(), R_tr.max()
        Zmin, Zmax = Z_tr.min(), Z_tr.max()
        span = max(Rmax - Rmin, Zmax - Zmin, 0.05)
        Rc, Zc = 0.5 * (Rmin + Rmax), 0.5 * (Zmin + Zmax)
        ax.set_xlim(Rc - 0.75 * span, Rc + 0.75 * span)
        ax.set_ylim(Zc - 0.75 * span, Zc + 0.75 * span)

    ids = np.unique(data["id"])
    cmap = plt.get_cmap("plasma")

    r0_per_id = {}
    for pid in ids:
        m = data["id"] == pid
        rs = data["radius"][m]
        rs = rs[rs > 0]
        r0_per_id[pid] = rs[0] if rs.size else 0.0

    lc_last = None
    for pid in ids:
        m = data["id"] == pid
        r0 = r0_per_id[pid]
        if r0 == 0:
            continue
        frac = np.where(data["radius"][m] > 0, data["radius"][m] / r0, np.nan)
        Rseg, Zseg = R[m], Z[m]
        pts = np.column_stack([Rseg, Zseg]).reshape(-1, 1, 2)
        if pts.shape[0] < 2:
            continue
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap,
                            norm=plt.Normalize(0, 1), lw=1.5)
        lc.set_array(frac[:-1])
        ax.add_collection(lc)
        lc_last = lc

    if lc_last is not None:
        cb = fig.colorbar(lc_last, ax=ax, shrink=0.85)
        cb.set_label(r"$r(t) / r_0$")

    out = base / "trajs.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  layout: {'axi (x=Z, y=R)' if data['axi'] else '2D Cart (x=R, y=Z)'}")
    print(f"  particles: {len(ids)}, "
          f"radii: {min(r0_per_id.values())*1e3:.1f}–{max(r0_per_id.values())*1e3:.1f} mm")


if __name__ == "__main__":
    main()
