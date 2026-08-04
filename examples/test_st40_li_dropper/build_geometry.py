#!/usr/bin/env python3
"""Build the ST40 wall surface for OpenEdge from the EIRENE wall — AXI layout.

The wall is parsed from SOLPS run's EIRENE input file (`input.dat`, block 3b
"additional surfaces"): 139 two-point segments in cm, chained into a single
watertight closed loop. This is the exact wall EIRENE's neutrals interact
with, so OpenEdge and SOLPS see the same geometry (the DivGeo `.ogr` limiter
template and the legacy ST40wall.xlsx agree with it to ~mm except two limiter
tabs EIRENE also drops).

True axisymmetric SPARTA mode (`boundary o ao p`): x = Z [m], y = R [m], so
the surf file is written with (x, y) = (Z, R) and CCW enforced in that plane
(normals into the vessel).

The Li powder dropper is not a separate surface: the deck groups the wall
segments on the upper dome (Z ~ 0.83-0.86 m, R ~ 0.47-0.55 m, per the ST40
team); this script prints their line ids for `group dropper surf id <> lo hi`
in in.st40.
"""
from __future__ import annotations
from pathlib import Path
import math
import re
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT_DAT = Path("/Users/42d/Downloads/13589/run_dir/input.dat")
WALL_SURF = ROOT / "input" / "st40_wall_axi.surf"

# Physical location of the Li-powder-dropper wall segments (per ST40 team:
# slanted upper-dome section, not the Z=0.5484 top shelf).
DROP_Z = (0.830, 0.857)  # m
DROP_R = (0.465, 0.550)  # m

FLOAT = re.compile(r"[-+]?\d+\.\d+E[+-]\d+")


def read_eirene_wall(input_dat: Path):
    """Parse block 3b additional surfaces -> list of ((R1,Z1),(R2,Z2)) [m]."""
    lines = input_dat.read_text().splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.startswith("*** 3b"))
    nsurf = int(lines[i0 + 1].split()[0])
    segs = []
    i = i0 + 2
    while len(segs) < nsurf and i < len(lines):
        if lines[i].startswith("***"):
            break
        v = FLOAT.findall(lines[i])
        # RLB line: R1 Z1 zmin R2 Z2 zmax with z = +-1e20 (infinite toroidal)
        if len(v) == 6 and abs(float(v[2])) > 1e19 and abs(float(v[5])) > 1e19:
            x1, y1, _, x2, y2, _ = map(float, v)
            segs.append(((x1 / 100.0, y1 / 100.0), (x2 / 100.0, y2 / 100.0)))
        i += 1
    if len(segs) != nsurf:
        raise RuntimeError(f"parsed {len(segs)} of {nsurf} surfaces in {input_dat}")
    return segs


def chain_loop(segs, tol=1e-6):
    """Chain 2-point segments into a single closed loop of (R, Z) points."""
    def key(p):
        return (round(p[0], 6), round(p[1], 6))

    adj = {}
    for a, b in segs:
        adj.setdefault(key(a), []).append(key(b))
        adj.setdefault(key(b), []).append(key(a))
    bad = [k for k, v in adj.items() if len(v) != 2]
    if bad:
        raise RuntimeError(f"wall is not a single closed loop; open at {bad[:4]}")

    start = key(segs[0][0])
    loop = [start, adj[start][0]]
    while loop[-1] != start:
        nxt = [v for v in adj[loop[-1]] if v != loop[-2]]
        loop.append(nxt[0])
    return [(float(r), float(z)) for r, z in loop[:-1]]


def signed_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def write_loop_surf(path, pts, header):
    n = len(pts)
    with open(path, "w") as f:
        f.write(f"{header}\n\n")
        f.write(f"{n} points\n{n} lines\n\nPoints\n\n")
        for i, (x, y) in enumerate(pts):
            f.write(f"{i+1} {x:.8f} {y:.8f}\n")
        f.write("\nLines\n\n")
        for i in range(n):
            f.write(f"{i+1} {i+1} {(i % n) + 2 if i < n-1 else 1}\n")
    print(f"wrote {path}: {n} points / {n} lines (closed loop, watertight)")


def main(input_dat: Path | str = INPUT_DAT):
    segs = read_eirene_wall(Path(input_dat))
    pts_rz = chain_loop(segs)
    print(f"EIRENE wall: {len(segs)} segments -> {len(pts_rz)}-point closed loop")

    # Axi layout: SPARTA (x, y) = (Z, R); CCW in the written plane = normals
    # into the vessel interior.
    pts = [(zz, rr) for (rr, zz) in pts_rz]
    if signed_area(pts) < 0:
        pts = pts[::-1]
        print("reversed loop to CCW ordering in the (Z, R) plane")

    write_loop_surf(WALL_SURF, pts,
                    "# ST40 EIRENE wall (input.dat block 3b) - axisymmetric"
                    " layout (x=Z, y=R) - watertight closed loop")

    xx = np.array([p[0] for p in pts])   # Z
    yy = np.array([p[1] for p in pts])   # R
    print(f"  wall Z in [{xx.min():.4f}, {xx.max():.4f}] m, "
          f"R in [{yy.min():.4f}, {yy.max():.4f}] m")

    n = len(pts)
    ids = []
    for i in range(n):
        z1, r1 = pts[i]
        z2, r2 = pts[(i + 1) % n]
        zm, rm = 0.5 * (z1 + z2), 0.5 * (r1 + r2)
        if DROP_Z[0] <= zm <= DROP_Z[1] and DROP_R[0] <= rm <= DROP_R[1]:
            ids.append(i + 1)
    if ids:
        print(f"  dropper segments (Z~{DROP_Z}, R in {DROP_R}): "
              f"line ids {ids} -> use `group dropper surf id <> "
              f"{min(ids)} {max(ids)}` in in.st40")
    else:
        print("  WARNING: no wall segments matched the dropper location")
    return ids


if __name__ == "__main__":
    main()
