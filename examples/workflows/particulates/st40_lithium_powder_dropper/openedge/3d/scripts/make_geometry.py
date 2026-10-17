#!/usr/bin/env python3
"""Revolve the 2-D ST40 wall and core contours (../input/*.surf, axi layout
(Z, R)) into full 360 deg tori for the 3-D case.

  (R, Z, phi) -> (x, y, z) = (R cos phi, R sin phi, Z)

Output (3d/input/):
  st40_wall_3d.surf / .stl   vessel wall, normals into the vessel
  st40_core_3d.surf / .stl   core sink torus, normals outward into the SOL
  st40_dropper.stl           the dropper patch, for viewing
  st40_box.inc               create_box / create_grid
  st40_groups.inc            group dropper (2-D wall lines 16-17, strips at phi = dropper_phi)

Triangle ids: 2-D segment k (0-based, file order) owns [2*NPHI*k+1, 2*NPHI*(k+1)];
strip j of that segment owns the two ids 2*NPHI*k + 2*j + 1, +2.

usage: python3 scripts/make_geometry.py [--nphi 240] [--cell 0.02]
       [--dropper-ids 16 17] [--dropper-phi 0] [--dropper-nstrips 1]
"""
import argparse, math, os
import numpy as np
from matplotlib.path import Path as MplPath

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)                       # openedge/3d
SRC = os.path.join(os.path.dirname(CASE), "input")  # openedge/input


def read_loop(path):
    """Closed 2-D loop in axi layout: returns (R, Z) points in file order."""
    pts, mode = {}, None
    for line in open(path):
        t = line.split()
        if not t: continue
        if t[0] == "Points": mode = "p"; continue
        if t[0] == "Lines": mode = "l"; continue
        if mode == "p" and len(t) == 3 and t[0].isdigit():
            pts[int(t[0])] = (float(t[2]), float(t[1]))      # file is (Z, R)
    return np.array([pts[i] for i in sorted(pts)])


def revolve(RZ, phis, fluid, tid0=0):
    """Triangulate the loop RZ revolved over the periodic angles phis.
    Normals are oriented toward the side where fluid((R,Z)) is True."""
    n, nphi = len(RZ), len(phis)
    points, tris, ids = [], [], []
    ring = np.zeros((n, nphi), dtype=int)
    for k in range(n):
        for j, ph in enumerate(phis):
            points.append((RZ[k, 0] * math.cos(ph), RZ[k, 0] * math.sin(ph), RZ[k, 1]))
            ring[k, j] = len(points)
    tid = tid0
    for k in range(n):
        p, q = RZ[k], RZ[(k + 1) % n]
        m = 0.5 * (p + q); t = q - p; nl = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        s = 1.0 if fluid(m + 1e-4 * nl) else -1.0        # left normal into the fluid?
        rp, rq = ring[k], ring[(k + 1) % n]
        for j in range(nphi):
            j1 = (j + 1) % nphi
            t1 = (rp[j], rq[j], rq[j1]); t2 = (rp[j], rq[j1], rp[j1])
            if s < 0: t1 = t1[::-1]; t2 = t2[::-1]
            for tt in (t1, t2):
                tid += 1; tris.append(tt); ids.append(tid)
    return points, tris, ids


def normals(points, tris):
    P = np.array(points); T = np.array(tris) - 1
    v0, v1, v2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0); area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm /= np.maximum(2 * area, 1e-300)[:, None]
    return P, T, nrm, area, (v0 + v1 + v2) / 3


def check_orientation(name, points, tris, fluid, delta):
    P, T, nrm, area, cen = normals(points, tris)
    probe = cen + delta * nrm
    ok = fluid(np.column_stack((np.hypot(probe[:, 0], probe[:, 1]), probe[:, 2])))
    bad = np.where(~ok)[0]
    if len(bad):
        i = bad[0]
        raise RuntimeError(f"{name}: {len(bad)} triangles point away from the fluid; "
                           f"first id {i+1} centroid {cen[i]} normal {nrm[i]}")
    return P, T, nrm, area


def write_surf(path, header, points, tris, ids):
    with open(path, "w") as f:
        f.write(f"{header}\n\n{len(points)} points\n{len(tris)} triangles\n\nPoints\n\n")
        for i, (x, y, z) in enumerate(points, 1): f.write(f"{i} {x:.8f} {y:.8f} {z:.8f}\n")
        f.write("\nTriangles\n\n")
        for i, t in zip(ids, tris): f.write(f"{i} {t[0]} {t[1]} {t[2]}\n")


def write_stl(path, name, P, T, nrm, sel=None):
    idx = range(len(T)) if sel is None else sel
    with open(path, "w") as f:
        f.write(f"solid {name}\n")
        for k in idx:
            f.write(f" facet normal {nrm[k,0]:.6e} {nrm[k,1]:.6e} {nrm[k,2]:.6e}\n  outer loop\n")
            for v in T[k]: f.write(f"   vertex {P[v,0]:.6e} {P[v,1]:.6e} {P[v,2]:.6e}\n")
            f.write("  endloop\n endfacet\n")
        f.write(f"endsolid {name}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nphi", type=int, default=240, help="toroidal strips over 360 deg")
    ap.add_argument("--cell", type=float, default=0.03, help="grid cell size (m); the chem fix does per-cell work every step, so keep the grid coarse")
    ap.add_argument("--dropper-ids", type=int, nargs=2, default=(16, 17), help="2-D wall line ids")
    ap.add_argument("--dropper-phi", type=float, default=0.0, help="dropper toroidal angle (deg)")
    ap.add_argument("--dropper-nstrips", type=int, default=1, help="toroidal width in strips")
    a = ap.parse_args()
    NPHI, ns = a.nphi, a.dropper_nstrips
    if not 0 < ns < NPHI // 2: raise SystemExit("dropper-nstrips must be < nphi/2")

    wall = read_loop(os.path.join(SRC, "st40_wall_axi.surf"))
    core = read_loop(os.path.join(SRC, "st40_core.surf"))
    wpath, cpath = MplPath(wall), MplPath(core)
    def fluid(rz):
        rz = np.atleast_2d(rz); return wpath.contains_points(rz) & ~cpath.contains_points(rz)

    # strips j0 .. j0+ns-1 are centred on dropper_phi; no wrap-around in that block
    dphi = 2 * math.pi / NPHI
    j0 = NPHI // 2 - ns // 2
    phis = math.radians(a.dropper_phi) + (np.arange(NPHI) - j0 - ns / 2) * dphi

    wp, wt, wi = revolve(wall, phis, fluid)
    cp, ct, ci = revolve(core, phis, fluid)
    Rmax = wall[:, 0].max()
    delta = 4 * Rmax * (1 - math.cos(dphi / 2)) + 1e-4        # clear the chord sag
    WP, WT, WN, WA = check_orientation("wall", wp, wt, fluid, delta)
    CP, CT, CN, CA = check_orientation("core", cp, ct, fluid, delta)

    out = os.path.join(CASE, "input"); os.makedirs(out, exist_ok=True)
    hdr = (f"(x=R cos phi, y=R sin phi, z=Z; {NPHI} strips over 360 deg; "
           f"dropper 2-D lines {a.dropper_ids[0]}-{a.dropper_ids[1]} at phi={a.dropper_phi:g} deg)")
    write_surf(os.path.join(out, "st40_wall_3d.surf"), "# ST40 EIRENE wall revolved " + hdr, wp, wt, wi)
    write_surf(os.path.join(out, "st40_core_3d.surf"), "# ST40 core sink (psi_N 0.615) revolved " + hdr, cp, ct, ci)
    write_stl(os.path.join(out, "st40_wall_3d.stl"), "st40_wall", WP, WT, WN)
    write_stl(os.path.join(out, "st40_core_3d.stl"), "st40_core", CP, CT, CN)

    # dropper group: one id block per 2-D segment
    lo, hi = a.dropper_ids
    blocks, sel = [], []
    for line_id in range(lo, hi + 1):
        k = line_id - 1
        first = 2 * NPHI * k + 2 * j0 + 1; last = first + 2 * ns - 1
        blocks.append((first, last)); sel += list(range(first - 1, last))
    write_stl(os.path.join(out, "st40_dropper.stl"), "st40_dropper", WP, WT, WN, sel)
    with open(os.path.join(out, "st40_groups.inc"), "w") as f:
        f.write("# generated by scripts/make_geometry.py: dropper patch triangle ids\n")
        for first, last in blocks: f.write(f"group dropper surf id <> {first} {last}\n")

    # box and grid: one cell of margin around the wall
    m = a.cell
    xhi = Rmax + m; zlo, zhi = wall[:, 1].min() - m, wall[:, 1].max() + m
    nxy = int(round(2 * xhi / m)); nz = int(round((zhi - zlo) / m))
    with open(os.path.join(out, "st40_box.inc"), "w") as f:
        f.write(f"# generated by scripts/make_geometry.py (nphi {NPHI}, cell {m} m)\n")
        f.write(f"create_box {-xhi:.4f} {xhi:.4f} {-xhi:.4f} {xhi:.4f} {zlo:.4f} {zhi:.4f}\n")
        f.write(f"create_grid {nxy} {nxy} {nz}\n")

    rev = lambda RZ: sum(np.hypot(*(RZ[(k+1) % len(RZ)] - RZ[k])) * 0.5 * (RZ[k, 0] + RZ[(k+1) % len(RZ), 0])
                         for k in range(len(RZ))) * 2 * math.pi
    print(f"wall: {len(wall)} segments x {NPHI} strips = {len(wt)} triangles, area {WA.sum():.3f} m^2 (revolved {rev(wall):.3f})")
    print(f"core: {len(core)} segments x {NPHI} strips = {len(ct)} triangles, area {CA.sum():.3f} m^2 (revolved {rev(core):.3f})")
    print(f"dropper: ids {blocks}, {len(sel)} triangles, area {WA[sel].sum()*1e4:.2f} cm^2, "
          f"strip width {Rmax*dphi*100:.1f} cm at R={Rmax:.2f}")
    print(f"box +-{xhi:.3f} x +-{xhi:.3f} x [{zlo:.3f},{zhi:.3f}]; grid {nxy}x{nxy}x{nz} = {nxy*nxy*nz:,} cells")
    print("wrote input/st40_wall_3d.surf/.stl, st40_core_3d.surf/.stl, st40_dropper.stl, st40_groups.inc, st40_box.inc")


if __name__ == "__main__":
    main()
