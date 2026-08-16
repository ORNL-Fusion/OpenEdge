#!/usr/bin/env python3
"""Generate the hybrid wall-test inputs: flat target surf + seeded He+
ensemble (fixed launch point, uniform random gyrophase). Stage A of
PLAN.md — uniform tilted B, no sheath, no collisions.

Frame: wall normal +z (target plane at z = 0), B tilted in the x-z
plane at angle alpha to the normal, pointing INTO the wall:
    bhat = (sin a, 0, -cos a)
Perpendicular basis: e1 = (cos a, 0, sin a), e2 = (0, -1, 0).
"""

import argparse
import os
import numpy as np

MHE = 6.6464731e-27
QE = 1.602176634e-19


def write_slab(path, half, zlo, zhi, label):
    v = [(-half, -half, zlo), (half, -half, zlo), (half, half, zlo),
         (-half, half, zlo), (-half, -half, zhi), (half, -half, zhi),
         (half, half, zhi), (-half, half, zhi)]
    tris = [(5, 6, 7), (5, 7, 8), (1, 3, 2), (1, 4, 3),
            (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
            (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]
    with open(path, "w") as f:
        f.write(f"# {label}\n\n{len(v)} points\n{len(tris)} triangles\n\nPoints\n\n")
        for k, (x, y, z) in enumerate(v, 1):
            f.write(f"{k} {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\nTriangles\n\n")
        for k, (a, b, c) in enumerate(tris, 1):
            f.write(f"{k} {a} {b} {c}\n")


def write_target(path, half=0.0299, zlo=-0.0015, zhi=0.0):
    # SPARTA requires watertight surfs: thin closed slab, top face at
    # z = zhi is the active target plane (triangles 1-2).
    v = [(-half, -half, zlo), (half, -half, zlo), (half, half, zlo),
         (-half, half, zlo), (-half, -half, zhi), (half, -half, zhi),
         (half, half, zhi), (-half, half, zhi)]
    tris = [(5, 6, 7), (5, 7, 8),          # top (+z) — the target plane
            (1, 3, 2), (1, 4, 3),          # bottom (-z)
            (1, 2, 6), (1, 6, 5),          # y = -half
            (2, 3, 7), (2, 7, 6),          # x = +half
            (3, 4, 8), (3, 8, 7),          # y = +half
            (4, 1, 5), (4, 5, 8)]          # x = -half
    with open(path, "w") as f:
        f.write("# thin slab target, active face at z = %g\n\n" % zhi)
        f.write(f"{len(v)} points\n{len(tris)} triangles\n\nPoints\n\n")
        for k, (x, y, z) in enumerate(v, 1):
            f.write(f"{k} {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\nTriangles\n\n")
        for k, (a, b, c) in enumerate(tris, 1):
            f.write(f"{k} {a} {b} {c}\n")


def write_ensemble(path, n, seed, alpha_deg, e_ev, x0, vpar_sign=1.0,
                   jitter=True):
    rng = np.random.default_rng(seed)
    a = np.radians(alpha_deg)
    bhat = np.array([np.sin(a), 0.0, -np.cos(a)])
    e1 = np.array([np.cos(a), 0.0, np.sin(a)])
    e2 = np.array([0.0, -1.0, 0.0])
    v = np.sqrt(2.0 * e_ev * QE / MHE)
    vpar = vperp = v / np.sqrt(2.0)        # 45-degree pitch
    vpar *= vpar_sign
    phases = rng.uniform(0.0, 2.0 * np.pi, n)
    # z-jitter over one gyration-descent wavelength (|vpar b_z| T_gyro):
    # a single-point single-phase-map launch keeps gyrophase COHERENT at
    # arrival, which no GC method can reproduce (phase memory). The
    # jitter makes the arrival-phase distribution stationary, so the
    # reference measures first-passage physics, not launch coherence.
    tg = 2.0 * np.pi * MHE / (QE * 1.0)
    lam = abs(vpar * bhat[2]) * tg
    zj = rng.uniform(0.0, lam, n) if jitter else np.zeros(n)
    with open(path, "w") as f:
        f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n%d\n" % n)
        f.write("ITEM: BOX BOUNDS oo oo oo\n-0.03 0.03\n-0.03 0.03\n-0.002 0.06\n")
        f.write("ITEM: ATOMS id type x y z vx vy vz\n")
        for k, (ph, dz) in enumerate(zip(phases, zj), 1):
            vv = vpar * bhat + vperp * (np.cos(ph) * e1 + np.sin(ph) * e2)
            f.write(f"{k} 1 {x0[0]:.6f} {x0[1]:.6f} {x0[2]+dz:.8f} "
                    f"{vv[0]:.10e} {vv[1]:.10e} {vv[2]:.10e}\n")


def write_ring(path, zlo=-0.0015, zhi=0.0, rlo=0.05, rhi=0.29):
    # Stage D1 axi target: annular plate in slot coords (x=Z, y=R).
    # 2D line normal = +z x (p2-p1), so a CLOCKWISE loop points normals
    # outward; line 1 is the active face at Z = zhi with normal +Z.
    pts = [(zhi, rhi), (zhi, rlo), (zlo, rlo), (zlo, rhi)]
    lns = [(1, 2), (2, 3), (3, 4), (4, 1)]
    with open(path, "w") as f:
        f.write("# axi ring target, active face at Z = %g\n\n" % zhi)
        f.write(f"{len(pts)} points\n{len(lns)} lines\n\nPoints\n\n")
        for k, (x, y) in enumerate(pts, 1):
            f.write(f"{k} {x:.6f} {y:.6f}\n")
        f.write("\nLines\n\n")
        for k, (a, b) in enumerate(lns, 1):
            f.write(f"{k} {a} {b}\n")


def write_axi_ensemble(path, n, seed, alpha_deg, e_ev, x0):
    # slot frame (x=Z, y=R, z=toroidal, right-handed). Wall normal +Z;
    # B tilted at alpha to it in the Z-toroidal plane (B_R = 0, so the
    # constant-background bz/bt pair reproduces it exactly):
    #     bhat = (-cos a, 0, sin a)
    # Z-jitter over one gyration-descent wavelength (phase decoherence,
    # same rationale as the 3D source).
    rng = np.random.default_rng(seed)
    a = np.radians(alpha_deg)
    bhat = np.array([-np.cos(a), 0.0, np.sin(a)])
    e1 = np.array([np.sin(a), 0.0, np.cos(a)])
    e2 = np.array([0.0, -1.0, 0.0])
    v = np.sqrt(2.0 * e_ev * QE / MHE)
    vpar = vperp = v / np.sqrt(2.0)
    phases = rng.uniform(0.0, 2.0 * np.pi, n)
    tg = 2.0 * np.pi * MHE / (QE * 1.0)
    lam = abs(vpar * bhat[0]) * tg
    zj = rng.uniform(0.0, lam, n)
    with open(path, "w") as f:
        f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n%d\n" % n)
        f.write("ITEM: BOX BOUNDS oo ao pp\n-0.005 0.06\n0.0 0.3\n-0.5 0.5\n")
        f.write("ITEM: ATOMS id type x y z vx vy vz\n")
        for k, (ph, dz) in enumerate(zip(phases, zj), 1):
            vv = vpar * bhat + vperp * (np.cos(ph) * e1 + np.sin(ph) * e2)
            f.write(f"{k} 1 {x0[0]+dz:.8f} {x0[1]:.6f} 0.0 "
                    f"{vv[0]:.10e} {vv[1]:.10e} {vv[2]:.10e}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--alpha", type=float, default=45.0)
    p.add_argument("--energy", type=float, default=5.0)
    args = p.parse_args()

    base = os.path.join(os.path.dirname(__file__), "input")
    os.makedirs(base, exist_ok=True)
    write_target(os.path.join(base, "target.surf"))
    # GC drifts +x at vpar*sin(a): start at -x so the footprint stays in-box
    write_ensemble(os.path.join(base, f"source.n{args.n}"),
                   args.n, args.seed, args.alpha, args.energy,
                   (-0.015, 0.0, 0.03))
    # hysteresis excursion: launched INSIDE the shell (z = 0.5 mm), moving
    # AWAY from the wall (vpar < 0). Must start Boris, retreat, and
    # re-enter GCA only after chi >= 4*pi AND d > 2*d_sw.
    write_ensemble(os.path.join(base, "source.hyst"),
                   32, args.seed + 1, args.alpha, args.energy,
                   (-0.015, 0.0, 0.0005), vpar_sign=-1.0, jitter=False)
    # ---- Stage B1 (alpha = 0, B || n): escape collector + ensembles ----
    # collector slab near the top; its BOTTOM face (facing the target)
    # is the escape detector — no sheath group membership, so escape
    # energy is measured cleanly
    write_slab(os.path.join(base, "escape.surf"), 0.0299, 0.050, 0.0515,
               "escape collector, bottom face at z = 0.050")
    # inbound: 5 eV, 45-deg pitch along B = -z, z-jittered launch
    write_ensemble(os.path.join(base, "source.b_in"),
                   args.n, args.seed + 2, 0.0, args.energy, (0.0, 0.0, 0.03))
    # outbound from INSIDE the sheath band (z = 0.5 mm, d_max ~ 3 mm):
    # sub-barrier E_par = 2 eV (total 4 eV at 45-deg pitch) must redeposit;
    # super-barrier E_par = 40 eV (total 80 eV) must escape minus e*phi
    write_ensemble(os.path.join(base, "source.b_sub"),
                   128, args.seed + 3, 0.0, 4.0, (0.0, 0.0, 0.0005),
                   vpar_sign=-1.0, jitter=False)
    write_ensemble(os.path.join(base, "source.b_sup"),
                   128, args.seed + 4, 0.0, 80.0, (0.0, 0.0, 0.0005),
                   vpar_sign=-1.0, jitter=False)
    # ---- Stage D1 (axi flat ring, 45-deg B via bz/bt constants) --------
    write_ring(os.path.join(base, "ring.surf"))
    write_axi_ensemble(os.path.join(base, "source.axi256"),
                       args.n, args.seed + 5, args.alpha, args.energy,
                       (0.03, 0.18))
    print(f"wrote input/{{target,escape,ring}}.surf, source.n{args.n}, "
          f"source.hyst, source.b_in/b_sub/b_sup, source.axi256 "
          f"(seed={args.seed})")


if __name__ == "__main__":
    main()
