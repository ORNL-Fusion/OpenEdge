#!/usr/bin/env python3
"""Build a SPARTA 2D core surface from a plasma.h5 psi_norm contour.

Extracts the psi_norm = LEVEL flux surface from the embedded equilibrium in
plasma.h5 and writes it as a closed SPARTA surf loop in the axisymmetric
slot layout (file x = Z, file y = R). The loop is oriented so SPARTA's
normal (n = z_hat x (p2 - p1) = (-dy, dx)) points OUTWARD from the core,
i.e. into the SOL/confined-plasma fluid. Used to geometrically exclude the
deep-core cells from the mesh (they get marked inside/solid), so the base
grid + adapt_grid resolution goes to the annulus between wall and core
instead of the region we do not care about.

Matches the psi_norm threshold of `fix reflect/psi` so the geometric core
boundary and the particle absorber coincide.

Usage:
  python3 make_core_surf.py input/plasma.h5 input/core.surf --level 0.1
"""

import argparse
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plasma_h5")
    ap.add_argument("outfile")
    ap.add_argument("--level", type=float, default=0.1,
                    help="psi_norm contour level for the core edge")
    args = ap.parse_args()

    with h5py.File(args.plasma_h5, "r") as f:
        psi = f["equilibrium/psi"][:]          # [iz, ir]
        r = f["equilibrium/r"][:]
        z = f["equilibrium/z"][:]
        psi_axis = float(f["equilibrium/psi_axis"][()])
        psib = float(f["equilibrium/psib"][()])

    psin = (psi - psi_axis) / (psib - psi_axis)
    RR, ZZ = np.meshgrid(r, z)                  # both [iz, ir], match psin

    cs = plt.contour(RR, ZZ, psin, levels=[args.level])
    segs = cs.allsegs[0]
    if not segs:
        sys.exit(f"no psi_norm = {args.level} contour found")
    # keep the longest closed loop (the core boundary)
    loop = max(segs, key=len)                   # (npts, 2) columns (R, Z)
    R_c, Z_c = loop[:, 0], loop[:, 1]

    # drop a duplicate closing point if matplotlib repeated the first
    if np.hypot(R_c[0] - R_c[-1], Z_c[0] - Z_c[-1]) < 1e-9:
        R_c, Z_c = R_c[:-1], Z_c[:-1]

    # file columns are (x = Z, y = R)
    X, Y = Z_c, R_c                             # X = Z (axial), Y = R (radial)

    # Orient so SPARTA normals n = (-dy, dx) point AWAY from the centroid
    # (outward from the core, into the SOL). Test on the mean projection of
    # each edge normal onto (midpoint - centroid); flip the walk if inward.
    cx, cy = X.mean(), Y.mean()
    dx = np.roll(X, -1) - X
    dy = np.roll(Y, -1) - Y
    mx = 0.5 * (X + np.roll(X, -1)) - cx
    my = 0.5 * (Y + np.roll(Y, -1)) - cy
    outward = np.sum((-dy) * mx + dx * my)      # >0 => normals point outward
    if outward < 0.0:
        X, Y = X[::-1], Y[::-1]

    npt = len(X)
    with open(args.outfile, "w") as fh:
        fh.write("surface geometry: core psi_norm=%g contour from %s\n\n"
                 % (args.level, args.plasma_h5))
        fh.write("%d points\n%d lines\n\nPoints\n\n" % (npt, npt))
        for i in range(npt):
            fh.write("%d %.10g %.10g\n" % (i + 1, X[i], Y[i]))
        fh.write("\nLines\n\n")
        for i in range(npt):
            fh.write("%d %d %d\n" % (i + 1, i + 1, (i + 1) % npt + 1))

    print("%s: psi_norm=%g core loop -> %s: %d points / %d lines"
          % (args.plasma_h5, args.level, args.outfile, npt, npt))
    print("  extent  R=[%.3f, %.3f]  Z=[%.3f, %.3f]  (outward normals into SOL)"
          % (R_c.min(), R_c.max(), Z_c.min(), Z_c.max()))


if __name__ == "__main__":
    sys.exit(main())
