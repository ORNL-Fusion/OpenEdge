#!/usr/bin/env python3
"""Build a thin Ta source plate for the MPEX 45-deg challenge case.

The plate is a flat rectangle at z = z_plate (default 1 mm), spanning the box
xy-footprint, triangulated as a regular Nx x Ny grid (default 50 x 50). Each
triangle's outward normal points +z so emitted Ta enters the fluid (toward
the skimmer/target) along +z.

Pairs with `fix surface/emit/source <mix> source file <flux_file> format inlet`
which looks up the per-surf gamma from ta_inlet.dat at each surf centroid.

The plate sits offset from z = 0 (box face) so it is not flush on the
boundary; default 1 mm = half a z-cell for the standard 100-cell axial grid.

Surface normals follow the unified 2026-04-21 convention: +normal = into the
fluid; no `invert` on read_surf.
"""

import argparse
from pathlib import Path


def write_plate(out_path, x0, x1, y0, y1, z, nx, ny):
    nv = (nx + 1) * (ny + 1)
    nc = nx * ny
    nt = 2 * nc

    # Vertex grid: (i, j) -> id = j * (nx + 1) + i + 1   (1-based)
    def vid(i, j):
        return j * (nx + 1) + i + 1

    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# Ta source plate for MPEX 45-deg challenge\n")
        f.write(f"# z = {z:g} m, xy = [{x0}, {x1}] x [{y0}, {y1}]\n")
        f.write(f"# {nx} x {ny} cells, +z normals (into fluid)\n\n")
        f.write(f"{nv} points\n")
        f.write(f"{nt} triangles\n\n")

        f.write("Points\n\n")
        pid = 1
        for j in range(ny + 1):
            for i in range(nx + 1):
                xv = x0 + i * dx
                yv = y0 + j * dy
                f.write(f"{pid} {xv:.10g} {yv:.10g} {z:.10g}\n")
                pid += 1

        f.write("\nTriangles\n\n")
        tid = 1
        for j in range(ny):
            for i in range(nx):
                v00 = vid(i,     j)
                v10 = vid(i + 1, j)
                v11 = vid(i + 1, j + 1)
                v01 = vid(i,     j + 1)
                # CCW winding when viewed from +z → outward normal +z
                f.write(f"{tid} {v00} {v10} {v11}\n"); tid += 1
                f.write(f"{tid} {v00} {v11} {v01}\n"); tid += 1

    print(f"[make_source_plate] wrote {out_path}")
    print(f"  z = {z:g} m, xy = [{x0}, {x1}] x [{y0}, {y1}]")
    print(f"  {nv} points, {nt} triangles")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", default="../surfaces/source_plate.txt",
                   help="Output SPARTA surf .txt path.")
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--x1", type=float, default=0.1)
    p.add_argument("--y0", type=float, default=0.0)
    p.add_argument("--y1", type=float, default=0.1)
    p.add_argument("-z",   "--z",  type=float, default=1.0e-3,
                   help="Plate z position [m] (default 1 mm).")
    p.add_argument("--nx", type=int, default=50)
    p.add_argument("--ny", type=int, default=50)
    args = p.parse_args()
    write_plate(args.out, args.x0, args.x1, args.y0, args.y1, args.z, args.nx, args.ny)


if __name__ == "__main__":
    main()
