#!/usr/bin/env python3
"""Closed 10 deg toroidal wedge (R 1-2 m, Z +-0.5 m) as 12 triangles, normals into the flow."""
import numpy as np
from pathlib import Path

HALF = np.radians(5.0)
R0, R1, Z0, Z1 = 1.0, 2.0, -0.5, 0.5
pts = []
for r in (R0, R1):
    for phi in (-HALF, HALF):
        for z in (Z0, Z1):
            pts.append((r * np.cos(phi), r * np.sin(phi), z))
pts = np.array(pts)                       # index = 4*ir + 2*iphi + iz
ctr = pts.mean(axis=0)
def quad(a, b, c, d):                      # two triangles, flipped if the normal points away from the centre
    tris = []
    for t in ((a, b, c), (a, c, d)):
        p = pts[list(t)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.dot(n, ctr - p.mean(axis=0)) < 0: t = (t[0], t[2], t[1])
        tris.append(t)
    return tris
faces = {"inner": quad(0, 1, 3, 2), "outer": quad(4, 5, 7, 6),
         "cap_lo": quad(0, 1, 5, 4), "cap_hi": quad(2, 3, 7, 6),
         "bottom": quad(0, 2, 6, 4), "top": quad(1, 3, 7, 5)}
tris = [t for f in faces.values() for t in f]
out = Path(__file__).resolve().parents[1] / "input/wedge.surf"
with open(out, "w") as f:
    f.write("# closed 10 deg wedge, R 1-2 m, Z -0.5..0.5 m; tris 1-4 inner/outer, 5-8 caps, 9-12 top/bottom\n\n")
    f.write(f"{len(pts)} points\n{len(tris)} triangles\n\nPoints\n\n")
    for i, p in enumerate(pts, 1): f.write(f"{i} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}\n")
    f.write("\nTriangles\n\n")
    for i, t in enumerate(tris, 1): f.write(f"{i} {t[0]+1} {t[1]+1} {t[2]+1}\n")
print(out, len(tris), "triangles")
