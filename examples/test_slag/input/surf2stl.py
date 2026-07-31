#!/usr/bin/env python3
"""SPARTA 3D tri .surf -> ASCII STL, one 'solid' per test_slag group so gmsh
imports them as separate surfaces (taggable as physical groups).
Usage: python3 surf2stl.py input/slag_wedge.surf output/slag_wedge.stl
"""
import sys

fin, fout = sys.argv[1], sys.argv[2]
lines = [l.split() for l in open(fin) if l.strip() and not l.startswith('#')]
npts, ntri = int(lines[0][0]), int(lines[1][0])
ip = lines.index(['Points']) + 1
it = lines.index(['Triangles']) + 1
pts = [[float(x) for x in l[1:4]] for l in lines[ip:ip+npts]]
tri = [[int(x)-1 for x in l[1:4]] for l in lines[it:it+ntri]]

GROUPS = [('plate', 0, 3360), ('source', 3360, 6720), ('ends', 6720, 6744), ('caps', 6744, ntri)]

def sub(a, b): return [a[i]-b[i] for i in range(3)]
def cross(a, b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

with open(fout, 'w') as f:
    for name, a, b in GROUPS:
        f.write(f'solid {name}\n')
        for t in tri[a:b]:
            p0, p1, p2 = (pts[i] for i in t)
            n = cross(sub(p1, p0), sub(p2, p0))
            m = max(1e-30, sum(x*x for x in n) ** 0.5)
            f.write('facet normal %.6e %.6e %.6e\n' % tuple(x/m for x in n))
            f.write(' outer loop\n')
            for p in (p0, p1, p2):
                f.write('  vertex %.8e %.8e %.8e\n' % tuple(p))
            f.write(' endloop\nendfacet\n')
        f.write(f'endsolid {name}\n')
print(f'wrote {fout}: ' + ', '.join(f'{n}[{b-a}]' for n, a, b in GROUPS))
