#!/usr/bin/env python3
"""WEST PFU toroidal-wedge geometry for the slag sigma test (3D).

One PFU: 35 monoblock segments along the line MB1 (R,Z)=(1.96155,-0.60047)
to MB35 (2.36674,-0.76417), swept over the PFU
toroidal extent WEDGE_DEG = 0.79 deg about the torus (z) axis:
x = R cos(phi), y = R sin(phi), z = Z.

The closed surface is a "duct" whose cross-section in the poloidal (t,n)
plane is a rectangle: floor = the plate line, ceiling = emitter face at
HEIGHT along the plate normal, two end walls; swept in phi and capped at
phi = 0 and phi = WEDGE_DEG (toroidal periodicity caps).

Groups by triangle id (for NSEG=35, NSUB=8, NPHI=6; the script prints the
actual ranges -- keep the group commands in in.slag in sync):
  plate   1..3360       floor (NSEG*NSUB*NPHI*2)   PWI + sigma
  source  3361..6720    ceiling, same grid         vanish (plasma-side sink)
  ends    6721..6744    end walls                  vanish
  caps    6745..7864    phi caps                   surf_collide toroidal
"""

import numpy as np

R1, Z1 = 1.96155, -0.60047
R2, Z2 = 2.36674, -0.76417
NSEG = 35
NSUB = 8                      # poloidal cuts per monoblock
WEDGE_DEG = 0.79
NPHI = 6                      # toroidal subdivisions (mesh quality)
HEIGHT = 0.30                 # duct height above plate along +n [m]

p1 = np.array([R1, Z1])
p2 = np.array([R2, Z2])
d = p2 - p1
L = np.linalg.norm(d)
t = d / L
n = np.array([-t[1], t[0]])   # poloidal normal, plasma side (up-left)
if n[1] < 0:
    n = -n

phis = list(np.linspace(0.0, np.radians(WEDGE_DEG), NPHI + 1))

def to3d(rz, phi):
    return np.array([rz[0]*np.cos(phi), rz[0]*np.sin(phi), rz[1]])

# cross-section corners in (R,Z), counterclockwise seen from +phi side
NROW = NSEG * NSUB
plate_rz = [p1 + d * i / NROW for i in range(NROW + 1)]
top_rz = [rz + HEIGHT * n for rz in plate_rz]

pts = []                      # list of 3D points
tris = []                     # (i,j,k) 1-based, ordered for INWARD normals

def addp(p):
    pts.append(p)
    return len(pts)

def quad(a, b, c, dd):
    """two tris for quad a-b-c-d (1-based point ids), ccw as given"""
    tris.append((a, b, c))
    tris.append((a, c, dd))

# tensor vertex grids: rows over phi, columns over plate/ceiling vertices
floor_g = [[addp(to3d(rz, ph)) for rz in plate_rz] for ph in phis]
top_g   = [[addp(to3d(rz, ph)) for rz in top_rz] for ph in phis]

# 1) plate floor: NSEG x NPHI quads, ordered monoblock-major so tris of
#    monoblock m are contiguous: ids (m-1)*2*NPHI+1 .. m*2*NPHI
for i in range(NROW):
    for j in range(NPHI):
        quad(floor_g[j][i], floor_g[j][i+1], floor_g[j+1][i+1], floor_g[j+1][i])

# 2) source ceiling, same grid
for i in range(NROW):
    for j in range(NPHI):
        quad(top_g[j][i], top_g[j][i+1], top_g[j+1][i+1], top_g[j+1][i])

# 3) end walls at MB1 and MB35 (NPHI quads each)
for j in range(NPHI):
    quad(floor_g[j][0], floor_g[j+1][0], top_g[j+1][0], top_g[j][0])
for j in range(NPHI):
    quad(floor_g[j][-1], floor_g[j+1][-1], top_g[j+1][-1], top_g[j][-1])

# 4) phi caps at phi=0 and phi=WEDGE (NSEG quads each)
for i in range(NROW):
    quad(floor_g[0][i], floor_g[0][i+1], top_g[0][i+1], top_g[0][i])
for i in range(NROW):
    quad(floor_g[-1][i], floor_g[-1][i+1], top_g[-1][i+1], top_g[-1][i])

# orient every triangle so its normal points INTO the duct (flow side).
# The duct is convex enough that one interior reference point suffices.
xyz_all = np.array(pts)
center = to3d((p1 + p2) / 2 + (HEIGHT / 2) * n, np.radians(WEDGE_DEG / 2))
fixed = []
for (a, b, c) in tris:
    pa, pb, pc = xyz_all[a-1], xyz_all[b-1], xyz_all[c-1]
    nt = np.cross(pb - pa, pc - pa)
    if np.dot(nt, center - (pa + pb + pc) / 3.0) < 0.0:
        a, b, c = a, c, b
    fixed.append((a, b, c))
tris = fixed

with open("input/slag_wedge.surf", "w") as f:
    f.write("# WEST PFU 35-monoblock toroidal wedge duct "
            f"({WEDGE_DEG} deg, x=RcosPhi y=RsinPhi z=Z)\n\n")
    f.write(f"{len(pts)} points\n{len(tris)} triangles\n\nPoints\n\n")
    for i, p in enumerate(pts, 1):
        f.write(f"{i} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
    f.write("\nTriangles\n\n")
    for i, (a, b, c) in enumerate(tris, 1):
        f.write(f"{i} {a} {b} {c}\n")

xyz = np.array(pts)
seg = L / NSEG
print(f"plate {L:.5f} m, {NSEG} segs of {seg*1e3:.3f} mm; wedge {WEDGE_DEG} deg")
print(f"arc at MB35 R: {R2*np.radians(WEDGE_DEG)*1e3:.2f} mm")
np_, ns_ = 2*NROW*NPHI, 2*NROW*NPHI
ne_ = 4*NPHI
print(f"tris: plate 1..{np_}, source {np_+1}..{np_+ns_}, "
      f"ends {np_+ns_+1}..{np_+ns_+ne_}, caps {np_+ns_+ne_+1}..{len(tris)}")
print("bounds x [%.4f %.4f] y [%.6f %.6f] z [%.4f %.4f]"
      % (xyz[:,0].min(), xyz[:,0].max(), xyz[:,1].min(), xyz[:,1].max(),
         xyz[:,2].min(), xyz[:,2].max()))
print("beam -n (3D, mid-wedge): (%.5f, %.5f, %.5f)"
      % (-n[0]*np.cos(np.radians(WEDGE_DEG/2)),
         -n[0]*np.sin(np.radians(WEDGE_DEG/2)), -n[1]))
print("wrote input/slag_wedge.surf")
