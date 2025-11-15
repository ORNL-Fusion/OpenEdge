#!/usr/bin/env python3
import gmsh
import math

# --- geometry parameters ---
Lx, Ly   = 0.10, 0.10          # outer plate size
xc, yc   = Lx / 2, Ly / 2      # center of skimmer
z_front  = 0.029               # upstream face z
slab_thk = 0.002               # plate thickness
R_in     = 0.025               # hole radius
lc       = 0.002               # target mesh size

def make_ring(filename="skimmer_ring.stl"):
    gmsh.initialize()
    gmsh.model.add("skimmer_ring")
    occ = gmsh.model.occ

    # tiny padding so the cylinder fully cuts the slab
    pad = 1e-4

    # 1) slab volume (full plate)
    slab = occ.addBox(
        0.0, 0.0, z_front,     # x0, y0, z0
        Lx,  Ly,  slab_thk     # dx, dy, dz
    )

    # 2) cylinder volume through the slab
    cyl = occ.addCylinder(
        xc, yc, z_front - pad,          # base center
        0.0, 0.0, slab_thk + 2*pad,     # axis vector
        R_in                            # radius
    )

    # 3) boolean difference: slab minus cylinder -> ring
    cut_out = occ.cut([(3, slab)], [(3, cyl)],
                      removeObject=True, removeTool=True)
    ring_vols = cut_out[0]   # list of (dim, tag); we don't actually need tags explicitly

    occ.synchronize()

    # 4) mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.008)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.008)

    # generate full 3D mesh; STL exporter will dump the boundary triangles
    gmsh.model.mesh.generate(3)

    # only save external boundary, not internal interfaces
    gmsh.option.setNumber("Mesh.SaveAll", 0)

    gmsh.write(filename)
    gmsh.finalize()
    print("Wrote ring skimmer STL:", filename)

if __name__ == "__main__":
    make_ring("skimmer_ring.stl")

