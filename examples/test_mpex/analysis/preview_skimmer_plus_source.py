#!/usr/bin/env python3
import gmsh
import math

# --- same parameters in both scripts ---
Lx, Ly   = 0.10, 0.10
xc, yc   = Lx / 2, Ly / 2
z_front  = 0.029
slab_thk = 0.002
R_in     = 0.025
R_src    = 0.020
lc       = 0.002

def preview_skimmer_plus_source(filename="skimmer_plus_Ta.stl",
                                src_length=0.02):
    gmsh.initialize()
    gmsh.model.add("skimmer_plus_Ta")
    occ = gmsh.model.occ

    pad = 1e-4

    # --- skimmer ring (same construction as skimmer_ring.py) ---
    slab = occ.addBox(0.0, 0.0, z_front, Lx, Ly, slab_thk)
    hole = occ.addCylinder(
        xc, yc, z_front - pad,
        0.0, 0.0, slab_thk + 2*pad,
        R_in
    )
    cut = occ.cut([(3, slab)], [(3, hole)],
                  removeObject=True, removeTool=True)
    ring_vol = cut[0][0][1]

    # --- Ta source cylinder upstream ---
    z0 = z_front - src_length
    ta_vol = occ.addCylinder(xc, yc, z0, 0.0, 0.0, src_length, R_src)

    occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(filename)
    gmsh.finalize()
    print(f"Wrote combined STL: {filename}")


if __name__ == "__main__":
    preview_skimmer_plus_source()

