
#!/usr/bin/env python3
import gmsh
import math

# --- geometry parameters (consistent with skimmer) ---
Lx, Ly   = 0.10, 0.10
xc, yc   = Lx / 2, Ly / 2
z_front  = 0.029          # skimmer upstream face
R_src    = 0.020          # Ta source radius
lc       = 0.01

def make_cylinder_source(filename="tantalum_cyl.stl",
                         length=0.005,   # <- half the old 0.02
                         radius=R_src,
                         gap=0.001):    # <- distance from skimmer
    """
    Tantalum source cylinder, coaxial with skimmer, upstream by 'gap'.

    Cylinder occupies z in [z_front - gap - length,  z_front - gap].
    """
    gmsh.initialize()
    gmsh.model.add("Ta_source")
    occ = gmsh.model.occ

    # downstream end of the source
    z_down = z_front - gap
    # upstream base where the cylinder starts
    z0 = z_down - length

    occ.addCylinder(xc, yc, z0, 0.0, 0.0, length, radius)
    occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(filename)
    gmsh.finalize()
    print(f"Wrote Ta cylinder STL: {filename}")


if __name__ == "__main__":

    make_cylinder_source("tantalum_cyl4.stl",
                         length=0.005,
                         radius=R_src,
                         gap=0.02)
