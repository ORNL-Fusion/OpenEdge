#!/usr/bin/env python3
import gmsh
import math

# --- geometry (meters) ---
Lx, Ly, Lz = 0.10, 0.10, 0.20
xc, yc = Lx / 2.0, Ly / 2.0

z_upstream, z_downstream = 0.0, Lz
z_skimmer = z_upstream + 0.03          # not used here, just context
z_target  = z_downstream - 0.05        # center of the W target

skimmer_R_in = 0.025
target_size  = 0.08                    # 8 cm x 8 cm

lc_min = 0.002
lc_max = 0.004

def _set_mesh():
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)

def make_box(filename="box.stl"):
    gmsh.initialize()
    gmsh.model.add("box")
    box = gmsh.model.occ.addBox(-1e-3, -1e-3, -1e-3, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [box], 1)
    gmsh.model.setPhysicalName(3, 1, "box_vol")

    bnd = gmsh.model.getBoundary([(3, box)], oriented=False, recursive=False)
    surf_tags = [tag for dim, tag in bnd if dim == 2]
    gmsh.model.addPhysicalGroup(2, surf_tags, 2)
    gmsh.model.setPhysicalName(2, 2, "box_surfaces")

#    _set_mesh()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.02)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.04)
    
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
    print(f"Wrote {filename}")

def make_target(angle_deg, filename, thickness=0.002):
    """
    Inclined rectangular target plate.

    - angle_deg = 0: plate is perpendicular to z (normal ~ +z),
                     i.e. facing the skimmer/beam.
    - angle_deg > 0: plate rotated about +y, tilting in x–z plane.
    """
    gmsh.initialize()
    gmsh.model.add(f"target_{angle_deg:g}deg")
    occ = gmsh.model.occ

    # Plate dimensions:
    #   large in x and y (8 cm x 8 cm),
    #   thin in z (thickness is the normal direction initially).
    dx = target_size       # extent in x
    dy = target_size       # extent in y
    dz = thickness         # small thickness along z

    # Center of the target
    xc0 = xc
    yc0 = yc
    zc0 = z_target

    # Create axis-aligned box centered at (xc0, yc0, zc0)
    x0 = xc0 - dx / 2.0
    y0 = yc0 - dy / 2.0
    z0 = zc0 - dz / 2.0
    plate = occ.addBox(x0, y0, z0, dx, dy, dz)

    # Rotate about the y-axis through the center of the plate
    angle_rad = math.radians(angle_deg)
    occ.rotate(
        [(3, plate)],
        xc0, yc0, zc0,   # rotation center
        1.0, 0.0, 0.0,   # axis = +y
        angle_rad
    )

    occ.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(3, [plate], 1)
    gmsh.model.setPhysicalName(3, 1, f"target_vol_{angle_deg:g}deg")

    bnd = gmsh.model.getBoundary([(3, plate)], oriented=False, recursive=False)
    surf_tags = [tag for dim, tag in bnd if dim == 2]
    gmsh.model.addPhysicalGroup(2, surf_tags, 2)
    gmsh.model.setPhysicalName(2, 2, f"target_surfs_{angle_deg:g}deg")

    _set_mesh()
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
    print(f"Wrote {filename}")

if __name__ == "__main__":
    make_box("box.stl")
    make_target(0.0,  "target_0deg.stl")   # normal ~ +z, facing skimmer
    make_target(45.0, "target_45deg.stl")  # tilted like the cartoon
    make_target(85.0, "target_85deg.stl")  # almost grazing incidence
