#!/usr/bin/env python3
import gmsh
import math

# --- global geometry parameters (meters) ---
Lx = 0.10   # 10 cm
Ly = 0.10
Lz = 0.20   # 20 cm

xc = Lx / 2.0
yc = Ly / 2.0

z_upstream   = 0.0
z_downstream = Lz

z_skimmer = z_upstream   + 0.03   # 3 cm from upstream end
z_target  = z_downstream - 0.05   # 5 cm from downstream end

skimmer_R_in = 0.025  # 5 cm inner diameter -> 2.5 cm radius
target_size  = 0.08   # 8 cm x 8 cm square

# mesh resolution (you can tune these)
lc_min = 0.002
lc_max = 0.004


def make_box(filename="box.stl"):
    """
    Axis-aligned Fe bounding box: 10 x 10 x 20 cm.
    STL will contain only the 6 outer faces.
    """
    gmsh.initialize()
    gmsh.model.add("box")

    # 3D box volume
    box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()

    # Mark volume and its boundary surfaces (for clarity; STL only keeps surfaces)
    gmsh.model.addPhysicalGroup(3, [box], 1)
    gmsh.model.setPhysicalName(3, 1, "box_vol")

    # Get all boundary surfaces of the volume
    boundary = gmsh.model.getBoundary([(3, box)], oriented=False, recursive=False)
    surf_tags = [tag for dim, tag in boundary if dim == 2]

    gmsh.model.addPhysicalGroup(2, surf_tags, 2)
    gmsh.model.setPhysicalName(2, 2, "box_surfaces")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
    print(f"Wrote {filename}")


def make_skimmer(filename="skimmer.stl"):
    """
    Full 10 x 10 cm plate at z = z_skimmer with a circular hole of ID 5 cm.
    """
    gmsh.initialize()
    gmsh.model.add("skimmer")

    # Full plate matching box cross-section
    plate = gmsh.model.occ.addRectangle(0.0, 0.0, z_skimmer, Lx, Ly)
    # Circular aperture centered on plasma axis
    hole = gmsh.model.occ.addDisk(xc, yc, z_skimmer, skimmer_R_in, skimmer_R_in)

    # Subtract hole from plate
    cut = gmsh.model.occ.cut([(2, plate)], [(2, hole)],
                             removeObject=True, removeTool=True)
    skimmer_surf = cut[0][0][1]

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [skimmer_surf], 1)
    gmsh.model.setPhysicalName(2, 1, "skimmer")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
    print(f"Wrote {filename}")


def make_target(angle_deg=0.0, filename="target.stl"):
    """
    8 x 8 cm W target centered on axis, located at z_target.
    Surface normal is rotated by angle_deg relative to +z
    about the y-axis through the plate center.
    """
    gmsh.initialize()
    gmsh.model.add(f"target_{angle_deg:g}")

    x0 = xc - target_size / 2.0
    y0 = yc - target_size / 2.0

    surf = gmsh.model.occ.addRectangle(x0, y0, z_target,
                                       target_size, target_size)

    if abs(angle_deg) > 1e-10:
        theta = math.radians(angle_deg)
        gmsh.model.occ.rotate([(2, surf)],
                              xc, yc, z_target,  # point on axis
                              0.0, 1.0, 0.0,      # rotate about y
                              theta)

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [surf], 1)
    gmsh.model.setPhysicalName(2, 1, f"target_{angle_deg:g}deg")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
    print(f"Wrote {filename}")


if __name__ == "__main__":
    # 1) box surfaces
    make_box("box.stl")

    # 2) skimmer plate
    make_skimmer("skimmer.stl")

    # 3) targets at 0°, 45°, 85°
    make_target(0.0,  "target_0deg.stl")
    make_target(45.0, "target_45deg.stl")
    make_target(85.0, "target_85deg.stl")

