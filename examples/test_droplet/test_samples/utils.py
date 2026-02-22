from shapely.geometry import Point, Polygon
import glob

class MathExtra:
    @staticmethod
    def sub3(v1, v2):
        """Subtracts two 3D vectors."""
        return np.array([v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]])

    @staticmethod
    def cross3(v1, v2):
        """Calculates the cross product of two 3D vectors."""
        return np.array([
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        ])

    @staticmethod
    def norm3(v):
        """Normalizes a 3D vector."""
        scale = 1.0 / np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        return v * scale

class Line:
    def __init__(self, p1, p2):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.norm = np.array([0.0, 0.0, 0.0])

class Surf:
    def __init__(self, lines):
        self.lines = lines

    def compute_line_normal(self, old=0):
        """Computes the normal for each line segment starting from index `old`."""
        z = np.array([0.0, 0.0, 1.0])  # Unit vector along the Z-axis
        
        for i in range(old, len(self.lines)):
            delta = MathExtra.sub3(self.lines[i].p2, self.lines[i].p1)  # Direction vector of the line
            norm = MathExtra.cross3(z, delta)  # Cross product to get normal vector
            norm = MathExtra.norm3(norm)       # Normalize the vector
            norm[2] = 0.0  # Set the Z-component to zero as in the original C++ code
            
            self.lines[i].norm = norm
#            print(f"Line {i}: p1 = {self.lines[i].p1}, p2 = {self.lines[i].p2}, normal = {self.lines[i].norm}")

def build_surf_from_wall(wall):
    """
    Convert `surface` (R,Z polygon) into Surf of Line objects in 3D (x,y,z),
    matching the C++ convention: x = R, y = Z, z = 0.
    """
    lines = []
    skipped = 0

    for line_id, materials, (p1_idx, p2_idx) in wall.lines:
        if p1_idx not in wall.points or p2_idx not in wall.points:
            print(f"[build_surf_from_wall] WARNING: skipping line {line_id} "
                  f"because point(s) {p1_idx}, {p2_idx} not in wall.points")
            skipped += 1
            continue

        R1, Z1 = wall.points[p1_idx]
        R2, Z2 = wall.points[p2_idx]

        # *** match C++: (x,y,z) = (R,Z,0) ***
        p1 = (R1, Z1, 0.0)
        p2 = (R2, Z2, 0.0)

        lines.append(Line(p1, p2))

    surf = Surf(lines)
    surf.compute_line_normal()  # uses z×delta, just like C++
    return surf


def point_to_segment_distance_2d(p, a, b):
    """
    p, a, b: 2D (R,Z) numpy arrays
    returns: (distance, closest_point_on_segment, t)
    """
    p = np.asarray(p, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    ab = b - a
    denom = np.dot(ab, ab)
    if denom == 0.0:
        # a and b are the same point
        return np.linalg.norm(p - a), a, 0.0

    t = np.dot(p - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    dist = np.linalg.norm(p - closest)
    return dist, closest, t
    
def closest_line_and_normal(surf, R, Z):
    """
    Find the closest line in `surf` to the point (R,Z) in the poloidal plane.

    Now that we embed as (x,y,z) = (R,Z,0), the poloidal coords are (p[0], p[1]).
    """
    p = np.array([R, Z], float)

    best_idx = None
    best_d = np.inf
    best_point = None

    for i, line in enumerate(surf.lines):
        a = np.array([line.p1[0], line.p1[1]])  # (R,Z)
        b = np.array([line.p2[0], line.p2[1]])

        d, closest, _ = point_to_segment_distance_2d(p, a, b)
        if d < best_d:
            best_d = d
            best_idx = i
            best_point = closest

    return best_idx, best_point, best_d

def angle_between_velocity_and_normal(surf, line_idx, vx, vy, vz):
    """
    Match C++: angle = acos( v·n / (|v||n|) ), in degrees.
    """
    n = np.asarray(surf.lines[line_idx].norm, float)
    v = np.array([vx, vy, vz], float)

    n_norm = np.linalg.norm(n)
    v_norm = np.linalg.norm(v)
    if n_norm == 0.0 or v_norm == 0.0:
        raise ValueError("Zero-length normal or velocity; cannot compute angle.")

    cos_theta = np.dot(n, v) / (n_norm * v_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))




class surface:
    def __init__(self, filename, geom_type):
        # read surface file
        with open(filename, 'r') as f:
            # skip first two lines
            data = f.readlines()[2:]
            # read the number of points and lines
            num_points = int(data[0].split()[0])
            num_lines = int(data[1].split()[0])
            print('Number of points: ', num_points)
            print('Number of lines: ', num_lines)
            # read points
            self.points = {}
            for line in data[5:num_points]:
                # print(line)
                point_id, *point_coords = map(float, line.split())
                self.points[int(point_id)] = tuple(point_coords)
            # read lines
            self.lines = []
            # self.material = {}
            for line in data[num_points+8:]:
                line_parts = line.split()
                materials= line_parts[-1]
                line_id= int(line_parts[0])
                line_points = [int(line_parts[1]), int(line_parts[2])]
                self.lines.append((int(line_id), str(materials), line_points))
        # create Polygon object from points
        points_list = [list(p) for p in self.points.values()]
        self.polygon = Polygon(points_list)
        # set surface ID
        self.id = filename.split('.')[0]
        

def parser(filename, start_species, end_species):
    results = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(lines[i + 1].strip())
                number_of_cells = int(lines[i + 3].strip())
                xcs = []
                ycs = []
                density_values = [[] for _ in range(start_species, end_species + 1)]
                for j in range(number_of_cells):
                    cell_data_index = i + 5 + j
                    if cell_data_index >= len(lines):  # Check to avoid going out of bounds
                        break
                    cell_data_line = lines[cell_data_index]
                    cell_data = cell_data_line.split()
                    if len(cell_data) < end_species + 1:  # Ensure we have enough data in the line
                        continue
                    try:
                        xc = float(cell_data[1])
                        yc = float(cell_data[2])
                        densities = [float(cell_data[k]) for k in range(start_species, end_species + 1)]
                        xcs.append(xc)
                        ycs.append(yc)
                        for idx, density in enumerate(densities):
                            density_values[idx].append(density)
                    except ValueError:
                        # Skip if not a valid float
                        continue
                results[timestep] = [xcs, ycs] + density_values
    return results


#wall = surface("wall.surf", "2D")
#domain = wall.polygon
#    domain = wall.polygon
#Rwall, Zwall = domain.exterior.xy
#    ax.plot(Rwall, Zwall, 'k-', lw=2.5)
#ax.plot(rcore, zcore, 'g-', lw=2.5)

import os, glob
import numpy as np
import matplotlib.pyplot as plt

class Surface2:
    def __init__(self, filename, geom_type="2D"):
        self.filename = filename
        self.geom_type = geom_type
        with open(filename, "r") as f:
            raw = f.readlines()

        # Skip the first two header lines
        data = raw[2:]

        # Counts
        num_points = int(data[0].split()[0])
        num_lines  = int(data[1].split()[0])

        # ---- points ----
        # Many .surf files have ~5 header lines before coordinates; keep your offset
        # but read exactly 'num_points' rows.
        pts_start = 5
        pts_end   = pts_start + num_points
        self.points = {}                 # {point_id: (x, y[, z])}
        for line in data[pts_start:pts_end]:
            toks = line.split()
            if not toks:
                continue
            pid = int(float(toks[0]))
            coords = tuple(map(float, toks[1:3]))     # (R,Z) for 2D
            self.points[pid] = coords

        # ---- lines ----
        lines_start = pts_end + 3    # your file seemed to have a small gap; adjust if needed
        lines_end   = lines_start + num_lines
        self.lines = []              # list of (line_id, material, [p1, p2])
        for line in data[lines_start:lines_end]:
            toks = line.split()
            if len(toks) < 4:
                continue
            line_id   = int(toks[0])
            p1, p2    = int(toks[1]), int(toks[2])
            material  = toks[-1]
            self.lines.append((line_id, material, [p1, p2]))

        # surface ID from filename
        self.id = os.path.splitext(os.path.basename(filename))[0]

    def plot(self, ax, label_points=True, label_lines=False):
        # draw segments
        for lid, mat, (a, b) in self.lines:
            xa, ya = self.points[a]
            xb, yb = self.points[b]
            ax.plot([xa, xb], [ya, yb], lw=1)
            if label_lines:
                xm, ym = 0.5*(xa+xb), 0.5*(ya+yb)
                ax.text(xm, ym, f"{self.id}-L{lid}", fontsize=7, ha="center", va="center")

        # label vertices
        if label_points:
            for pid, (x, y) in self.points.items():
                ax.scatter([x], [y], s=12)
                ax.text(x, y, f"{self.id}:{pid}", fontsize=8, ha="center", va="bottom")

import os
import numpy as np
import matplotlib.pyplot as plt

class Surface3:
    def __init__(self, filename, geom_type="2D"):
        self.filename = filename
        self.geom_type = geom_type
        with open(filename, "r") as f:
            raw = f.readlines()

        data = raw[2:]
        num_points = int(data[0].split()[0])
        num_lines  = int(data[1].split()[0])

        pts_start = 5
        pts_end   = pts_start + num_points
        self.points = {}
        for line in data[pts_start:pts_end]:
            toks = line.split()
            if not toks:
                continue
            pid = int(float(toks[0]))
            self.points[pid] = tuple(map(float, toks[1:3]))  # (R,Z)

        lines_start = pts_end + 3
        lines_end   = lines_start + num_lines
        self.lines = []
        for line in data[lines_start:lines_end]:
            toks = line.split()
            if len(toks) < 4:
                continue
            line_id = int(toks[0])
            p1, p2  = int(toks[1]), int(toks[2])
            mat     = toks[-1]
            self.lines.append((line_id, mat, [p1, p2]))

        self.id = os.path.splitext(os.path.basename(filename))[0]
        
    def plot(self, ax, selected_point_ids=None,
         selected_color="tab:green", selected_size=80, number_selected=True,
         show_other_points=False, other_point_size=8, other_alpha=0.25,
         show_edge_midpoints=False, edge_midpoint_style=dict(s=30, marker="x")):

        # 1) segments: black
        mids = []  # collect midpoints if requested
        for lid, mat, (a, b) in self.lines:
            xa, ya = self.points[a]
            xb, yb = self.points[b]
            ax.plot([xa, xb], [ya, yb], color="k", lw=1)
            if show_edge_midpoints:
                xm, ym = 0.5*(xa+xb), 0.5*(ya+yb)
                mids.append((xm, ym, lid))

        # 2) optional faint other points
        if show_other_points:
            Xo, Yo = zip(*[self.points[pid] for pid in self.points.keys()])
            ax.scatter(Xo, Yo, s=other_point_size, color="k", alpha=other_alpha)

        # 3) highlights
        if selected_point_ids:
            sel_ids = [pid for pid in selected_point_ids if pid in self.points]
            Xs, Ys = zip(*[self.points[pid] for pid in sel_ids])
            ax.scatter(Xs, Ys, s=selected_size, color=selected_color, zorder=5)
            if number_selected:
                for k, pid in enumerate(sel_ids, 1):
                    x, y = self.points[pid]
                    ax.text(x, y, f"{k}", ha="center", va="bottom", fontsize=9, color=selected_color)

        # 4) draw edge midpoints
        if show_edge_midpoints and mids:
            Xm, Ym = zip(*[(m[0], m[1]) for m in mids])
            ax.scatter(Xm, Ym, **edge_midpoint_style)
            # optionally label with line IDs:
            # for xm, ym, lid in mids:
            #     ax.text(xm, ym, f"L{lid}", fontsize=8, ha="center", va="bottom")

        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")

#
#    def plot(self,
#             ax,
#             selected_point_ids=None,
#             selected_color="tab:green",
#             selected_size=80,
#             number_selected=True,
#             show_other_points=False,
#             other_point_size=8,
#             other_alpha=0.25):
#        """
#        Draws all segments in black. Highlights only 'selected_point_ids' as colored dots.
#        If show_other_points=True, other vertices are drawn faintly.
#        """
#        # 1) segments: black
#        for lid, mat, (a, b) in self.lines:
#            xa, ya = self.points[a]
#            xb, yb = self.points[b]
#            ax.plot([xa, xb], [ya, yb], color="k", lw=1)
#
#        # 2) optional faint other points
#        if show_other_points:
#            Xo, Yo = zip(*[self.points[pid] for pid in self.points.keys()])
#            ax.scatter(Xo, Yo, s=other_point_size, color="k", alpha=other_alpha)
#
#        # 3) highlights
#        if selected_point_ids:
#            sel_ids = [pid for pid in selected_point_ids if pid in self.points]
#            Xs, Ys = zip(*[self.points[pid] for pid in sel_ids])
#            ax.scatter(Xs, Ys, s=selected_size, color=selected_color, zorder=5)
#            print("Selected positions", Xs, Ys)
#            if number_selected:
#                for k, pid in enumerate(sel_ids, 1):
#                    x, y = self.points[pid]
#                    ax.text(x, y, f"{k}", ha="center", va="bottom", fontsize=9, color=selected_color)
#
#        ax.set_aspect("equal")
#        ax.set_xlabel("R [m]")
#        ax.set_ylabel("Z [m]")


def plot_surfaces(pattern="*.surf", label_points=True, label_lines=False):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    for fn in glob.glob(pattern):
        S = Surface2(fn, "2D")
        S.plot(ax, label_points=label_points, label_lines=label_lines)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Surfaces with vertex IDs")
    plt.show()
    return ax


def parse_file(filename):
    timesteps = []
    x_coords = []
    y_coords = []
    z_coords = []
    vx_coords = []
    vy_coords = []
    vz_coords = []
    mass = []
    temp = []
    radius = []
    ids = []

    with open(filename, 'r') as file:
        lines = file.readlines()

    i = 0
    timestep = 0
    num_atoms = 0
    while i < len(lines):
        s = lines[i].strip()

        if s == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2
            continue

        if s == "ITEM: NUMBER OF ATOMS":
            num_atoms = int(lines[i + 1].strip())
            i += 2
            continue

        if s.startswith("ITEM: ATOMS"):
            cols = s.split()[2:]
            col_idx = {c: k for k, c in enumerate(cols)}
            need = ["id", "x", "y", "z", "vx", "vy", "vz", "v_pmass", "temp", "radius"]
            if not all(c in col_idx for c in need):
                i += 1 + max(0, num_atoms)
                continue

            for j in range(num_atoms):
                row = lines[i + 1 + j].strip().split()
                if len(row) < len(cols):
                    continue
                ids.append(int(float(row[col_idx["id"]])))
                x_coords.append(float(row[col_idx["x"]]))
                y_coords.append(float(row[col_idx["y"]]))
                z_coords.append(float(row[col_idx["z"]]))
                vx_coords.append(float(row[col_idx["vx"]]))
                vy_coords.append(float(row[col_idx["vy"]]))
                vz_coords.append(float(row[col_idx["vz"]]))
                mass.append(float(row[col_idx["v_pmass"]]))
                temp.append(float(row[col_idx["temp"]]))
                radius.append(float(row[col_idx["radius"]]))
                timesteps.append(timestep)

            i += 1 + max(0, num_atoms)
            continue

        i += 1

    return timesteps, x_coords, y_coords, z_coords, vx_coords, vy_coords, vz_coords, mass, temp, radius, ids

def load_one_id(fname):
    ts,x,y,z,vr,vz_c,vphi,mass,temp,r,ids = parse_file(fname)
    ids = np.asarray(ids); pick = np.unique(ids)[0]
    m = (ids == pick)
    tstep = np.asarray(ts,float)[m]
    vr    = np.asarray(vr,float)[m]
    vz    = np.asarray(vz_c,float)[m]   # this is v_z (cyl)
    vphi  = np.asarray(vphi,float)[m]
    zpos  = np.asarray(z,float)[m]
    m_d   = float(np.asarray(mass,float)[m][0])
    return tstep, vr, vz, vphi, zpos, m_d
        
#wall = surface("wall.surf", "2D")
##    fig, axes = plt.subplots(2, len(cases), figsize=(3.5*len(cases), 6), dpi=400, sharex=True)
#fig, ax = plt.subplots(figsize=(4,3), dpi=400)
#core = surface("core.surf", "2D")
#rcore, zcore = core.polygon.exterior.xy
#
#domain = wall.polygon
#Rwall, Zwall = domain.exterior.xy
#ax.plot(Rwall, Zwall, 'k-', lw=2.5)
#ax.plot(rcore, zcore, 'g-', lw=2.5)
#innersite=np.loadtxt("innersite.txt", unpack=True, skiprows=1)
#outersite=np.loadtxt("outersite.txt", unpack=True, skiprows=1)
#ax.plot(innersite[1], innersite[2], 'ro', ms=2.5)
#ax.plot(outersite[1], outersite[2], 'ko', ms=2.5)
#ax.tick_params(axis="both", labelsize=10, direction="in")
#ax.set_ylabel(r"Z(m)",fontsize=14)
#ax.set_ylabel(r"R(m)",fontsize=14)
#ax.grid(ls="--", alpha=0.5)
##fig.subplots_adjust(hspace=0.1, wspace=0.3)
#
#fig.savefig('launching_sites.png', dpi=400, bbox_inches="tight", facecolor="white")
##plt.show()
#plt.show()
###

# choose the vertex IDs you want to show as launch sites
##launch_ids = [i for i in range(38,50,1)]
#launch_ids = [11,12,13,14,15,16,17]

#for id in launch_ids:
#    tstep, x, y, z, vx, vy, vz, mass, temp, radius, ids = parse_file(f"case.{id}")
#    ax.plot(x,y, 'ro', label='atom')
#wall = Surface3("wall.surf", "2D")
#wall.plot(ax,
#          selected_point_ids=launch_ids,
#          show_other_points=False,
#          show_edge_midpoints=True,
#          edge_midpoint_style=dict(s=36, marker="x", color="tab:orange"))
##plt.show()
##ax.plot(3.397879,-3.666131 , 'ks', ms=10, label='atom')
##    ax.plot(x,y, 'ro', label='atom')
#ax.set_xlim(2.40,2.6)
#ax.set_ylim(-3.05,-2.7)
#ax.set_title(f"{wall.id}: segments black, selected launch sites highlighted")
#ax.legend()
#plt.show()




#import numpy as np
#
#case_ids =launch_ids # [37, 38, 39, 40, 42, 44, 46, 48, 50]
#rows = []
#
#for cid in case_ids:
#    tstep, x, y, z, vx, vy, vz, mass, temp, radius, ids = parse_file(f"case.{cid}")
#
#    # Make sure x,y are arrays
#    x = np.asarray(x)
#    y = np.asarray(y)
#
#    # Choose what to save:
#    # A) last sample of a 1D trajectory
#    if x.ndim == 1:
#        rows.append([cid, float(x[-1]), float(y[-1])])
#
#    # B) last sample of particle 0 in a (npart, nt) array
#    elif x.ndim == 2:
#        rows.append([cid, float(x[0, -1]), float(y[0, -1])])
#
#    else:
#        raise ValueError(f"Unexpected x/y shape for case {cid}: {x.shape}")
#
## Save as plain text: case_id x y
#np.savetxt(
#    "innersite.txt",
#    np.array(rows, dtype=float),
#    fmt=["%d", "%.8f", "%.8f"],
#    header="case_id x y",
#    comments=""
#)
#print("Wrote innersite.txt with", len(rows), "rows")
#


def angle_vnorm_df(case_path, droplet_class, wall_path="/Users/42d/OpenedgeGPU/examples/test_droplet/wall.surf"):
    """
    Given a particle dump `case_path` and a wall surface file,
    compute the angle between velocity and wall normal for the
    first appearance of each particle.

    Returns a DataFrame with columns: id, angle_deg, vnorm, droplet.
    """

    # --- load particle data ---
    tstep, x, y, z, vx, vy, vz, mass, temp, radius, ids = parse_file(case_path)

    x  = np.asarray(x,  float)
    y  = np.asarray(y,  float)
    z  = np.asarray(z,  float)
    vx = np.asarray(vx, float)
    vy = np.asarray(vy, float)
    vz = np.asarray(vz, float)
    ids = np.asarray(ids)

    if ids.size == 0:
        return pd.DataFrame(columns=["id", "angle_deg", "vnorm", "droplet"])

    # ---- first record for each particle id ----
    unique_ids, first_idx = np.unique(ids, return_index=True)

    x0  = x[first_idx]
    y0  = y[first_idx]
    z0  = z[first_idx]
    vx0 = vx[first_idx]
    vy0 = vy[first_idx]
    vz0 = vz[first_idx]
    ids0 = unique_ids

    # --- build wall + normals ---
    wall = surface(wall_path, "2D")
    surf = build_surf_from_wall(wall)

    # --- loop over particles at first step ---
    angles = []
    vnorms = []
    pids   = []

    for pid, x_part, y_part, z_part, vx_part, vy_part, vz_part in zip(
            ids0, x0, y0, z0, vx0, vy0, vz0):

        # find closest wall segment
        line_idx, closest_point_RZ, distance = closest_line_and_normal(
            surf, x_part, y_part
        )

        # angle between v and normal for that segment
        angle_deg = angle_between_velocity_and_normal(
            surf, line_idx, vx_part, vy_part, vz_part
        )

        # speed magnitude (|v|)
        vnorm = np.sqrt(vx_part**2 + vy_part**2 + vz_part**2)

        pids.append(pid)
        angles.append(angle_deg)
        vnorms.append(vnorm)

    df = pd.DataFrame(
        {
            "id": pids,
            "angle_deg": angles,
            "vnorm": vnorms,
            "droplet": droplet_class,
        }
    )
    return df
