from shapely.geometry import Point, Polygon
import glob

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
    mass=[]
    temp =[]
    radius  = []
    ids = []
#        timesteps, x_coords, y_coords, z_coords, vx_coords, vy_coords, vz_coords, mass, temp, radius

    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() == "ITEM: TIMESTEP":
                timestep = int(lines[i + 1].strip())
                i += 2  # Move to next line after timestep

            elif lines[i].strip() == "ITEM: NUMBER OF ATOMS":
                num_atoms = int(lines[i + 1].strip())
                i += 2  # Move to the line after the number of atoms

            elif lines[i].strip() == "ITEM: ATOMS id type x y z vx vy vz v_pmass temp radius":
                if num_atoms > 0:
                    # Only add timestep if atoms are present
                    timesteps.append(timestep)
                    # Loop to parse all atoms for the current timestep
                    for _ in range(num_atoms):
                        atom_data = lines[i + 1].strip().split()
                        ids.append(int(atom_data[0]))  # atom_data[0]=id, [1]=type
                        x_coords.append(float(atom_data[2]))
                        y_coords.append(float(atom_data[3]))
                        z_coords.append(float(atom_data[4]))
                        vx_coords.append(float(atom_data[5]))
                        vy_coords.append(float(atom_data[6]))
                        vz_coords.append(float(atom_data[7]))
                        mass.append(float(atom_data[8]))
                        temp.append(float(atom_data[9]))
                        radius.append(float(atom_data[10]))
                        i += 1  # Move to the next atom data line
                i += 1  # Move to the next line after the "ITEM: ATOMS" section

            else:
                i += 1  # Move to next line if no match

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
        
