
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
import numpy as np

class surface:
    def __init__(self, filename, geom_type):
        with open(filename, 'r') as f:
            data = f.readlines()[2:]

        num_points = int(data[0].split()[0])
        num_lines  = int(data[1].split()[0])
        print('Number of points: ', num_points)
        print('Number of lines: ', num_lines)

        # ---- points ----
        self.points = {}
        first_point_idx = 5        # as in your current file format
        last_point_idx  = first_point_idx + num_points

        for line in data[first_point_idx:last_point_idx]:
            if not line.strip():
                continue
            point_id, *point_coords = map(float, line.split())
            self.points[int(point_id)] = tuple(point_coords)

        # ---- lines ----
        self.lines = []
        first_line_idx = last_point_idx + 3   # skip blank line + "Lines" + blank
        last_line_idx  = first_line_idx + num_lines

        for line in data[first_line_idx:last_line_idx]:
            parts = line.split()
            if not parts:
                continue
            line_id = int(parts[0])
            p1 = int(parts[1])
            p2 = int(parts[2])
            # if you ever add material as a 4th column, handle it here
            materials = None
            self.lines.append((line_id, materials, [p1, p2]))

        # Polygon from points in file order
        from shapely.geometry import Polygon
        points_list = [self.points[i] for i in sorted(self.points.keys())]
        self.polygon = Polygon(points_list)
        self.id = filename.split('.')[0]

wall = surface("wall.surf", "2D")
Rwall, Zwall = wall.polygon.exterior.xy

#geofile = "mesh.extra"
#d = np.loadtxt(geofile)
#
#wall_r = np.array([d[:, 0], d[:, 2]])
#wall_z = np.array([d[:, 1], d[:, 3]])

fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
#ax.plot(wall_r, wall_z, "k-", lw=2.5, label="mesh.extra")
ax.plot(Rwall, Zwall, "r-", lw=2.5, label="wall.surf")

plt.show()

