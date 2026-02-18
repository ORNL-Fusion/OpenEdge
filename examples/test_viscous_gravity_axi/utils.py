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

