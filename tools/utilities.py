from shapely.geometry import Point, Polygon
import glob


class surface:
    """Class to represent and process surface geometry files.

    Attributes:
        filename (str): Path to the surface file.
        geom_type (str): Type of geometry represented in the file.
        points (dict): Dictionary of point IDs and their coordinates.
        lines (list): List of line definitions including line ID, materials, and point IDs.
        polygon (Polygon): Shapely Polygon object created from the points.
        id (str): Identifier for the surface, derived from filename.
    """

    def __init__(self, filename, geom_type):
        """Initialize a Surface object by reading geometry data from a file."""
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
            for line in data[5:num_points+5]:
                point_id, *point_coords = map(float, line.split())
                self.points[int(point_id)] = tuple(point_coords)
            # read lines
            self.lines = []
            for line in data[num_points+8:]:
                line_parts = line.split()
                materials = line_parts[-1]
                line_id = int(line_parts[0])
                line_points = [int(line_parts[1]), int(line_parts[2])]
                self.lines.append((line_id, str(materials), line_points))
        # create Polygon object from points
        points_list = [list(p) for p in self.points.values()]
        self.polygon = Polygon(points_list)
        # set surface ID
        self.id = filename.split('.')[0]

def parser_density(filename, start_species, end_species):
    """Parse 2D density output files to extract density data across species.

    Args:
        filename (str): Path to the density output file.
        start_species (int): Starting index for species data.
        end_species (int): Ending index for species data.

    Returns:
        dict: A dictionary with timesteps as keys and lists of x, y coordinates and densities for species as values.
    """
    results = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(lines[i + 1].strip())
                number_of_cells = int(lines[i + 3].strip())
                xcs = []
                ycs = []
                density_values = [[] for _ in range(end_species - start_species + 1)]
                for j in range(number_of_cells):
                    cell_data_index = i + 5 + j
                    if cell_data_index >= len(lines):
                        break
                    cell_data_line = lines[cell_data_index]
                    cell_data = cell_data_line.split()
                    if len(cell_data) < end_species + 1:
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


def parse_state(filename):
    """
    Parses a file containing simulation data of particles over multiple timesteps.

    Parameters:
    - filename (str): The path to the file to be parsed.

    Returns:
    - tuple: Contains multiple lists:
        - timesteps (list of int): List of timesteps at which data is recorded.
        - x_coords (list of float): X-coordinates of particles at each timestep.
        - y_coords (list of float): Y-coordinates of particles at each timestep.
        - z_coords (list of float): Z-coordinates of particles at each timestep.
        - vx_coords (list of float): X-components of velocity of particles at each timestep.
        - vy_coords (list of float): Y-components of velocity of particles at each timestep.
        - vz_coords (list of float): Z-components of velocity of particles at each timestep.

    The function reads through the file, identifying sections marked with specific "ITEM" labels
    to extract relevant data points. It handles sections for timesteps, number of atoms, and atom
    properties including position, velocity.
    """
    timesteps = []
    x_coords = []
    y_coords = []
    z_coords = []
    vx_coords = []
    vy_coords = []
    vz_coords = []


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

            elif lines[i].strip().startswith("ITEM: ATOMS"):
                if num_atoms > 0:
                    # Only add timestep if atoms are present
                    timesteps.append(timestep)
                    # Loop to parse all atoms for the current timestep
                    for _ in range(num_atoms):
                        atom_data = lines[i + 1].strip().split()
                        x_coords.append(float(atom_data[2]))
                        y_coords.append(float(atom_data[3]))
                        z_coords.append(float(atom_data[4]))
                        vx_coords.append(float(atom_data[5]))
                        vy_coords.append(float(atom_data[6]))
                        vz_coords.append(float(atom_data[7]))
                        i += 1  # Move to the next atom data line
                i += 1  # Move to the next line after the "ITEM: ATOMS" section

            else:
                i += 1  # Move to next line if no match

    return timesteps, x_coords, y_coords, z_coords, vx_coords, vy_coords, vz_coords

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

