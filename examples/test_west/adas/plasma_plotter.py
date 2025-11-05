import numpy as np
import matplotlib.pyplot as plt

def parse_file(filename, charge):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Initialize variables
    IMESH = []
    JMESH = []
    ion = []
    rec = []

    # Read the IMESH and JMESH values
    for line in lines:
        if line.startswith('DMESH'):
            IMESH = list(map(float, line.split()[1:]))
        elif line.startswith('TMESH'):
            JMESH = list(map(float, line.split()[1:]))
        elif line[0].isdigit():
            values = line.split()
            ion.append(float(values[2]))  # Assuming 'ion' is the third column
            rec.append(float(values[3]))  # Assuming 'rec' is the fourth column

    # Convert to 2D arrays
    IMESH = np.array(IMESH)
    JMESH = np.array(JMESH)
    NI = len(IMESH)
    NJ = len(JMESH)
    ion = np.array(ion).reshape(NI, NJ)
    rec = np.array(rec).reshape(NI, NJ)

    return IMESH, JMESH, ion, rec

def plot_pcolormesh(IMESH, JMESH, ion, rec, charge):
    rg, zg = np.meshgrid(IMESH, JMESH, indexing='ij')

    plt.figure(figsize=(16, 6))
    
    plt.title(f"{charge}")

    plt.subplot(1, 2, 1)
    plt.pcolormesh(rg, zg, ion, shading='auto', cmap='viridis')
    plt.colorbar(label='ion')
    plt.xlabel('r')
    plt.ylabel('z')
    plt.title('ion')

    plt.subplot(1, 2, 2)
    plt.pcolormesh(rg, zg, rec, shading='auto', cmap='viridis')
    plt.colorbar(label='rec')
    plt.xlabel('r')
    plt.ylabel('z')
    plt.title('rec')


# File path
plt.figure(figsize=(16, 6))
for j in [8]:

    filename = f'adas_rate_8_{j}.txt'

    # Parse the file
    IMESH, JMESH, ion, rec = parse_file(filename, j)
    plt.subplot(1, 2, 1)
    for i in [0, 10,24]:
        plt.plot(JMESH,ion[i,:], label= f"charge {j}")
    plt.legend()
    plt.subplot(1, 2, 2)
    for i in [0, 10,24]:
        plt.plot(JMESH,rec[i,:] , label= f"charge {j}")
plt.legend()
plt.show()
#    # Plot the data
#    plot_pcolormesh(IMESH, JMESH, ion, rec, i)
#plt.show()
    
## Add interpolation functions for bx, by, and bz
#def create_interpolators(IMESH, JMESH, bx, by, bz):
#    r_interpolator = RegularGridInterpolator((IMESH, JMESH), bx.T, bounds_error=False, fill_value=None)
#    phi_interpolator = RegularGridInterpolator((IMESH, JMESH), by.T, bounds_error=False, fill_value=None)
#    z_interpolator = RegularGridInterpolator((IMESH, JMESH), bz.T, bounds_error=False, fill_value=None)
#    return r_interpolator, phi_interpolator, z_interpolator
#
## Create interpolators
#bx_interpolator, by_interpolator, bz_interpolator = create_interpolators(IMESH, JMESH, bx, by, bz)
#
## Function to get Bfields for a given r and z
#def get_Bfields(r, z):
#    B_r = bx_interpolator((r, z))
#    B_phi = by_interpolator((r, z))
#    B_z = bz_interpolator((r, z))
#    return B_r, B_phi, B_z
#
## Example usage
#r, z = 2.07061, 0.592786
#B_r, B_phi, B_z = get_Bfields(r, z)
#print(f"At r = {r}, z = {z}: B_r = {B_r}, B_phi = {B_phi}, B_z = {B_z}")
#
