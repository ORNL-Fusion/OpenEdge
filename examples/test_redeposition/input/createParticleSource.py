import random
from scipy.constants import k, e, m_p

import math
from matplotlib import pyplot as plt
import numpy as np


def write_openedge_format(filename, x, y, z, vx, vy, vz, min_x, max_x, min_y, max_y, min_z, max_z):
    n_atoms = len(x)
    print(f"Writing {n_atoms} atoms to {filename}")
    print("len vx", len(vx))

    with open(filename, 'w') as file:
        # Write the headers
        file.write("ITEM: TIMESTEP\n")
        file.write("0\n")
        file.write("ITEM: NUMBER OF ATOMS\n")
        file.write(f"{n_atoms}\n")
        file.write("ITEM: BOX BOUNDS pp pp pp\n")
        file.write(f"{min_x} {max_x}\n")
        file.write(f"{min_y} {max_y}\n")
        file.write(f"{min_z} {max_z}\n")
        file.write("ITEM: ATOMS id type x y z vx vy vz\n")
        
        # Assuming all particles have type 3
        particle_type = 1
        for i in range(0,n_atoms):
            print(f"Writing particle {i+1} of {n_atoms}", end='\r')
            file.write(f"{i+1} {particle_type} {x[i]} {y[i]} {z[i]} {vx[i]} {vy[i]} {vz[i]}\n")
            
def cosine_polar_angle_distribution():
    while True:
        theta = math.acos(1 - 2 * random.random())  # Inverse transform sampling
        weight = 2 * math.cos(theta) * math.sin(theta)
        if random.random() < weight / math.sin(theta):  # Rejection sampling
            return theta
            
def thompson_energy_distribution(ub, emax):
    """
    Generate a random number 'e' from a Thompson distribution function F(e, ub, emax).
    The distribution is given by F(e, ub, emax) = const * e / (e + ub)**3, for 0 < e < emax.
    The constant is calculated as const = ub / (0.5 * 1./(emax/ub+1.)**2 - 1./(emax/ub+1.) + 0.5).
    
    :param ub: The parameter UB in the distribution.
    :param emax: The maximum value of e (EMAX).
    :return: A random number 'e' from the Thompson distribution.
    """
    emu = 1.0 / (emax / ub + 1.0)
    betad2 = 1.0 / (emu * emu - emu - emu + 1.0)

    # generate random number
    r = random.uniform(0, 1)
    arg = r / betad2

    energy_sample = ub / (1.0 - math.sqrt(arg)) - ub

    return energy_sample

def uniform_azimuthal_angle_distribution():
    
    return random.uniform(0, 2.0 * math.pi)

def sample_velocity(ub, emax, MASS_PARTICLE):
    E = thompson_energy_distribution(ub, emax)
    print("E", E)
    theta = cosine_polar_angle_distribution()
    phi = uniform_azimuthal_angle_distribution()
    
    E = E * e
    v = math.sqrt(2 * E / (m_p * MASS_PARTICLE))
    
    # Convert spherical coordinates to Cartesian velocity components
    vx = v * math.sin(theta) * math.cos(phi)
    vy = v * math.sin(theta) * math.sin(phi)
    vz = v * math.cos(theta)
    
    return vx, vy, vz

MASS_PARTICLE = 184.
ub = 8.68
te = 5
psi_c = 2 * te/ub
emax = psi_c* ub

#Larmor radius main ion
psi_sheath = 5
B= 2.25 #T
Ti = te * 11605.
Zi = 1
mi = 2.0 * m_p
vth = np.sqrt(k*Ti/mi)
omega_c = Zi*e*B/mi
LarmorRadius_ion = vth / omega_c
lambda_sheath = psi_sheath * LarmorRadius_ion

# Generate particle positions and velocities
min_x, max_x = 0, 1.1
min_y, max_y = 0, 1.1
min_z, max_z = 0, 0.05
npart = 1000
start_z = 5.e-3 +1e-6 # Starting z position for all particles

x_positions = [0.25]* npart
y_positions = [0.25]* npart

#x_positions = [0.25, 0.3, 0.35]  # Different x positions for each particle
#y_positions = [0.25, 0.3, 0.35]  # Different y positions for each particle


z_positions = [start_z] * npart
velocities = [sample_velocity(ub, emax, MASS_PARTICLE) for _ in range(npart)]

# Unpacking velocities for use in the write_openedge_format function
vx_list, vy_list, vz_list = zip(*velocities)


# Write the particles to a file
output_filename = "particles.dat"
write_openedge_format(output_filename, x_positions, y_positions, z_positions, vx_list, vy_list, vz_list, min_x, max_x, min_y, max_y, min_z, max_z)

