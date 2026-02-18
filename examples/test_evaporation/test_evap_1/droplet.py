import numpy as np
import matplotlib.pyplot as plt
import h5py


from matplotlib.colors import LogNorm

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k, e, m_e, m_p, epsilon_0
from matplotlib.colorbar import make_axes

# Constants
m_i = 6.94 * m_p  # Lithium ion mass (kg)
A = 1.2e6  # Richardson's constant (A/m^2/K^2)
W_inf = 2.9 * e  # Bulk work function in joules


def compute_charge(r_d, T_e, T_i, T_d, n_e, n_i):
    # Define functions
    def f(phi):
        I_e = -np.pi * r_d**2 * n_e * np.sqrt(8 * k * T_e / (np.pi * m_e)) * np.exp(e * phi / (k * T_e))
        I_i = np.pi * r_d**2 * n_i * np.sqrt(8 * k * T_i / (np.pi * m_i)) * (1 - e * phi / (k * T_i))
        J_th = A * T_d**2 * np.exp(-W_inf / (k * T_d) + e**2 / (16 * np.pi * epsilon_0 * r_d * k * T_d))
        return I_e + I_i + 4 * np.pi * r_d**2 * J_th

    def f_prime(phi):
        dI_e_dphi = -np.pi * r_d**2 * n_e * np.sqrt(8 * k * T_e / (np.pi * m_e)) * (e / (k * T_e)) * np.exp(e * phi / (k * T_e))
        dI_i_dphi = -np.pi * r_d**2 * n_i * np.sqrt(8 * k * T_i / (np.pi * m_i)) * (e / (k * T_i))
        return dI_e_dphi + dI_i_dphi

    # Newton-Raphson Iteration
    phi_guess = -1.0  # Initial guess
    tolerance = 1e-6
    max_iter = 200

    phi = phi_guess

    for i in range(max_iter):
        phi_new = phi - f(phi) / f_prime(phi)
        if abs(phi_new - phi) < tolerance:
#            print(f"Converged after {i+1} iterations.")
            break
        phi = phi_new
    else:
        print("Did not converge.")
#    print(phi)
    # Calculate the charge
    Q = 4 * np.pi * epsilon_0 * phi * r_d
    return Q / e  # Charge in elementary units


def update_droplet_state(R, T, Qs=50.E+06, DT=1.0E-02,
                         AM=1.53E-26, Rho=534.0, Cp=4200., DH=3.158E+03,
                         AN=6.022E+23, Pi=3.141, BK=1.380649E-23):
    """
    Update the state of a Li droplet in plasma for one time step.

    Parameters:
        R (float): Current droplet radius in meters.
        T (float): Current droplet temperature in degrees Celsius.
        Qs (float): Heat flux from plasma to droplet surface in W/m^2.
        DT (float): Integration time step in seconds.
        AM (float): Mass of one Li atom in kg.
        Rho (float): Density of Li in kg/m^3.
        Cp (float): Specific heat of Li in J/kg-K.
        DH (float): Latent heat in J/mol.
        AN (float): Avogadro number.
        Pi (float): Pi number.
        BK (float): Boltzmann constant in J/K.

    Returns:
        R_new (float): Updated droplet radius in meters.
        T_new (float): Updated droplet temperature in degrees Celsius.
        Gevap (float): Evaporation flux in kg/m²/s.
        HF (float): Heat flux entering the droplet in W/m².
    """
    # Convert temperature to Kelvin
    TK = T + 273.15

    # Antoine Equation for vapor pressure
    a1 = 5.055
    b1 = -8023.0
    xm1 = 6.939
    vpres1 = 760.0 * 10.0 ** (a1 + b1 / TK)

    # Evaporation flux
    Gevap = 1.0E+04 * 3.513E+22 * vpres1 / np.sqrt(xm1 * TK)

    # Rate of change of radius
    dRdt = -AM * Gevap / Rho

    # Heat flux
    HF = (Qs - Gevap * DH / AN)

    # Rate of change of temperature
    dTdt = 3.0 / Rho / Cp * HF

    # Update radius and temperature
    R_new = R + dRdt * DT
    T_new = T + dTdt * DT

    return R_new, T_new, Gevap, HF



# Initialize droplet state
case = "Ligament" # "Instability"
R = 0.01 #E-6 #2.5E-3  # Initial radius (m)
T0 = 0
T =  500.0  # Initial temperature (Celsius)
T0 =500.0
Qs = 50.E+06  # Heat flux (W/m^2)
DT = 5.0E-02  # Time step (s)

# Time parameters
time_steps = int(8.5/DT)
times = [0.0]  # Time array
radii = [R]  # Radius array
temperatures = [T]  # Temperature array
charges = []  # Charge array

# Iteratively update the droplet state
for n in range(time_steps):
    R, T, Gevap, HF = update_droplet_state(R, T, Qs=Qs, DT=DT)
    times.append((n + 1) * DT)
    radii.append(R)
    temperatures.append(T)
    
    # Convert droplet temperature to Kelvin
    T_d = T + 273.15
    
    # Given values
    T_e = 10.0 * e / k  # Electron temperature in K (1 eV)
    T_i = 10.0 * e / k  # Ion temperature in K (1 eV)
    n_e = 1e19  # Electron density (m^-3)
    n_i = n_e   # Ion density (m^-3)

    charge = compute_charge(R, T_e, T_i, T_d, n_e, n_i)
    charges.append(charge)
    print("charge", charge, "radii", R, "times", n,)
    

# Convert to numpy arrays for plotting
times = np.array(times)
radii = np.array(radii)
temperatures = np.array(temperatures)
charges = np.array(charges)

# Example: Ensure norm_charges uses the magnitude of charges
norm_charges = abs(charges)

# Define log scale limits
maxlog = np.max(norm_charges)  # Maximum value for log scale
minlog = maxlog / 1e7          # Three orders of magnitude smaller

# Ensure consistent array sizes
min_len = min(len(times), len(radii), len(temperatures), len(norm_charges))
times = times[:min_len]
radii = radii[:min_len]
temperatures = temperatures[:min_len]
norm_charges = norm_charges[:min_len]

# Create the plot


#fig, ax1 = plt.subplots(figsize=(10, 6))
fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
ax1.tick_params(axis='both', direction='in')
# Slice data to show every 10th point
times_sampled = times[::10]
radii_sampled = radii[::10]
norm_charges_sampled = norm_charges[::10] * 1e-7

# Plot radius with log-scaled charge color using 'cividis' colormap
sc = ax1.scatter(times_sampled, radii_sampled, c=norm_charges_sampled, cmap='cividis', s=80, edgecolor='k') #, label="Radius (m)")
ax1.plot(times, radii, color="navy", lw=2)  # Keep the full line plot for context
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Radius (m)", color="navy")
ax1.tick_params(axis="y", labelcolor="navy")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.set_ylim(0, max(radii_sampled)+max(radii_sampled)*1e-1)
# Second y-axis for temperature using a contrasting color
ax2 = ax1.twinx()
ax2.plot(times, temperatures, color="darkorange", lw=2)
ax2.set_ylabel("Temperature (°C)", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")

# Lower and resize the colorbar
cbar_ax = fig.add_axes([0.25, 0.3, 0.5, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal", label="Charge ($10^7 e$)")
cbar.ax.tick_params(labelsize=8)  # Reduce tick label size if needed


        
plt.tight_layout(pad=2.0)  # Increase padding
Q_g = 1e-6*Qs
ax1.set_title(f"Droplet lifetime: ($Q_g$ = {Q_g} $MW/m^2$, $T_0$ = {T0}°C)", fontsize=12, weight='bold')

ax1.grid(True, linestyle='--', alpha=0.7)

# Save the figure with improved DPI, tight bounding box, and specified background color
plt.savefig(f"Figs/case_{case}.png",
        dpi=300, bbox_inches="tight", facecolor='white')



plt.show()
