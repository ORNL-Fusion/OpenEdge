import numpy as np


Lx = Ly = 0.10
xc, yc = Lx / 2, Ly / 2      # 0.05, 0.05
R0     = 0.02                # match R_src
Gamma0 =1e20
NI, NJ = 100, 100                # tiny test mesh
x_vals = np.linspace(0.0, Lx, NI)
y_vals = np.linspace(0.0, Ly, NJ)

T_K = 5.0 * 11600.0          # 5 eV in K (if you need it)

with open("ta_inlet.dat", "w") as f:
    # boundary-ID: must match the last arg in your fix emit/face/file line
    # e.g. fix TaIn emit/face/file TantalumSource xlo ta_inlet.dat XLO ...
    f.write("#MPEX inlet case\n")
    f.write("\n")
    f.write("ZLO\n")

    # mesh sizes: Ni and Nj on SAME line
    f.write(f"NIJ {NI} {NJ}\n")

    # number of values per mesh point
    f.write("NV 4\n")
    f.write("VALUES nrho vz temp gamma\n")
    # mesh coordinates
    f.write("IMESH " + " ".join(f"{x:.6e}" for x in x_vals) + "\n")
    f.write("JMESH " + " ".join(f"{y:.6e}" for y in y_vals) + "\n")
    f.write("\n")
    # data lines: I, J, nrho, vz, temp
    for i, x in enumerate(x_vals, start=1):
        for j, y in enumerate(y_vals, start=1):
            r = np.sqrt((x - xc)**2 + (y - yc)**2)
            if r <= 0.03:  # inside column
                nrho = 1e17 * np.exp(-(r / R0)**12)
                gamma = Gamma0 * np.exp(-(r / R0)**12)
            else:
                nrho = 0.0
                gamma = 0.0
            vz = 1000.0
            f.write(f"{i} {j} {nrho:.6e} {vz:.6e} {T_K:.6e} {gamma:.6e}\n")

exit()
import numpy as np
import matplotlib.pyplot as plt

# ----- parameters from the MPEX spec -----
R0 = 0.02             # 2 cm [m]   (super-Gaussian radius)
Gamma0 = 1e20         # peak flux [m^-2 s^-1]
vz = 1000.0           # axial speed [m/s]
n0 = Gamma0 / vz      # central density [m^-3] -> 1e17

# ----- radial grid -----
r_max = 0.04          # out to 4 cm just to see the tail
r = np.linspace(0.0, r_max, 500)   # [m]

# ----- profiles -----
Gamma = Gamma0 * np.exp(-(r / R0)**12)   # flux profile
n = n0 * np.exp(-(r / R0)**12)           # density profile
ratio = Gamma / n                        # should be identically = vz

# ----- plotting -----
fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

# 1) flux
axes[0].semilogy(r * 100.0, Gamma)
axes[0].set_ylabel(r"$\Gamma_z(r)$ [m$^{-2}$ s$^{-1}$]")
axes[0].set_title("MPEX Ta source: flux, density, and Γ/n")

# 2) density
axes[1].semilogy(r * 100.0, n)
axes[1].set_ylabel(r"$n_z(r)$ [m$^{-3}$]")

# 3) ratio = Γ / n (should be constant = vz)
axes[2].plot(r * 100.0, ratio)
axes[2].axhline(vz, linestyle="--")  # reference line
axes[2].set_ylabel(r"$\Gamma_z/n_z$ [m/s]")
axes[2].set_xlabel(r"radius $r$ [cm]")

for ax in axes:
    ax.grid(True, which="both", linestyle=":")

plt.tight_layout()
plt.show()
