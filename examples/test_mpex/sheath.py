#
#
#import numpy as np
#import matplotlib.pyplot as plt
#
## constants
#QE   = 1.602176634e-19   # C
#EPS0 = 8.8541878128e-12  # F/m
#
#def fd_poly_deg(a_deg):
#    a = np.asarray(a_deg, dtype=float)
#    fd = (
#        0.98992
#        + 5.1220e-3  * a
#        - 7.0040e-4  * a**2
#        + 3.3591e-5  * a**3
#        - 8.2917e-7  * a**4
#        + 9.5856e-9  * a**5
#        - 4.2682e-11 * a**6
#    )
#    return np.clip(fd, 0.0, 1.0)
#
#
#def eval_ds_mps(d_m, Te_eV, Ti_eV, ne0_m3, B_T, alpha_deg, mi_kg,
#                pot_mult=3.0):
#    d = np.asarray(d_m, dtype=float)
#    d = np.maximum(d, 0.0)
#
#    if Te_eV <= 0.0 or ne0_m3 <= 0.0 or mi_kg <= 0.0:
#        phi   = np.zeros_like(d)
#        Emag  = np.zeros_like(d)
#        ne    = ne0_m3 * np.ones_like(d)
#        lambdaD = 0.0
#        L_MPS   = 0.0
#        return phi, Emag, ne, lambdaD, L_MPS
#
#    fd  = fd_poly_deg(alpha_deg)
#    pot = pot_mult * Te_eV  # volts
#
#    # Debye length
#    lambdaD = np.sqrt(EPS0 * Te_eV / (ne0_m3 * QE))
#
#    # ion-sound speed and gyro radius
#    cs       = np.sqrt(max(Te_eV + Ti_eV, 0.0) * QE / (2*mi_kg))
#    omega_ci = QE * B_T / mi_kg if B_T > 0.0 else 0.0
#    rho_i    = cs / omega_ci if omega_ci > 0.0 else 1e300
#
#    cs       = np.sqrt((Te_eV + Ti_eV) * QE / (2*mi_kg))
#    omega_ci = QE * B_T / mi_kg #if B_T > 0.0 else 0.0
#    rho_i    = cs / omega_ci #if omega_ci > 0.0 else 1e300
#    
#    # --- angle-dependent magnetic presheath scale along the normal ---
#    theta   = np.deg2rad(alpha_deg)
#    s       = np.sin(theta)
#    s_min   = 0.05          # avoid infinite length at alpha -> 0
#    L_MPS   = rho_i / max(s, s_min)
#
#    # exponentials
#    e_DS  = np.exp(- d / (2.0 * lambdaD))
#    e_MPS = np.exp(- d / max(L_MPS, 1e-300))
#
#    # potential
#    phi = - pot * (fd * e_DS + (1.0 - fd) * e_MPS)
#
#    # |E|
#    E_DS  =  pot * (fd / (2.0 * lambdaD)) * e_DS
#    E_MPS =  pot * ((1.0 - fd) / max(L_MPS, 1e-300)) * e_MPS
#    Emag  = np.abs(E_DS + E_MPS)
#
#    # Boltzmann electrons
#    x = phi / max(Te_eV, 1e-300)
#    x = np.clip(x, -100.0, 50.0)
#    ne = ne0_m3 * np.exp(x)
#
#    return E_DS, E_MPS, phi, Emag, ne, lambdaD, L_MPS,rho_i
#
#
#if __name__ == "__main__":
#    import matplotlib.ticker as mticker
#
#    # --- plasma / geometry parameters ---
#    Te_eV = 5.0
#    Ti_eV = 5.0
#    ne0   = 1e18
#    B_T   = 0.5
#    mi_Ta = 2 * 1.66053906660e-27  # just using 2 amu here
#
#    Z_TARGET_C     = 0.15        # m, target center
#    Z_TARGET_FRONT = Z_TARGET_C - 0.0001  # m, front face (surface)
#
#    # how far upstream you want to go from the surface (sheath+presheath extent)
#    d_max = 0.05   # 5 cm from surface into plasma
#    Npts  = 400
#
#    # z is the coordinate along the plasma column:
#    # from "sheath entrance" (upstream) to the target front
#    z_surf   = Z_TARGET_FRONT
#    z_min    = z_surf - d_max
#    z        = np.linspace(z_min, z_surf, Npts)
#
#    # convert to distance from surface along the normal (what eval_ds_mps expects)
#    d = z_surf - z   # d=0 at surface, d=d_max at sheath entrance
#
#    alphas = [85]  # you can add 45, 0, etc.
#
#    # for now, do a single alpha
#    alpha = alphas[0]
#    E_DS, E_MPS, phi, Emag, ne, lambdaD, L_MPS, rho_i = eval_ds_mps(
#        d, Te_eV, Ti_eV, ne0, B_T, alpha, mi_Ta
#    )
#
#    # parallel model: Te is constant along z in this challenge
#    Te = Te_eV * np.ones_like(z)
#
#    # --- crude "sheath entrance" marker: where ne rises to 99% of upstream ---
#    ne_norm = ne / ne0
#    idx_se  = np.argmax(ne_norm > 0.99)  # first point where ne ~ ne0
#    z_se    = z[idx_se]
#
#    # --- plotting ---
#    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), dpi=300, sharex=True)
#
#    # 1) |E|(z)
#    ax1.plot(z, Emag, color="navy", lw=2, label="|E|(z)")
#    ax1.axvline(z_se, color="k", ls="--", lw=1, label="sheath entrance")
#    ax1.set_ylabel(r"$|E|(z)$ [V/m]", color="navy")
#    ax1.tick_params(axis="y", labelcolor="navy")
#    ax1.tick_params(axis="both", direction="in")
#    ax1.grid(True, linestyle="--", alpha=0.5)
#    ax1.legend(loc="upper right")
#
#    # 2) ne(z) and Te(z)
#    ax2.plot(z, ne, color="darkorange", lw=2, label=r"$n_e(z)$")
#    ax2.axhline(ne0, color="darkorange", ls=":", lw=1, label=r"$n_{e,0}$")
#
#    ax2_t = ax2.twinx()
#    ax2_t.plot(z, Te, color="forestgreen", lw=2, label=r"$T_e(z)$")
#
#    ax2.set_xlabel(r"$z$ [m]")
#    ax2.set_ylabel(r"$n_e(z)$ [m$^{-3}$]", color="darkorange")
#    ax2.tick_params(axis="y", labelcolor="darkorange")
#    ax2.tick_params(axis="both", direction="in")
#    ax2.set_yscale("log")
#    ax2.grid(True, linestyle="--", alpha=0.5)
#
#    ax2_t.set_ylabel(r"$T_e(z)$ [eV]", color="forestgreen")
#    ax2_t.tick_params(axis="y", labelcolor="forestgreen")
#
#    # nicer y formatting for the field
#    fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
#    fmt.set_scientific(False)
#    ax1.yaxis.set_major_formatter(fmt)
#
#    plt.tight_layout(pad=2.0)
#    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- constants ---
QE   = 1.602176634e-19   # C
EPS0 = 8.8541878128e-12  # F/m

def fd_poly_deg(a_deg):
    a = np.asarray(a_deg, dtype=float)
    fd = (
        0.98992
        + 5.1220e-3  * a
        - 7.0040e-4  * a**2
        + 3.3591e-5  * a**3
        - 8.2917e-7  * a**4
        + 9.5856e-9  * a**5
        - 4.2682e-11 * a**6
    )
    return np.clip(fd, 0.0, 1.0)


def eval_ds_mps(d_m, Te_eV, Ti_eV, ne0_m3, B_T, alpha_deg, mi_kg,
                pot_mult=3.0):
    """
    d_m: distance from target surface along the normal (0 at surface, >0 into plasma)
    """
    d = np.asarray(d_m, dtype=float)
    d = np.maximum(d, 0.0)

    if Te_eV <= 0.0 or ne0_m3 <= 0.0 or mi_kg <= 0.0:
        phi   = np.zeros_like(d)
        Emag  = np.zeros_like(d)
        ne    = ne0_m3 * np.ones_like(d)
        lambdaD = 0.0
        L_MPS   = 0.0
        return phi, Emag, ne, lambdaD, L_MPS

    fd  = fd_poly_deg(alpha_deg)
    pot = pot_mult * Te_eV  # volts

    # Debye length
    lambdaD = np.sqrt(EPS0 * Te_eV / (ne0_m3 * QE))

    # ion-sound speed and gyro radius
    cs       = np.sqrt((Te_eV + Ti_eV) * QE / (2*mi_kg))
    omega_ci = QE * B_T / mi_kg
    rho_i    = cs / omega_ci
    
    # angle-dependent magnetic presheath scale along the normal
    theta   = np.deg2rad(alpha_deg)
    s       = np.sin(theta)
    s_min   = 0.05          # avoid infinite length at alpha -> 0
    L_MPS   = rho_i / max(s, s_min)

    # exponentials
    e_DS  = np.exp(- d / (2.0 * lambdaD))
    e_MPS = np.exp(- d / max(L_MPS, 1e-300))

    # potential
    phi = - pot * (fd * e_DS + (1.0 - fd) * e_MPS)

    # |E|
    E_DS  =  pot * (fd / (2.0 * lambdaD)) * e_DS
    E_MPS =  pot * ((1.0 - fd) / max(L_MPS, 1e-300)) * e_MPS
    Emag  = np.abs(E_DS + E_MPS)

    # Boltzmann electrons
    x = phi / max(Te_eV, 1e-300)
    x = np.clip(x, -100.0, 50.0)
    ne = ne0_m3 * np.exp(x)

    return E_DS, E_MPS, phi, Emag, ne, lambdaD, L_MPS, rho_i


# --- MPEX analytic radial profiles (r in meters) ---
R0 = 0.02  # 2 cm nominal radius

def ne_profile(r):
    return 1e19 * np.exp(-(r / R0)**12)

def Te_profile(r):
    # Te(r) = (1 eV) + (4 eV) * exp[-(r / 2cm)^12]
    return 1.0 + 4.0 * np.exp(-(r / R0)**12)


if __name__ == "__main__":

    # geometry
    Z_TARGET_C      = 0.15       # target center
    Z_TARGET_FRONT  = Z_TARGET_C - 0.0001  # front (surface)

    # choose a radius to sample the column (here: on axis)
    r0      = 0.0               # m
    ne0     = ne_profile(r0)
    Te_eV   = Te_profile(r0)
    Ti_eV   = Te_eV             # Ti = Te in the brief
    B_T     = 0.5
    mi_Ta   = 2 * 1.66053906660e-27  # using 2 amu mass for the ion species

    # z-grid: from upstream (sheath entrance region) to target surface
    d_max   = 0.05   # 5 cm extent upstream from surface
    Npts    = 600
    z_surf  = Z_TARGET_FRONT
    z_min   = z_surf - d_max
    z       = np.linspace(z_min, z_surf, Npts)

    # convert to distance from surface (what the DS/MPS model wants)
    d = z_surf - z   # d=0 at surface, d=d_max at upstream side

    alpha  = 85  # angle between B and target normal

    E_DS, E_MPS, phi, Emag, ne, lambdaD, L_MPS, rho_i = eval_ds_mps(
        d, Te_eV, Ti_eV, ne0, B_T, alpha, mi_Ta
    )

    # Te is constant along z in this analytic model for fixed r0
    Te_z = Te_eV * np.ones_like(z)

    # --- define sheath entrance: where phi ~ -Te_eV (Bohm-like) ---
    phi_Bohm = -3*Te_eV  # volts
    idx_se   = np.argmin(np.abs(phi - phi_Bohm))
    z_se     = z[idx_se]

    phi_Bohm2 = -1*Te_eV  # volts
    idx_se2   = np.argmin(np.abs(phi - phi_Bohm2))
    z_se2     = z[idx_se2]
    
    # --- plotting ---
    fig, (axE, axn) = plt.subplots(1, 2, figsize=(6, 4), dpi=300, sharex=True)

    # 1) |E|(z)
    axE.plot(z, Emag, color="navy", lw=2, label="|E|(z)")
    axE.axvline(z_se, color="k", ls="--", lw=1, label="sheath entrance (M≈1)")
    axE.axvline(z_se2, color="r", ls="--", lw=1, label="sheath entrance (M≈3)")

    axE.set_ylabel(r"$|E|(z)$ [V/m]", color="navy")
    axE.tick_params(axis="y", labelcolor="navy", direction="in")
    axE.tick_params(axis="x", direction="in")
    axE.grid(True, linestyle="--", alpha=0.5)
    axE.legend(loc="upper left")

    # nicer y formatting for E
    fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    fmt.set_scientific(False)
    axE.yaxis.set_major_formatter(fmt)

    # 2) ne(z) and Te(z)
    axn.plot(z, ne, color="darkorange", lw=2, label=r"$n_e(z)$")
    axn.axhline(ne0, color="darkorange", ls=":", lw=1, label=r"$n_{e,0}(r_0)$")

    axT = axn.twinx()
    axT.plot(z, Te_z, color="forestgreen", lw=2, label=r"$T_e(z)$")

    axn.set_xlabel(r"$z$ [m]")
    axn.set_ylabel(r"$n_e(z)$ [m$^{-3}$]", color="darkorange")
    axn.tick_params(axis="y", labelcolor="darkorange", direction="in")
    axn.tick_params(axis="x", direction="in")
    axn.set_yscale("log")
    axn.grid(True, linestyle="--", alpha=0.5)

    axT.set_ylabel(r"$T_e(z)$ [eV]", color="forestgreen")
    axT.tick_params(axis="y", labelcolor="forestgreen", direction="in")

    fig.suptitle(
        fr"Sheath along $z$ at $r={r0*100:.1f}$ cm, "
        fr"$n_0(r)={ne0:.1e}\,\mathrm{{m^{{-3}}}}$, $T_e(r)={Te_eV:.1f}$ eV",
        y=0.98
    )
    plt.tight_layout(pad=2.0)
    plt.show()



#
#
##    for i, alpha in enumerate(alphas):
##
##        E_DS, E_MPS,phi, Emag, ne, lambdaD, L_MPS,rho_i = eval_ds_mps(
##            d, Te_eV, Ti_eV, ne0, B_T, alpha, mi_Ta
##        )
##        beta = 90-alpha
##        label = f"α = {beta}°"
##        KB=1.380649e-23
##        E0= 11600*KB*Te_eV/rho_i/QE
##        axes.plot(d, Emag, label=f"{label}")
##        axes.legend()
##    axes.set_xlim(0,10)
##    axes.set_ylim(0,.5)
##
##    plt.tight_layout()
##    plt.show()
#    
#exit()
#if __name__ == "__main__":
#    d = np.linspace(0.0, 0.0643871715567022, 1000)  # 0 → 5 mm
#    d = np.logspace(-6, -1, 100)  # 0 → 5 mm
#
#    Te_eV = 20 #5.0
#    Ti_eV = 20 #5.0
#    ne0   = 1e18 #1e19
#    B_T   = .1 #0.5
#    mi_Ta = 2 * 1.66053906660e-27
##    mi_Ta = 181.0 * 1.66053906660e-27
#
#    alphas = [85,87,88]
##    alphas = [5, 45, 85]
#
#    fig, axes = plt.subplots(
#        len(alphas), 3, figsize=(12, 8),
#        sharex=True, sharey=True
#    )
#
#    for i, alpha in enumerate(alphas):
#        ax_phi, ax_E, ax_ne = axes[i]
#
#        E_DS, E_MPS,phi, Emag, ne, lambdaD, L_MPS,rho_i = eval_ds_mps(
#            d, Te_eV, Ti_eV, ne0, B_T, alpha, mi_Ta
#        )
##        print("EDS 1e5", max(np.unique(E_DS)/1e5), "EMPS 1e4", max(np.unique(E_MPS))/1e4, 'lambdaD',lambdaD*1e6,'L_MPS', L_MPS*1e6)
###        exit()
#        label = f"α = {alpha}°"
#
#        # basic curves
##        ax_phi.plot(d*1e3, phi, label=label)
#        KB=1.380649e-23
#        E0= 11600*KB*Te_eV/rho_i/QE
##        print(rho_i)
#        
##        print(E0)
##        ax_E.semilogy(d*1e3, Emag)
#        ax_E.plot(d/rho_i, Emag/E0)
##        ax_ne.semilogy(d*1e3, ne)
##
##        # alpha-dependent region markers
##        ds_edge  = 2.0 * lambdaD        # Debye sheath thickness
##        mps_edge = L_MPS                # MPS scale
##        se_edge  = 3.0 * L_MPS          # "sheath entrance" ~ few L_MPS
##
##        for ax in (ax_phi, ax_E, ax_ne):
##            ax.grid(True, ls="--", alpha=0.4)
##
##            ax.axvspan(0.0, ds_edge*1e3, color="C0", alpha=0.08)
##            ax.axvline(ds_edge*1e3, color="C0", ls="--", alpha=0.8)
##
##            ax.axvspan(ds_edge*1e3, mps_edge*1e3, color="C1", alpha=0.06)
##            ax.axvline(mps_edge*1e3, color="C1", ls="-.", alpha=0.8)
##
##            ax.axvline(se_edge*1e3, color="k", ls=":", alpha=0.6)
##
##        # left labels & row label
##        ax_phi.set_ylabel("φ (V)")
##        ax_E.set_ylabel("|E| (V/m)")
##        ax_ne.set_ylabel("nₑ (m⁻³)")
##        ax_phi.text(
##            0.02, 0.85, label,
##            transform=ax_phi.transAxes,
##            fontsize=10, fontweight="bold"
##        )
#
#    axes[-1,0].set_xlabel("d (mm)")
#    axes[-1,1].set_xlabel("d (mm)")
#    axes[-1,2].set_xlabel("d (mm)")
#
#    axes[0,0].set_title("Sheath potential")
#    axes[0,1].set_title("Field magnitude")
#    axes[0,2].set_title("Electron density")
#    axes[0,0].set_xlim(0,10)
#    axes[0,0].set_ylim(0,.5)
#
#    plt.tight_layout()
#    plt.show()
#
#
#exit()
#import numpy as np
#
#QE   = 1.602176634e-19
#EPS0 = 8.8541878128e-12
#ME   = 9.10938356e-31
#
#def sheath_stangeby(Te_eV, Ti_eV, ne, B_T, alpha_deg, mi_kg):
#    """
#    Return DS/MPS lengths and average fields using the
#    Stangeby/Schmid constant–field slab model.
#    """
#    Te = Te_eV
#    Ti = Ti_eV
#
#    # Debye length
#    lambda_D = np.sqrt(EPS0 * Te / (ne * QE))
#
#    # Ion sound speed and gyro radius
#    cs       = np.sqrt((Te + Ti) * QE / (2*mi_kg))
#    omega_ci = QE * B_T / mi_kg
#    rho_i    = cs / omega_ci
#
#    # magnetic field angle
#    delta = np.deg2rad(alpha_deg)
#
#    # total normalized floating potential e φ_w / T_e
#    Vw = 0.5 * np.log((2.0 * np.pi * ME / mi_kg) * (1.0 + Ti/Te))
#
#    # MPS drop: U_MPS / T_e = - ln(sin δ)
#    V_mps = - np.log(np.sin(delta))
#
#    # clip if it tries to exceed total drop
#    if V_mps > abs(Vw):
#        V_mps = 0.9 * abs(Vw)
#
#    V_ds = Vw - V_mps
#
#    # convert to volts (1 eV ↦ 1 V)
#    dphi_mps = Te * V_mps
#    dphi_ds  = Te * V_ds
#
#    # lengths (Schmid eq. (2))
#    delta =np.pi-delta
#    L_mps = np.sqrt(6.0) * rho_i * np.sin(delta)
#    L_ds  = lambda_D
#
#    # average fields
#    E_mps = dphi_mps / L_mps
#    E_ds  = dphi_ds  / L_ds
#
#    return {
#        "lambda_D":     lambda_D,
#        "rho_i":        rho_i,
#        "L_mps":        L_mps,
#        "Vw":           Vw,
#        "V_mps":        V_mps,
#        "V_ds":         V_ds,
#        "DeltaPhi_MPS": dphi_mps,
#        "DeltaPhi_DS":  dphi_ds,
#        "E_MPS":        E_mps,
#        "E_DS":         E_ds,
#    }
#
#
#if __name__ == "__main__":
#    # MPEX-ish parameters
#    Te_eV = 20.0
#    Ti_eV = 20.0
#    ne0   = 1.0e18      # m^-3
#    B_T   = 0.1         # T
#    mi_D  = 2.0 * 1.66053906660e-27   # deuterium ion mass
#
#    for alpha in [5.0]: #, 30.0, 85.0]:
#        out = sheath_stangeby(Te_eV, Ti_eV, ne0, B_T, alpha, mi_D)
#
#        print(f"\n=== α = {alpha:.1f}° ===")
#        print(f"λ_D        = {out['lambda_D']*1e6:8.2f} µm")
#        print(f"ρ_i        = {out['rho_i']*1e3:8.2f} mm")
#        print(f"L_MPS      = {out['L_mps']*1e3:8.2f} mm")
#        print(f"V_w (norm) = {out['Vw']:8.3f}")
#        print(f"V_MPS      = {out['V_mps']:8.3f}")
#        print(f"V_DS       = {out['V_ds']:8.3f}")
#        print(f"Δφ_MPS     = {out['DeltaPhi_MPS']:8.2f} V")
#        print(f"Δφ_DS      = {out['DeltaPhi_DS']:8.2f} V")
#        print(f"E_MPS      = {out['E_MPS']/1e4:8.2f} ×10^4 V/m")
#        print(f"E_DS       = {out['E_DS']/1e5:8.2f} ×10^5 V/m")
