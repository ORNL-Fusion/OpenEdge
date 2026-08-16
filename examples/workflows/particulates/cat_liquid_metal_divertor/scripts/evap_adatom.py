# Author: Abdourahmane Diaw (diawa @ornl.gov, Oak Ridge National Laboratory)
# Copyright (c) 2023-2026 Oak Ridge National Laboratory.
# License: GPL-2.0, following SPARTA. The full license text is available at:
#   - https://github.com/ORNL-Fusion/OpenEdge/blob/main/LICENSE

"""Li atomic-emission flux models for the CAT liquid-metal divertor workflow.

Two source channels are exposed:
  * `li_evap_flux`  — Hertz-Knudsen evaporation from the Antoine vapor pressure.
  * `li_adatom_flux` — Arrhenius-modulated adatom desorption driven by the
    incident D+ flux.

Both mirror the C++ in `src/OPENEDGE/liquid_metal_strip.h` and the
`compute surface/chemical/{evaporation,adatom}` fixes.
"""

import numpy as np

KB        = 1.380649e-23      # Boltzmann constant [J/K]
EV        = 1.602176634e-19   # eV -> J
ATM_TO_PA = 101325.0          # standard atmosphere -> Pa

# Lithium-7
MLI       = 1.1526e-26        # atom mass [kg]  (= 6.94 amu)
ANTOINE_A = 5.66797           # log10(P_atm) = A - B / T_K
ANTOINE_B = 8310.41


def li_evap_flux(T_C, sticking=1.0):
    """Hertz-Knudsen evaporative flux of Li [atoms / m^2 / s].

        Gamma_evap = sticking * P_sat(T) / sqrt(2 * pi * m_Li * k_B * T)

    Antoine is fit above the Li melting point (180.5 degC); we clamp to zero
    below 25 degC where the extrapolation goes negative.
    """
    T_K  = np.asarray(T_C) + 273.15
    P_Pa = 10.0 ** (ANTOINE_A - ANTOINE_B / T_K) * ATM_TO_PA
    flux = sticking * P_Pa / np.sqrt(2.0 * np.pi * MLI * KB * T_K)
    return np.where(T_K < 298.15, 0.0, flux)


def li_adatom_flux(T_C, Gamma_Dp,
                   Yad_D_Li=1e-3, Yad_Yps=1.0,
                   f_ad=1.0, A_arr=1e-7, E_eff=0.9):
    """Li adatom-desorption flux [atoms / m^2 / s].

        Gamma_adat = f_ad * (Yad_Yps / (1 + A * exp(E_eff / E_surf))) * Yad_D_Li * Gamma_Dp

    where E_surf = k_B T / e [eV]. Mirrors `compute surface/chemical/adatom`.
    """
    T_K = np.asarray(T_C) + 273.15
    E_surf_eV = KB * T_K / EV
    denom = 1.0 + A_arr * np.exp(E_eff / E_surf_eV)
    return f_ad * (Yad_Yps / denom) * Yad_D_Li * Gamma_Dp
