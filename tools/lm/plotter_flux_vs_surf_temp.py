#!/usr/bin/env python3
"""Plot Li source fluxes versus liquid-metal surface temperature.

Emitted Li flux components from a liquid-metal surface as a function of
surface temperature T_surf, for a range of ad-atom-to-physical-sputter
yield ratios (Y_ad / Y_ps).

This script supports two parameterizations:

1. `paper`: reproduce the standalone analytic trends described around
   Figure 7 of Islam et al., Nucl. Fusion 64 (2024) 056036.
2. `openedge`: match the current OpenEdge liquid-metal implementation in
   `src/OPENEDGE/liquid_metal_strip.h`.

Equations:

    Gamma_Evp  = alpha * P_Li / sqrt(2 pi m_Li k T_surf)
    Gamma_Sput = f_neut * Gamma_D+ * Y_ps
    Gamma_Ad   = f_ad * (Y_ad / Y_ps) * Gamma_D+ * Y_ad
                 / [1 + A * exp(E_eff / (k T_surf))]
    Gamma_Loss = Gamma_Evp + Gamma_Sput + Gamma_Ad

Usage:
    python3 plotter_flux_vs_surf_temp.py --model paper --out fig7_islam.png
    python3 plotter_flux_vs_surf_temp.py --model openedge --out fig7_openedge.png
"""

import argparse
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
})

# Physical constants (SI)
KB_J     = 1.380649e-23
KB_EV    = 8.617333262e-5
M_LI_AMU = 6.94
AMU_KG   = 1.66053906660e-27
M_LI     = M_LI_AMU * AMU_KG
MMHG_PA  = 133.322387415
ATM_PA   = 101325.0

PARAM_SETS = {
    "paper": {
        "antoine_a": 5.055,
        "antoine_b": 8023.0,
        "pressure_unit": "mmHg",
        "alpha": 1.0,
        "f_neut": 0.35,
        "f_ad": 1.,
        "y_ps": 1.0e-3,
        "y_ad": 1.0e-3,
        "A": 6.9e-7,
        "e_eff": 0.9,
        "tsurf_min": 100.0,
        "tsurf_max": 800,
        "title": "Islam et al. (NF 2024) Fig. 7 analytic scan",
        "cases": [
            {"label": r"$Y/Y=1$, $A=1e{-7}$, $E_\mathrm{eff}=0.9$",
             "y_ratio": 1.0, "A": 1.0e-7, "e_eff": 0.9, "color": "C3", "ls": "-"},
        ],
    },
    "openedge": {
        "antoine_a": 5.66797,
        "antoine_b": 8310.41,
        "pressure_unit": "atm",
        "alpha": 1.0,
        "f_neut": 0.35,
        "f_ad": 1.0,
        "y_ps": 1.0e-3,
        "y_ad": 1.0e-3,
        "A": 1.0e-7,
        "e_eff": 0.9,
        "tsurf_min": 100.0,
        "tsurf_max": 800.0,
        "title": "OpenEdge liquid-metal source model",
        "cases": [
            {"label": r"$Y/Y=50$", "y_ratio": 50.0, "A": 1.0e-7, "e_eff": 0.9, "color": "C3", "ls": "-"},
            {"label": r"$Y/Y=100$", "y_ratio": 100.0, "A": 1.0e-7, "e_eff": 0.9, "color": "C1", "ls": "-"},
        ],
    },
}


def pressure_li_pa(t_surf_k, antoine_a, antoine_b, pressure_unit):
    """Antoine-like Li vapor pressure in Pa."""
    pressure = 10.0 ** (antoine_a - antoine_b / t_surf_k)
    if pressure_unit == "mmHg":
        return pressure * MMHG_PA
    if pressure_unit == "atm":
        return pressure * ATM_PA
    raise ValueError(f"unsupported pressure unit: {pressure_unit}")


def gamma_evp_si(t_surf_k, alpha=1.0, antoine_a=5.055,
                 antoine_b=8023.0, pressure_unit="mmHg"):
    """Hertz-Knudsen evaporation flux [m^-2 s^-1] in SI units."""
    p = pressure_li_pa(t_surf_k, antoine_a, antoine_b, pressure_unit)
    return alpha * p / np.sqrt(2.0 * np.pi * M_LI * KB_J * t_surf_k)


def gamma_evp_legacy(t_surf_k, alpha=1.0):
    """Legacy OpenEdge evaporation formula used in the old droplet model."""
    a1 = 5.055
    b1 = -8023.0
    xm1 = 6.939
    vpres_mmHg = 760.0 * 10.0 ** (a1 + b1 / t_surf_k)
    return alpha * (1.0e4 * 3.513e22) * vpres_mmHg / np.sqrt(xm1 * t_surf_k)


def gamma_sput(gamma_d, y_ps, f_neut=0.35):
    """Physical-sputter flux [m^-2 s^-1]; T-independent."""
    return f_neut * gamma_d * y_ps


def gamma_ad(t_surf_k, gamma_d, y_ad, y_ratio, e_eff_ev,
             A=6.9e-7, f_ad=0.35):
    """Ad-atom desorption flux [m^-2 s^-1] with thermal-saturation denominator."""
    denom = 1.0 + A * np.exp(e_eff_ev / (KB_EV * t_surf_k))
    return f_ad * y_ratio * gamma_d * y_ad / denom


def main(argv):
    ap = argparse.ArgumentParser(
        description="Plot Li surface-source fluxes vs Tsurf.")
    ap.add_argument("--model", choices=sorted(PARAM_SETS), default="paper",
                    help="Parameter set to use: paper or openedge")
    ap.add_argument("--gamma-d", type=float, default=None,
                    help="Incident D+ flux Gamma_D+ in m^-2 s^-1")
    ap.add_argument("--y-ps", type=float, default=None,
                    help="Physical sputter yield Y_ps")
    ap.add_argument("--y-ad", type=float, default=None,
                    help="Ad-atom yield Y_ad")
    ap.add_argument("--y-ratio", type=float, nargs="+", default=None,
                    help="Optional override for Y_ad/Y_ps values")
    ap.add_argument("--A", type=float, default=None,
                    help="Fitting coefficient A in ad-atom denominator")
    ap.add_argument("--e-eff", type=float, default=None,
                    help="Effective ad-atom binding energy E_eff in eV")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Evaporation sticking coefficient")
    ap.add_argument("--f-neut", type=float, default=None,
                    help="Neutral fraction for sputtered Li")
    ap.add_argument("--f-ad", type=float, default=None,
                    help="Neutral fraction for ad-atom Li")
    ap.add_argument("--tsurf-min", type=float, default=None,
                    help="Min T_surf in deg C")
    ap.add_argument("--tsurf-max", type=float, default=None,
                    help="Max T_surf in deg C")
    ap.add_argument("--ymin", type=float, default=1.0e19,
                    help="y-axis lower limit (default 1e19)")
    ap.add_argument("--ymax", type=float, default=1.0e25,
                    help="y-axis upper limit (default 1e25)")
    ap.add_argument("--out", default="islam_fig7.png", help="Output PNG path")
    args = ap.parse_args(argv)

    defaults = PARAM_SETS[args.model]
    gamma_d = args.gamma_d if args.gamma_d is not None else 1.0e24
    y_ps = args.y_ps if args.y_ps is not None else defaults["y_ps"]
    y_ad = args.y_ad if args.y_ad is not None else defaults["y_ad"]
    A = args.A if args.A is not None else defaults["A"]
    e_eff = args.e_eff if args.e_eff is not None else defaults["e_eff"]
    alpha = args.alpha if args.alpha is not None else defaults["alpha"]
    f_neut = args.f_neut if args.f_neut is not None else defaults["f_neut"]
    f_ad = args.f_ad if args.f_ad is not None else defaults["f_ad"]
    tsurf_min = args.tsurf_min if args.tsurf_min is not None else defaults["tsurf_min"]
    tsurf_max = args.tsurf_max if args.tsurf_max is not None else defaults["tsurf_max"]

    # Sweep T_surf
    t_c = np.linspace(tsurf_min, tsurf_max, 401)
    t_k = t_c + 273.15

    # T-independent components
    g_sput = gamma_sput(gamma_d, y_ps, f_neut=f_neut)
    if args.model == "paper":
        g_evp = gamma_evp_legacy(t_k, alpha=alpha)
    else:
        g_evp = gamma_evp_si(
            t_k, alpha=alpha, antoine_a=defaults["antoine_a"],
            antoine_b=defaults["antoine_b"], pressure_unit=defaults["pressure_unit"])

    # Plot
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    # Sputter (flat horizontal line)
    ax.semilogy(t_c, np.full_like(t_c, g_sput),
                color="C2", ls="-", lw=2.2,
                label=r"$\Gamma_\mathrm{Sput}$ (T-indep.)")

    # Evaporation (single line; same for all Y/Y values)
    ax.semilogy(t_c, g_evp,
                color="C0", ls="-", lw=2.2,
                label=r"$\Gamma_\mathrm{Evp}$")

    # Ad-atom curves: explicit paper-style cases rather than a long ratio scan.
    cases = defaults["cases"]
    if args.y_ratio is not None:
        cases = [
            {"label": fr"$Y/Y={yr:g}$", "y_ratio": yr, "A": A, "e_eff": e_eff,
             "color": f"C{idx+3}", "ls": "-"}
            for idx, yr in enumerate(args.y_ratio)
        ]

    for case in cases:
        g_ad = gamma_ad(t_k, gamma_d, y_ad, case["y_ratio"], case["e_eff"],
                        A=case["A"], f_ad=f_ad)
        ax.semilogy(t_c, g_ad,
                    color=case["color"], ls=case["ls"], lw=2.0,
                    label=rf"$\Gamma_\mathrm{{Ad}}$ {case['label']}")

    ax.set_xlabel(r"$T_\mathrm{surf}\ (^\circ\mathrm{C})$")
    ax.set_ylabel(r"emitted Li flux $(\mathrm{m^{-2}\,s^{-1}})$")
    ax.set_xlim(tsurf_min, tsurf_max)
    ax.set_ylim(args.ymin, args.ymax)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", ncol=1, framealpha=0.9)

    if args.model == "paper":
        title = (rf"{defaults['title']}, "
                 rf"$\Gamma_{{D^+}} = {gamma_d:.0e}\ \mathrm{{m^{{-2}}\,s^{{-1}}}}$")
    else:
        title = (rf"{defaults['title']}, "
                 rf"$\Gamma_{{D^+}} = {gamma_d:.0e}\ \mathrm{{m^{{-2}}\,s^{{-1}}}}$, "
                 rf"$E_\mathrm{{eff}} = {e_eff:.2f}$ eV, $A = {A:.1e}$")
    ax.set_title(title, fontsize=11)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
