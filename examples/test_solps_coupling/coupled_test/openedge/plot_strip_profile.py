#!/usr/bin/env python3
"""Plot OpenEdge LM strip profile (Sergey-style smooth output, ~201 stations).

Reads the CSV produced by `fix surface/state/lm ... csv <path>` and
plots Islam Fig 9-style panels (Tsurf vs arc length) plus a Fig 7-style
panel (Li source-flux components vs Tsurf), using the strip's smooth
Smolentsev solution rather than the per-surf SPARTA dump (which is
constrained to ~36 SOLPS data points and shows step structure).

Source-flux components computed analytically from the strip arrays:

    Gamma_evap (already in CSV, from strip's li_evap_flux)
    Gamma_sput = f_neut_sput * Gamma_D * Y_ps
    Gamma_ad   = f_neut_ad   * (Y_ad/Y_ps) * Gamma_D * Y_ad
                 / [1 + A * exp(E_eff / (k T_surf))]

Usage:
    python3 plot_strip_profile.py --csv output/strip_ol.csv \
                                  [--csv-il output/strip_il.csv] \
                                  --out strip_profile.png
"""

import argparse
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
})

KB_EV = 8.617333262e-5


def read_strip_csv(path):
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append({k: float(v) for k, v in r.items()})
    out = {}
    for k in rows[0]:
        out[k] = np.array([r[k] for r in rows])
    return out


def gamma_sput(gamma_d, y_ps, f_neut_sput=0.35):
    return f_neut_sput * gamma_d * y_ps


def gamma_ad(t_c, gamma_d, y_ad, y_ratio, e_eff_ev,
             A=1.0e-7, f_neut_ad=1.0):
    t_k = t_c + 273.15
    denom = 1.0 + A * np.exp(e_eff_ev / (KB_EV * t_k))
    return f_neut_ad * y_ratio * gamma_d * y_ad / denom


def plot_leg(ax_in, ax_T, ax_F, leg, leg_name, args):
    s = leg["s_m"]                    # arc length [m] from inlet
    T = leg["Tsurf_C"]
    GD = leg["Gamma_D_m2s"]
    Ge = leg["Gamma_evap_m2s"]
    Gs = gamma_sput(GD, args.y_ps, f_neut_sput=args.f_neut_sput)
    Ga = gamma_ad(T, GD, args.y_ad, args.y_ratio, args.e_eff,
                  A=args.A, f_neut_ad=args.f_neut_ad)
    Gtot = Ge + Gs + Ga

    common = dict(lw=2.2)

    # Top: incident plasma particle flux (Gamma_D)
    ax_in.semilogy(s, GD, color="C4", **common,
                   label=r"$\Gamma_{D^+}$")
    ax_in.set_ylabel(r"$\Gamma_{D^+}\ (\mathrm{m^{-2}\,s^{-1}})$")
    ax_in.grid(True, which="both", alpha=0.3)
    ax_in.legend(loc="lower right", framealpha=0.9)
    ax_in.set_title(leg_name)

    # Middle: surface temperature
    ax_T.plot(s, T, color="C3", **common,
              label=r"$T_\mathrm{surf}$")
    ax_T.set_ylabel(r"$T_\mathrm{surf}\ (^\circ\mathrm{C})$")
    ax_T.grid(True, which="both", alpha=0.3)
    ax_T.legend(loc="lower right", framealpha=0.9)

    # Bottom: emitted Li flux components
    ax_F.semilogy(s, Ge,   color="C0", **common,
                  label=r"$\Gamma_\mathrm{Evp}$")
    ax_F.semilogy(s, Ga,   color="C1", ls="--", **common,
                  label=rf"$\Gamma_\mathrm{{Ad}}$ ($Y/Y={args.y_ratio:g}$)")
    ax_F.semilogy(s, Gs,   color="C2", ls=":", **common,
                  label=r"$\Gamma_\mathrm{Sput}$")
    ax_F.semilogy(s, Gtot, color="k",  lw=2.6,
                  label=r"$\Gamma_\mathrm{Loss}$")
    ax_F.set_xlabel(r"distance from inlet $s$ (m)")
    ax_F.set_ylabel(r"$\Gamma_\mathrm{Li}\ (\mathrm{m^{-2}\,s^{-1}})$")
    ax_F.grid(True, which="both", alpha=0.3)
    ax_F.legend(loc="lower right", framealpha=0.9)


def main(argv):
    ap = argparse.ArgumentParser(
        description="Plot LM strip profile from fix surface/state/lm csv dump.")
    ap.add_argument("--csv", required=True,
                    help="Outer-leg CSV (or single-leg).")
    ap.add_argument("--csv-il", default=None,
                    help="Optional inner-leg CSV. If omitted, plot only outer.")
    ap.add_argument("--y-ps", type=float, default=1.0e-3,
                    help="Physical sputter yield Y_ps (default 1e-3)")
    ap.add_argument("--y-ad", type=float, default=1.0e-3,
                    help="Ad-atom yield Y_ad (default 1e-3)")
    ap.add_argument("--y-ratio", type=float, default=70.0,
                    help="Y_ad/Y_ps ratio for Gamma_Ad curve (default 70)")
    ap.add_argument("--A", type=float, default=1.0e-7,
                    help="Ad-atom denominator coeff A (default 1e-7, paper D+ fit)")
    ap.add_argument("--e-eff", type=float, default=0.9,
                    help="Effective ad-atom binding energy E_eff in eV (default 0.9)")
    ap.add_argument("--f-neut-sput", type=float, default=0.35,
                    help="Neutral fraction for sputter (default 0.35)")
    ap.add_argument("--f-neut-ad", type=float, default=1.0,
                    help="Neutral fraction for ad-atoms (default 1.0)")
    ap.add_argument("--out", default="strip_profile.png", help="Output PNG path")
    args = ap.parse_args(argv)

    ol = read_strip_csv(args.csv)
    il = read_strip_csv(args.csv_il) if args.csv_il else None

    if il is None:
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 11.0), sharex=True)
        plot_leg(axes[0], axes[1], axes[2], ol, "outer", args)
    else:
        fig, axes = plt.subplots(3, 2, figsize=(13.0, 11.0), sharex="col")
        plot_leg(axes[0, 0], axes[1, 0], axes[2, 0], ol, "outer", args)
        plot_leg(axes[0, 1], axes[1, 1], axes[2, 1], il, "inner", args)

#    fig.suptitle(
#        rf"OpenEdge LM strip profile (Smolentsev, $N_x={len(ol['s_m'])}$ stations)",
#        fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
