#!/usr/bin/env python3
"""
Side-by-side SPARTA vs standalone, inner leg | outer leg.

Layout:
              | inner leg            | outer leg            |
  -----------+-----------------------+-----------------------+
   row 0     | Tsurf vs arc length s | Tsurf vs arc length s |
   row 1     | Γ_evap vs s           | Γ_evap vs s           |
   row 2     | Γ_adatom vs s         | Γ_adatom vs s         |

All three rows share the SOLPS-anchored arc length axis so SPARTA
scatter sits directly on the standalone curves.

Usage:
    python3 plot_sparta_vs_standalone.py \\
        --sparta /path/to/output/surf_fluxes.txt \\
        --standalone-il inner.csv \\
        --standalone-ol outer.csv \\
        --ld-tg-il /path/to/ld_tg_i.dat \\
        --ld-tg-ol /path/to/ld_tg_o.dat \\
        --out compare.png
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
})

# Liquid-Li physical constants (kept in sync with liquid_metal_strip.h)
KB = 1.380649e-23
MLI = 1.1526e-26
ATM_TO_PA = 101325.0
ANTOINE_A = 5.66797
ANTOINE_B = 8310.41


def li_evap_flux(T_C, sticking=1.0):
    T_K = T_C + 273.15
    P = 10 ** (ANTOINE_A - ANTOINE_B / T_K) * ATM_TO_PA
    return sticking * P / np.sqrt(2 * np.pi * MLI * KB * T_K)


def li_arrhenius(T_C, A_arr=1e-7, E_eff=0.9, Yad=1e-3, f_ad=1.0):
    T_K = np.asarray(T_C) + 273.15
    E_surf_eV = KB * T_K / 1.602176634e-19
    return f_ad * Yad / (1.0 + A_arr * np.exp(E_eff / E_surf_eV))


def read_sparta_surf_dump(path):
    with open(path) as f:
        text = f.read()
    blocks = re.split(r"ITEM: TIMESTEP\s*\n\d+\s*\n", text)
    block = blocks[-1]
    m = re.search(r"ITEM: SURFS([^\n]*)\n(.*)", block, flags=re.DOTALL)
    cols = m.group(1).split()
    rows = [line.split()[:len(cols)]
            for line in m.group(2).strip().splitlines() if line.strip()]
    return cols, np.array(rows, dtype=float)


def leg_panels(ax_T, ax_F, sd, sp_data, leg_name, axis_label):
    """Draw two stacked panels for one leg: Tsurf, then Γ_evap + Γ_adatom.

    sd has the field ``x_axis_sd`` (already shifted to the chosen axis);
    sp_data carries ``smid`` remapped onto the same axis by select_leg.
    """
    Tsd = sd["Tsurf_C"]; evap_sd = sd["evap_flux"]; adat_sd = sd["adatom_flux"]
    x_sd = sd["x_axis_sd"]

    Tsp, evp, adp, gdp, smid = sp_data

    # Tsurf vs axis
    ax_T.plot(x_sd, Tsd, color="C0", label="standalone")
    if len(Tsp) > 0:
        ax_T.scatter(smid, Tsp, color="C3", marker="o", s=60, zorder=3,
                     label="OpenEdge")
    ax_T.set_xlabel(axis_label)
    ax_T.set_ylabel(r"$T_\mathrm{surf}\ (^\circ\mathrm{C})$")
    ax_T.set_title(f"{leg_name}: $T_\\mathrm{{surf}}$")
    ax_T.grid(True, alpha=0.3)
    ax_T.legend(loc="best")

    # Γ_evap and Γ_adatom on a shared log axis
    ax_F.semilogy(x_sd, evap_sd, color="C0", lw=2,
                  label=r"standalone $\Gamma_\mathrm{evap}$")
    pos_sd = adat_sd > 0
    if pos_sd.any():
        ax_F.semilogy(x_sd[pos_sd], adat_sd[pos_sd], color="C1", lw=2, ls="--",
                      label=r"standalone $\Gamma_\mathrm{adatom}$")
    if len(Tsp) > 0:
        ax_F.scatter(smid, evp, facecolors="none", edgecolors="C0", s=70,
                     linewidths=1.6, zorder=3,
                     label=r"OpenEdge $\Gamma_\mathrm{evap}$")
        pos_sp = adp > 0
        if pos_sp.any():
            ax_F.scatter(smid[pos_sp], adp[pos_sp], color="C1", marker="^",
                         s=60, zorder=3,
                         label=r"OpenEdge $\Gamma_\mathrm{adatom}$")
    ax_F.set_xlabel(axis_label)
    ax_F.set_ylabel(r"flux $(\mathrm{m^{-2}\,s^{-1}})$")
    ax_F.set_title(f"{leg_name}: $\\Gamma_\\mathrm{{evap}}$ and "
                   r"$\Gamma_\mathrm{adatom}$")
    ax_F.grid(True, which="both", alpha=0.3)
    ax_F.legend(loc="best", fontsize=9)


def solps_arc_for_surf(R_mid, Z_mid, tgt_R, tgt_Z, tgt_s):
    """Return the arc length of the SOLPS data point closest to (R, Z)."""
    d2 = (tgt_R - R_mid) ** 2 + (tgt_Z - Z_mid) ** 2
    return tgt_s[int(np.argmin(d2))]


def load_ld_tg(path):
    """Return (x_target, R_target, Z_target, xMP) from an ld_tg_*.dat file.

    Column layout matches SOLPS-ITER b2plot output: x [1], xMP [15],
    Rclfc [16], Zclfc [17] (1-indexed in the file header)."""
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            t = line.split()
            if len(t) >= 16:
                # x, xMP, Rclfc, Zclfc — 0-indexed: 0, 13, 14, 15
                rows.append([float(t[0]), float(t[13]),
                             float(t[14]), float(t[15])])
    a = np.asarray(rows)
    return a[:, 0], a[:, 2], a[:, 3], a[:, 1]   # x, R, Z, xMP


def select_leg(sp, idx, sid_lo, sid_hi, ld_tg_path=None, axis="s"):
    sid = sp[:, idx["id"]].astype(int)
    Tsp_all = sp[:, idx["s_Tsurf_lm"]]
    mask = (sid >= sid_lo) & (sid <= sid_hi) & (Tsp_all > 0.0)
    if not mask.any():
        return (np.array([]),) * 5
    sub = sp[mask]
    sid_sub = sid[mask]
    order = np.argsort(sid_sub)
    Tsp = sub[:, idx["s_Tsurf_lm"]][order]
    evp = sub[:, idx["c_cevap"]][order]
    adp = sub[:, idx["c_cadat"]][order]
    gdp = sub[:, idx["s_Gamma_D_lm"]][order] if "s_Gamma_D_lm" in idx \
        else np.zeros_like(Tsp)
    v1x = sub[:, idx["v1x"]][order]; v1y = sub[:, idx["v1y"]][order]
    v2x = sub[:, idx["v2x"]][order]; v2y = sub[:, idx["v2y"]][order]

    # SOLPS-anchored axis: each SPARTA surf gets the axis value of its
    # (R, Z)-nearest SOLPS data point. axis="s" -> arc along target,
    # axis="lsep" -> upstream midplane projection (xMP) shifted to put 0
    # at the strike point. Fall back to SPARTA cumulative segment length.
    if ld_tg_path is not None and os.path.isfile(ld_tg_path):
        x, tgt_R, tgt_Z, xMP = load_ld_tg(ld_tg_path)
        if axis == "lsep":
            i_strike = int(np.argmin(np.abs(x)))
            tgt_axis = xMP - xMP[i_strike]
        else:
            tgt_axis = x - x[0]
        Z_mid = 0.5 * (v1x + v2x)        # axi: SPARTA x = Z
        R_mid = 0.5 * (v1y + v2y)        # axi: SPARTA y = R
        smid = np.array([
            solps_arc_for_surf(R_mid[i], Z_mid[i], tgt_R, tgt_Z, tgt_axis)
            for i in range(len(R_mid))
        ])
    else:
        seg_len = np.hypot(v2x - v1x, v2y - v1y)
        smid = np.cumsum(seg_len) - 0.5 * seg_len
    return Tsp, evp, adp, gdp, smid


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparta", required=True)
    ap.add_argument("--standalone-il", required=True)
    ap.add_argument("--standalone-ol", required=True)
    ap.add_argument("--il-range", nargs=2, type=int, default=[117, 124])
    ap.add_argument("--ol-range", nargs=2, type=int, default=[125, 154])
    ap.add_argument("--ld-tg-il", default=None,
                    help="ld_tg_i.dat for SOLPS-anchored x-axis on inner leg")
    ap.add_argument("--ld-tg-ol", default=None,
                    help="ld_tg_o.dat for SOLPS-anchored x-axis on outer leg")
    ap.add_argument("--axis", choices=["s", "lsep"], default="s",
                    help="x-axis: 's' = arc along target, 'lsep' = "
                         "upstream midplane projection l - l_sep (m)")
    ap.add_argument("--out", default="sparta_vs_standalone.png")
    args = ap.parse_args(argv)

    cols, sp = read_sparta_surf_dump(args.sparta)
    idx = {c: i for i, c in enumerate(cols)}
    il = select_leg(sp, idx, *args.il_range,
                    ld_tg_path=args.ld_tg_il, axis=args.axis)
    ol = select_leg(sp, idx, *args.ol_range,
                    ld_tg_path=args.ld_tg_ol, axis=args.axis)

    def remap_standalone(sd_csv, sd_recarray, ld_tg_path, axis):
        """Add an `x_axis_sd` column matching the chosen axis.

        Standalone CSV's `x_m` is the SOLPS target column-1 coordinate
        (negative in PFR). For axis="s" we shift to put 0 at the PFR end,
        i.e. x - x[0]. For axis="lsep" we resample the SOLPS xMP profile
        at each CSV x_m and subtract xMP at the strike point (x = 0).
        """
        x_csv = sd_recarray["x_m"]
        if axis == "lsep" and ld_tg_path and os.path.isfile(ld_tg_path):
            x, _, _, xMP = load_ld_tg(ld_tg_path)
            order = np.argsort(x)
            i_strike = int(np.argmin(np.abs(x)))
            xMP_at_csv = np.interp(x_csv, x[order], xMP[order])
            return xMP_at_csv - xMP[i_strike]
        return x_csv - x_csv[0]

    from numpy.lib.recfunctions import append_fields
    sd_il = np.genfromtxt(args.standalone_il, delimiter=",", names=True,
                          comments="#")
    sd_ol = np.genfromtxt(args.standalone_ol, delimiter=",", names=True,
                          comments="#")
    sd_il = append_fields(sd_il, "x_axis_sd",
                          remap_standalone(args.standalone_il, sd_il,
                                           args.ld_tg_il, args.axis),
                          usemask=False)
    sd_ol = append_fields(sd_ol, "x_axis_sd",
                          remap_standalone(args.standalone_ol, sd_ol,
                                           args.ld_tg_ol, args.axis),
                          usemask=False)

    axis_label = (r"$l - l_\mathrm{sep}$ at OMP (m)"
                  if args.axis == "lsep"
                  else r"target arc length $s$ (m)")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    leg_panels(axes[0, 0], axes[1, 0], sd_il, il,
               "inner leg", axis_label=axis_label)
    leg_panels(axes[0, 1], axes[1, 1], sd_ol, ol,
               "outer leg", axis_label=axis_label)
    fig.suptitle("LM strip: OpenEdge vs. standalone — inner / outer divertor",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
