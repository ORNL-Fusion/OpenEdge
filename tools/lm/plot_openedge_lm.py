#!/usr/bin/env python3
"""OpenEdge-only LM divertor diagnostics.

Quad layout (2 rows x 2 cols, inner leg | outer leg):
  row 0: T_surf vs l - l_sep (OMP)
  row 1: Γ_evap, Γ_adatom, Γ_phys-sputter on a shared log axis

x-axis is the SOLPS upstream-midplane coordinate (xMP) from
ld_tg_*.dat, shifted so 0 sits at the strike point. Each SPARTA surf
takes the xMP of its (R, Z)-nearest SOLPS data point.
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
    "legend.fontsize": 9,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
})


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


def load_ld_tg(path):
    """Return (x, R, Z, xMP) from an ld_tg_*.dat file."""
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            t = line.split()
            if len(t) >= 16:
                rows.append([float(t[0]), float(t[13]),
                             float(t[14]), float(t[15])])
    a = np.asarray(rows)
    return a[:, 0], a[:, 2], a[:, 3], a[:, 1]   # x, R, Z, xMP


def select_leg(sp, idx, sid_lo, sid_hi, ld_tg_path):
    sid = sp[:, idx["id"]].astype(int)
    Tsp_all = sp[:, idx["s_Tsurf_lm"]]
    mask = (sid >= sid_lo) & (sid <= sid_hi) & (Tsp_all > 0.0)
    if not mask.any():
        return {"smid": np.array([])}
    sub = sp[mask]
    sid_sub = sid[mask]
    order = np.argsort(sid_sub)
    Tsp = sub[:, idx["s_Tsurf_lm"]][order]
    evp = sub[:, idx["c_cevap"]][order]
    adp = sub[:, idx["c_cadat"]][order]
    sput = sub[:, idx["c_cpmiLi"]][order] if "c_cpmiLi" in idx \
        else np.zeros_like(Tsp)
    v1x = sub[:, idx["v1x"]][order]; v1y = sub[:, idx["v1y"]][order]
    v2x = sub[:, idx["v2x"]][order]; v2y = sub[:, idx["v2y"]][order]
    # 2D Cartesian convention (current OpenEdge default): SPARTA x = R, y = Z.
    # The legacy axi convention swapped these; check the deck's `boundary` line
    # if you need to revisit. Pre-2026-04-29 wall.surf files used (x=Z, y=R).
    R_mid = 0.5 * (v1x + v2x)
    Z_mid = 0.5 * (v1y + v2y)

    x, tgt_R, tgt_Z, xMP = load_ld_tg(ld_tg_path)
    i_strike = int(np.argmin(np.abs(x)))
    tgt_axis = xMP - xMP[i_strike]
    smid = np.array([
        tgt_axis[int(np.argmin((tgt_R - R_mid[i]) ** 2 +
                                (tgt_Z - Z_mid[i]) ** 2))]
        for i in range(len(R_mid))
    ])
    o2 = np.argsort(smid)
    return {
        "smid": smid[o2], "T": Tsp[o2], "evp": evp[o2],
        "adp": adp[o2], "sput": sput[o2],
    }


def leg_panels(ax_T, ax_F, leg, leg_name, axis_label):
    if len(leg["smid"]) == 0:
        for ax in (ax_T, ax_F):
            ax.set_title(f"{leg_name}: no data")
        return
    s = leg["smid"]

    ax_T.plot(s, leg["T"], color="C3", marker="o", ms=7, zorder=3)
    ax_T.set_xlabel(axis_label)
    ax_T.set_ylabel(r"$T_\mathrm{surf}\ (^\circ\mathrm{C})$")
    ax_T.set_title(f"{leg_name}: $T_\\mathrm{{surf}}$")
    ax_T.grid(True, alpha=0.3)

    ev = leg["evp"]; ad = leg["adp"]; sp = leg["sput"]
    ax_F.semilogy(s, ev, color="C0", marker="o", ms=7,
                  label=r"$\Gamma_\mathrm{evap}$")
    pos = ad > 0
    if pos.any():
        ax_F.semilogy(s[pos], ad[pos], color="C1", marker="^", ms=8, ls="--",
                      label=r"$\Gamma_\mathrm{adatom}$")
    pos = sp > 0
    if pos.any():
        ax_F.semilogy(s[pos], sp[pos], color="C2", marker="s", ms=7, ls=":",
                      label=r"$\Gamma_\mathrm{phys\,sputter}$")
    ax_F.set_xlabel(axis_label)
    ax_F.set_ylabel(r"flux $(\mathrm{m^{-2}\,s^{-1}})$")
    ax_F.set_title(f"{leg_name}: Li source fluxes")
    ax_F.grid(True, which="both", alpha=0.3)
    ax_F.legend(loc="best")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparta", required=True)
    ap.add_argument("--il-range", nargs=2, type=int, default=[103, 116])
    ap.add_argument("--ol-range", nargs=2, type=int, default=[137, 150])
    ap.add_argument("--ld-tg-il", required=True)
    ap.add_argument("--ld-tg-ol", required=True)
    ap.add_argument("--out", default="openedge_lm.png")
    args = ap.parse_args(argv)

    cols, sp = read_sparta_surf_dump(args.sparta)
    idx = {c: i for i, c in enumerate(cols)}
    il = select_leg(sp, idx, *args.il_range, ld_tg_path=args.ld_tg_il)
    ol = select_leg(sp, idx, *args.ol_range, ld_tg_path=args.ld_tg_ol)

    axis_label = r"$l - l_\mathrm{sep}$ at OMP (m)"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    leg_panels(axes[0, 0], axes[1, 0], il, "inner leg", axis_label)
    leg_panels(axes[0, 1], axes[1, 1], ol, "outer leg", axis_label)
    fig.suptitle("OpenEdge LM divertor — $T_\\mathrm{surf}$ and Li sources",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
