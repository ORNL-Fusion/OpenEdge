#!/usr/bin/env python3
"""Overlay the three Li source channels on the outer divertor leg vs L - L_sep:
  - physical sputter      (c_cpmiLi)   from compute surface/physical/sputter
  - bulk evaporation      (c_cevap)    from compute surface/chemical/evaporation
  - adatom desorption     (c_cadat)    from compute surface/chemical/adatom

Reads sources.dump (3-column per-surf dump from in.solo / in.coupled) and
plasma.h5 to set L_sep at the outer-leg strike (psi minimum within R-half).

Usage:  python3 plot_li_sources.py [sources.dump]
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import RegularGridInterpolator


def parse_wall(path):
    text = Path(path).read_text().splitlines()
    pts, lns, sec = {}, [], None
    for line in text:
        s = line.strip()
        if s == "Points": sec = "p"; continue
        if s == "Lines":  sec = "l"; continue
        if not s or not s[0].isdigit(): continue
        tok = s.split()
        if sec == "p":
            pts[int(tok[0])] = (float(tok[1]), float(tok[2]))
        elif sec == "l":
            lns.append((int(tok[0]), int(tok[1]), int(tok[2])))
    return pts, lns


def parse_dump_last_frame(path):
    """Parse last frame of a surf-style dump with N data columns. Returns
    (header_cols, dict of {sid: tuple_of_values})."""
    text = Path(path).read_text().splitlines()
    frames, cur = [], []
    for line in text:
        if line.startswith("ITEM: TIMESTEP") and cur:
            frames.append(cur); cur = []
        cur.append(line)
    if cur:
        frames.append(cur)
    if not frames:
        return [], {}, None

    last = frames[-1]
    out, t, in_data, header = {}, None, False, []
    for i, line in enumerate(last):
        s = line.strip()
        if s.startswith("ITEM: TIMESTEP"):
            t = int(last[i + 1]); continue
        if s.startswith("ITEM: SURFS") or s.startswith("ITEM: ATOMS"):
            header = s.split()[2:]
            in_data = True; continue
        if s.startswith("ITEM:"):
            in_data = False; continue
        if in_data and s and s[0].isdigit():
            tok = s.split()
            sid = int(tok[0])
            out[sid] = tuple(float(x) for x in tok[1:])
    return header, out, t


def main():
    base = Path(__file__).resolve().parent
    project = base.parent
    dump = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        project / "coupled_test" / "openedge" / "sources.dump"
    plasma_h5 = project / "coupled_test" / "openedge" / "plasma.h5"
    wall = project / "input" / "wall.surf"
    if not dump.exists():
        sys.exit(f"no dump at {dump}\n  run mpirun first to generate it")

    pts, lns = parse_wall(wall)
    header, rows, tstep = parse_dump_last_frame(dump)
    if not rows:
        sys.exit(f"no surface records in {dump}")
    print(f"  dump columns: id  {' '.join(header)}")
    print(f"  segments: {len(rows)}")

    # Equilibrium for psi -> L_sep
    with h5py.File(plasma_h5, "r") as f:
        eq_r = f["equilibrium/r"][:]; eq_z = f["equilibrium/z"][:]
        eq_psi = f["equilibrium/psi"][:]
        psib = float(f["equilibrium/psib"][()])
    psi_interp = RegularGridInterpolator(
        (eq_z, eq_r), eq_psi, bounds_error=False, fill_value=np.nan)

    # Lower divertor only
    lower_ids = [s for s in rows
                 if pts[next(p1 for sid, p1, p2 in lns if sid == s)][0] < 0]
    seg = []
    for sid in sorted(lower_ids):
        p1, p2 = next((p1, p2) for sidx, p1, p2 in lns if sidx == sid)
        za, ra = pts[p1]; zb, rb = pts[p2]
        zc, rc = 0.5 * (za + zb), 0.5 * (ra + rb)
        L = float(np.hypot(zb - za, rb - ra))
        seg.append((sid, zc, rc, L))
    sids = np.array([s[0] for s in seg])
    Zc = np.array([s[1] for s in seg])
    Rc = np.array([s[2] for s in seg])
    arc = np.zeros(len(seg))
    for k in range(1, len(seg)):
        arc[k] = arc[k - 1] + 0.5 * (seg[k - 1][3] + seg[k][3])

    # Restrict to outer leg (R > midpoint of R range), with strike at min |psi-psib|
    R_split = 0.5 * (Rc.min() + Rc.max())
    is_outer = Rc >= R_split
    psi_at = psi_interp(np.column_stack([Zc, Rc]))
    abs_dpsi = np.abs(psi_at - psib)
    if not is_outer.any():
        sys.exit("no outer-leg segments")
    idx = np.where(is_outer)[0]
    k_strike = idx[int(np.argmin(abs_dpsi[idx]))]
    L_sep = float(arc[k_strike])

    L_rel = arc[is_outer] - L_sep

    # Per-channel arrays — last n columns of header are the source values
    cols = {}
    for sid in sids[is_outer]:
        vals = rows[int(sid)]
        for j, name in enumerate(header[1:]):
            cols.setdefault(name, []).append(vals[j])
    for k in cols:
        cols[k] = np.array(cols[k])

    label_map = {
        "c_cpmiLi": (r"physical sputter $\Gamma_{\rm pmi}$",
                     "C3", "-o"),
        "c_cevap":  (r"bulk evaporation $\Gamma_{\rm evap}$",
                     "C2", "-s"),
        "c_cadat":  (r"adatom desorption $\Gamma_{\rm adat}$",
                     "C0", "-^"),
    }
    # 3 subplots stacked vertically — one channel each. Same x-axis,
    # independent y-axes (each channel can differ by 5+ decades).
    channels = [c for c in ("c_cpmiLi", "c_cevap", "c_cadat") if c in cols]
    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.0 * n + 0.5), dpi=140,
                             sharex=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, channels):
        y = cols[name]
        pos = y > 0
        lab, color, marker = label_map.get(name, (name, None, "-o"))
        if pos.sum() == 0:
            ax.text(0.5, 0.5, f"{lab}: all zero", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="0.6"))
            print(f"  {name}: all zero")
        else:
            ax.plot(L_rel[pos], y[pos], marker, lw=1.6, ms=4, color=color,
                    label=lab)
            ax.set_yscale("log")
            ax.legend(fontsize=9, loc="best")
            print(f"  {name}: max = {y.max():.3e} atoms/m^2/s @ "
                  f"L-Lsep = {L_rel[int(np.argmax(y))]:+.3f} m")
        ax.set_ylabel(r"$\Gamma$ (atoms/m$^2$/s)")
        ax.axvline(0, color="0.5", lw=0.7, ls="--")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(r"$L - L_{\rm sep}$ (m)")
    axes[0].set_title(f"Li sources on outer divertor (step {tstep})")

    out = base / "li_sources_outer.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
