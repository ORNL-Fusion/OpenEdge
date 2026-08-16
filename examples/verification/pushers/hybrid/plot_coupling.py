#!/usr/bin/env python3
"""Stage C gates: volume operators must act identically on GCA-state and
Boris particles. B = -z, so v_par = -vz and the perpendicular plane is xy.

For GCA-mode rows the stored p_gca_vpar is authoritative; for Boris rows
v_par = -vz. Gates are ratios GCA/reference of each operator's own
observable, with an absolute-effect floor so a silent no-op (the pre-hook
behavior: kicks discarded at reconstruction) cannot pass.
"""

import os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
DT = 1e-8


def read_traj(tag):
    per = {}
    with open(os.path.join(OUT, f"traj.{tag}")) as f:
        for line in f:
            if "ITEM: TIMESTEP" in line:
                ts = int(next(f)); next(f); n = int(next(f)); next(f)
                for _ in range(3): next(f)
                next(f)
                for _ in range(n):
                    a = next(f).split()
                    per.setdefault(int(a[0]), []).append(
                        [ts] + [float(q) for q in a[2:]])
    return {pid: np.array(rr) for pid, rr in per.items()}


def vpar_of(r):
    # stored vpar when the row is GCA-mode, else -vz (B = -z)
    return np.where(r[:, 12] > 0.5, r[:, 10], -r[:, 6])


def series(tag):
    """per-frame ensemble stats: mean v_par, var v_par, perp MSD"""
    tr = read_traj(tag)
    ts = sorted({int(t) for r in tr.values() for t in r[:, 0]})
    mv, vv, ms = [], [], []
    for k, t in enumerate(ts):
        vps, d2 = [], []
        for r in tr.values():
            j = np.searchsorted(r[:, 0], t)
            if j >= len(r) or r[j, 0] != t:
                continue
            vps.append(vpar_of(r[j:j+1])[0])
            d2.append((r[j, 1]-r[0, 1])**2 + (r[j, 2]-r[0, 2])**2)
        vps = np.array(vps)
        mv.append(vps.mean()); vv.append(vps.var()); ms.append(np.mean(d2))
    return np.array(ts)*DT, np.array(mv), np.array(vv), np.array(ms)


def main():
    checks = []

    def gate(name, ok, info=""):
        checks.append((name, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f"  ({info})" if info else ""))

    # --- force/thermal: parallel drift rate ------------------------------
    t, mr, _, _ = series("c_force_ref")
    _, mg, _, _ = series("c_force_gca")
    drift_r = mr[-1] - mr[0]
    drift_g = mg[-1] - mg[0]
    gate("force: reference drift is a real signal (> 1 km/s)",
         abs(drift_r) > 1e3, f"ref {drift_r:.0f} m/s")
    gate("force: GCA parallel drift matches Boris within 2%",
         abs(drift_r) > 0 and abs(drift_g/drift_r - 1.0) < 0.02,
         f"gca {drift_g:.0f} vs ref {drift_r:.0f} m/s "
         f"(ratio {drift_g/drift_r:.4f})")

    # --- cross-field diffusion: perpendicular MSD ------------------------
    t, _, _, sr = series("c_cd_ref")
    _, _, _, sg = series("c_cd_gca")
    msd_r, msd_g = sr[-1], sg[-1]
    exp4dt = 4.0 * 0.5 * t[-1]
    gate("cd: reference perp MSD is a real signal (~4 D t)",
         msd_r > 0.3 * exp4dt, f"ref {msd_r:.2e} vs 4Dt {exp4dt:.2e} m^2")
    gate("cd: GCA perp MSD matches Boris within 25% (stochastic)",
         msd_r > 0 and 0.75 < msd_g/msd_r < 1.33,
         f"ratio {msd_g/msd_r:.3f}")

    # --- coulomb/background: v_par diffusion -----------------------------
    t, _, vr, _ = series("c_coul_ref")
    _, _, vg, _ = series("c_coul_gca")
    dvr, dvg = vr[-1] - vr[0], vg[-1] - vg[0]
    gate("coul: reference var(v_par) growth is a real signal",
         dvr > 0.05 * vr[0] if vr[0] > 0 else dvr > 0,
         f"ref dvar {dvr:.2e} (m/s)^2")
    gate("coul: GCA var(v_par) growth matches Boris within 40% (stochastic)",
         dvr > 0 and 0.6 < dvg/dvr < 1.67, f"ratio {dvg/dvr:.3f}")

    ok = all(p for _, p in checks)
    print(("PASS" if ok else "FAIL") + ": verification/pushers/hybrid Stage C")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
