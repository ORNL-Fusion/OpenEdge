#!/usr/bin/env python3
"""Stage-D1 gates: A1 in production geometry (axi ring, 45-deg B).

The axi mover collides pure-GCA particles at their CHORD velocity (the
retrace contract xnew = x + dtremain*v forbids carrying the gyro
reconstruction in v), so a0 impact statistics are provably wrong — the
collision-site materialization (gc_wall flux) must repair them to within
the kick-drift Boris reference's own convergence band.

Tags: d_ref/d_ref2 (axi kick-drift Boris at dt, dt/2), d_a0 (GCA chord
impact — must FAIL the distribution gates), d_a1 (GCA + flux operator).
Frame: slot x = Z is the wall normal; E_n = m vx^2 / 2.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_impact import read_csv, read_traj, ks_stat, M, Q, OUT

NENS = 256
CFG = {"d_ref": 5e-10, "d_ref2": 2.5e-10, "d_a0": 1e-8, "d_a1": 1e-8}
QS = [0.10, 0.50, 0.90]


def impact_obs(imp):
    vn = imp["vx"]                       # slot x = Z = wall normal
    v = np.sqrt(imp["vx"]**2 + imp["vy"]**2 + imp["vz"]**2)
    en = 0.5 * M * vn**2 / Q
    et = 0.5 * M * v**2 / Q
    ang = np.degrees(np.arccos(np.clip(np.abs(vn) / v, 0, 1)))
    return en, ang, et, imp["y"]


def main():
    checks = []

    def gate(name, ok, info=""):
        checks.append((name, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f"  ({info})" if info else ""))

    imp, fin = {}, {}
    for t in CFG:
        imp[t] = read_csv(os.path.join(OUT, f"impacts.{t}.csv.rank*"))
        _, _, fin[t] = read_traj(t, with_final=True)

    # --- HARD lifecycle gates -------------------------------------------
    for t in CFG:
        if imp[t] is None:
            gate(f"{t}: impact log present", False)
            continue
        ids = imp[t]["id"].astype(int)
        uids, cnt = np.unique(ids, return_counts=True)
        gate(f"{t}: no duplicate vanish IDs", uids.size == ids.size)
        gate(f"{t}: all {NENS} absorbed",
             uids.size == NENS and len(fin[t]) == 0,
             f"vanish {uids.size} final-alive {len(fin[t])}")
    if any(imp[t] is None for t in CFG):
        print("FAIL: verification/pushers/hybrid Stage D1")
        return 1
    obs = {t: impact_obs(imp[t]) for t in CFG}

    # --- branch proof: the impacts happened in the claimed mode ----------
    gate("d_ref: all impacts in Boris mode",
         np.all(imp["d_ref"]["gca_mode"] < 0.5))
    for t in ("d_a0", "d_a1"):
        gate(f"{t}: all impacts in GCA mode",
             np.all(imp[t]["gca_mode"] > 0.5))

    # --- distribution gates: band from the reference's own convergence ---
    rngq = np.random.default_rng(7)
    names = ["E_n", "angle"]
    for i, k in enumerate(names):
        ks_self = ks_stat(obs["d_ref"][i], obs["d_ref2"][i])
        boot = np.array([ks_stat(rngq.choice(obs["d_ref"][i], NENS),
                                 obs["d_ref"][i]) for _ in range(200)])
        band = 1.3 * max(ks_self, np.quantile(boot, 0.95))
        ks0 = ks_stat(obs["d_a0"][i], obs["d_ref"][i])
        ks1 = ks_stat(obs["d_a1"][i], obs["d_ref"][i])
        gate(f"a0 {k} FAILS the band (chord impact is wrong)",
             ks0 > band, f"KS {ks0:.3f} vs band {band:.3f}")
        gate(f"a1 {k} within the reference band",
             ks1 <= band, f"KS {ks1:.3f} vs band {band:.3f}")
        gate(f"a1 improves on a0 for {k}", ks1 < ks0,
             f"{ks1:.3f} < {ks0:.3f}")

    # total impact energy: the chord collapses it; A1 must restore it
    med_r = np.median(obs["d_ref"][2])
    med_1 = np.median(obs["d_a1"][2])
    gate("a1 median total impact energy within 10% of ref",
         abs(med_1/med_r - 1.0) < 0.10,
         f"a1 {med_1:.2f} vs ref {med_r:.2f} eV")

    # --- figure ----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for t, c in [("d_ref", "k"), ("d_a0", "tab:red"), ("d_a1", "tab:blue")]:
        ax[0].hist(obs[t][0], bins=30, histtype="step", color=c, label=t)
        ax[1].hist(obs[t][1], bins=30, histtype="step", color=c, label=t)
        ax[2].hist(obs[t][3], bins=30, histtype="step", color=c, label=t)
    ax[0].set_xlabel("E_n (eV)"); ax[1].set_xlabel("angle (deg)")
    ax[2].set_xlabel("impact R (m)")
    ax[0].legend(fontsize=8)
    fig.suptitle("Stage D1: axi ring impacts — ref vs chord (a0) vs flux (a1)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stageD_summary.png"), dpi=130)
    print(f"Wrote {os.path.join(OUT, 'stageD_summary.png')}")

    ok = all(p for _, p in checks)
    print(("PASS" if ok else "FAIL") + ": verification/pushers/hybrid Stage D1")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
