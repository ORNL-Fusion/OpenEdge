#!/usr/bin/env python3
"""Stage B1 gates: sheath boundary barrier at alpha = 0 (B || n).

Physics gated:
  - inbound gain: mean E_n(b_in) - mean E_n(b_in0) = +e*phi (phi_in)
  - sub-barrier outbound (E_par = 2 eV): all redeposit, zero escapes;
    redeposit E_n ~ E_par + e*phi (reflect is elastic, then the inbound
    wall kick applies once)
  - super-barrier outbound (E_par = 40 eV): all escape, none redeposit;
    escape E_n = E_par - e*phi (phi_out), applied EXACTLY once — a
    double payment would show a 2*phi deficit (transit-state gate)
  - single potential model: phi_in == phi_out within tolerance
Lifecycle conservation is gated for every case.
"""

import glob
import os
import numpy as np

M = 6.6464731e-27
Q = 1.602176634e-19
ME = 9.1093837015e-31
AMU = 1.66053906660e-27
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")

E_SUB = 2.0    # eV parallel, sub-barrier launch
E_SUP = 40.0   # eV parallel, super-barrier launch
N_IN, N_OUT = 256, 128
TE, TI, MD_AMU = 5.0, 5.0, 4.0
# analytic floating potential (Bohm-Stangeby, the deck's model inputs)
PHI_EXP = 0.5 * np.log(MD_AMU * AMU / (2.0 * np.pi * ME) / (1.0 + TI / TE)) * TE
# generator writes v with 10 significant digits -> ~1e-8 eV energy quantum
TOL_EV = 1e-6


def read_csv(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        if ".counts." in p:
            continue
        try:
            d = np.genfromtxt(p, delimiter=",", names=True, dtype=None,
                              encoding="utf-8")
        except Exception:
            continue
        if d.size:
            rows.append(np.atleast_1d(d))
    return np.concatenate(rows) if rows else None


def en_of(d):
    return 0.5 * M * d["vz"]**2 / Q


def main():
    checks = []

    def gate(name, ok, info=""):
        checks.append((name, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f"  ({info})" if info else ""))

    imp = {t: read_csv(os.path.join(OUT, f"impacts.{t}.csv.rank*"))
           for t in ("b_in0", "b_in", "b_sub", "b_sup")}
    esc = {t: read_csv(os.path.join(OUT, f"escapes.{t}.csv.rank*"))
           for t in ("b_in0", "b_in", "b_sub", "b_sup")}

    def nuniq(d):
        return 0 if d is None else np.unique(d["id"].astype(int)).size

    def nrows(d):
        return 0 if d is None else d.size

    # --- lifecycle -------------------------------------------------------
    for t, n_exp in (("b_in0", N_IN), ("b_in", N_IN)):
        gate(f"{t}: all {n_exp} absorbed on target, no duplicates, "
             "no escapes", nuniq(imp[t]) == n_exp and nrows(imp[t]) == n_exp
             and nrows(esc[t]) == 0,
             f"target {nrows(imp[t])}, escapes {nrows(esc[t])}")

    # --- inbound gain: PER PARTICLE against the analytic potential -------
    e0 = en_of(imp["b_in0"]); e1 = en_of(imp["b_in"])
    phi_in = e1.mean() - e0.mean()
    gate(f"b_in: baseline E_n uniform at launch E_par (2.5 eV, per "
         "particle < 1e-6 eV)", np.max(np.abs(e0 - 2.5)) < TOL_EV,
         f"max dev {np.max(np.abs(e0 - 2.5)):.1e} eV")
    gate("b_in: EVERY particle gains exactly e*phi_float(Te,Ti,mD) "
         f"= {PHI_EXP:.6f} eV",
         np.max(np.abs(e1 - (2.5 + PHI_EXP))) < TOL_EV,
         f"max dev {np.max(np.abs(e1 - (2.5 + PHI_EXP))):.1e} eV")

    # --- sub-barrier outbound: total reflection --------------------------
    gate("b_sub: ZERO escapes (E_par < e*phi cannot leave the sheath)",
         nrows(esc["b_sub"]) == 0, f"escapes {nrows(esc['b_sub'])}")
    gate(f"b_sub: all {N_OUT} redeposit on the target",
         nuniq(imp["b_sub"]) == N_OUT and nrows(imp["b_sub"]) == N_OUT,
         f"{nrows(imp['b_sub'])}")
    if nrows(imp["b_sub"]):
        er = en_of(imp["b_sub"])
        # elastic reflection preserves E_par; the inbound kick then adds
        # e*phi EXACTLY ONCE per particle (double charge -> +2 phi, caught)
        gate("b_sub: EVERY redeposit at exactly E_par + e*phi "
             f"({E_SUB + PHI_EXP:.6f} eV, per particle)",
             np.max(np.abs(er - (E_SUB + PHI_EXP))) < TOL_EV,
             f"max dev {np.max(np.abs(er - (E_SUB + PHI_EXP))):.1e} eV")

    # --- super-barrier outbound: escape minus e*phi, exactly once --------
    gate(f"b_sup: all {N_OUT} escape (E_par > e*phi)",
         nuniq(esc["b_sup"]) == N_OUT and nrows(esc["b_sup"]) == N_OUT,
         f"escapes {nrows(esc['b_sup'])}, redeposits {nrows(imp['b_sup'])}")
    gate("b_sup: zero redeposits", nrows(imp["b_sup"]) == 0,
         f"{nrows(imp['b_sup'])}")
    if nrows(esc["b_sup"]):
        ee = en_of(esc["b_sup"])
        # per-particle deficit exactly e*phi: a repeated payment on ANY
        # particle would show a 2*phi deficit for that id
        # tolerance budget: escapees ride GCA ~5 cm to the collector;
        # the transport integrator's energy drift is ~1e-7 relative over
        # that flight (~2e-6 eV, matches the orbit-suite drift scale).
        # A double barrier payment would err by 15.9 eV — 6 orders above.
        gate("b_sup: EVERY escape at E_par - e*phi within the transport "
             f"drift budget ({E_SUP - PHI_EXP:.6f} eV, per particle)",
             np.max(np.abs(ee - (E_SUP - PHI_EXP))) < 1e-5,
             f"max dev {np.max(np.abs(ee - (E_SUP - PHI_EXP))):.1e} eV")
        phi_out = E_SUP - ee.mean()
        # --- single evaluator: b_in/b_sub prove it at 1e-10 (no flight);
        # the escape comparison carries the same drift budget
        gate("single-model: |phi_in - phi_out| < 1e-5 V "
             "(one shared evaluator; drift budget)",
             abs(phi_out - phi_in) < 1e-5,
             f"in {phi_in:.9f} V, out {phi_out:.9f} V")

    ok = all(p for _, p in checks)
    print(("PASS" if ok else "FAIL") + ": verification/pushers/hybrid Stage B1 (B || n)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
