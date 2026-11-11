#!/usr/bin/env python3
"""Tight baseline of the auxiliary-layer coupling at D tau/h^2 = 0.5 (empirical accuracy rule
of this harness, not a stability condition). Budget fixed before running: transfer error
<= 0.005 of the total inventory (engineering tolerance). Per-seed values are printed; a test
is PASS if every seed is within budget, FAIL if the seed mean minus 3 seed-sigma exceeds it,
INCONCLUSIVE otherwise. Exit code 1 on FAIL, 2 on INCONCLUSIVE only."""
import json, sys
from multiprocessing import Pool
import numpy as np
import harness as H

BUDGET = 0.005
TESTS = ("equilibrium", "mode", "pulse_core", "pulse_mc", "mixing")
SEEDS = (1, 2, 3)
KW = dict(T=0.1, nc=50, h=0.02, tau=2e-4, dtp=2e-5, nmark=200000)   # D tau/h^2 = 0.5

def job(a):
    mode, test, seed = a
    k = dict(KW); k["mode"] = mode
    if mode == "brownian": k.pop("h")
    m = H.metrics(test, H.run(test, seed=seed, **k), k["T"]); m.update(mode=mode, test=test, seed=seed)
    return m

if __name__ == "__main__":
    jobs = [(m, t, s) for t in TESTS for m in ("aux", "brownian") for s in SEEDS]
    with Pool(10) as p: out = p.map(job, jobs, chunksize=1)
    json.dump(out, open("baseline.json", "w"), indent=1)
    a0 = next(o for o in out if o["mode"] == "aux")
    print(f"D tau/hc^2 = {a0['Dtau_hc2']:.2f}, D tau/hm^2 = {a0['Dtau_hm2']:.2f}, "
          f"rms marker displacement sqrt(2 D dtp)/hm = {a0['disp_hm']:.3f}, hc = hm = {a0['hm']:g}, dx = {a0['dx']:g}")
    print(f"transfer-error budget {BUDGET} of total inventory")
    worst = 0
    for t in TESTS:
        a = [o for o in out if o["mode"] == "aux" and o["test"] == t]
        b = [o for o in out if o["mode"] == "brownian" and o["test"] == t]
        va = np.array([o["transfer_err"] for o in a]); vb = np.array([o["transfer_err"] for o in b])
        if va.max() <= BUDGET: verdict = "PASS"
        elif va.mean() - 3 * va.std() > BUDGET: verdict = "FAIL"; worst = max(worst, 2)
        else: verdict = "INCONCLUSIVE"; worst = max(worst, 1)
        print(f"{t:12s} transfer aux " + " ".join(f"{v:.4f}" for v in va) + f"  mean {va.mean():.4f} ± {va.std():.4f}"
              f" | Brownian " + " ".join(f"{v:.4f}" for v in vb) + f"  mean {vb.mean():.4f} ± {vb.std():.4f}  {verdict}")
        print(f"{'':12s} dens_rms aux {np.mean([o['dens_rms'] for o in a]):.4f} / Brownian {np.mean([o['dens_rms'] for o in b]):.4f};"
              f" jump aux {np.mean([o['jump'] for o in a]):+.4f} ± {np.std([o['jump'] for o in a]):.4f} / Brownian {np.mean([o['jump'] for o in b]):+.4f} ± {np.std([o['jump'] for o in b]):.4f};"
              f" cons {max(o['cons'] for o in a):.1e}; umin {min(o['umin'] for o in a):.2g}")
    print(("PASS", "INCONCLUSIVE", "FAIL")[worst]); sys.exit((0, 2, 1)[worst])
