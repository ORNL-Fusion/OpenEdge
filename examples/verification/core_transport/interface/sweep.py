#!/usr/bin/env python3
"""Interface harness: baseline table with acceptance checks, then resolution sweeps.

    python sweep.py            # everything; exit code 1 if an acceptance check fails
Results in results.json. Acceptance (aux coupling, baseline resolution, 3 seeds):
  per-label conservation < 1e-10, densities finite and never negative at any update,
  bank < one marker weight, and density rms / interface jump / transfer errors within
  3 sigma of the full-domain Brownian noise floor plus TOL. The legacy analogue is
  reported for comparison and is expected to fail.
"""
import json, sys
from multiprocessing import Pool
import numpy as np
import harness as H

TESTS = ("equilibrium", "mode", "pulse_core", "pulse_mc", "mixing")
BASE = dict(T=0.1, nc=50, dtp=2e-5, tau=2e-4, nmark=200000)
TOL = 0.01
KEYS = ("dens_rms", "jump", "transfer_err")

def job(a):
    name, test, kw, seed = a
    k = dict(BASE); k.update(kw)
    r = H.run(test, seed=seed, **k)
    m = H.metrics(test, r, k["T"]); m.update(name=name, test=test, seed=seed)
    return m

def stat(out, name, test, key):
    v = [abs(q[key]) if key != "jump" else q[key] for q in out if q["name"] == name and q["test"] == test]
    return float(np.mean(v)), float(np.std(v))

if __name__ == "__main__":
    jobs = []
    for test in TESTS:
        for seed in (1, 2, 3):
            jobs.append(("brownian", test, dict(mode="brownian"), seed))
            jobs.append(("aux", test, dict(mode="aux"), seed))
            jobs.append(("legacy_analogue_d0", test, dict(mode="legacy", depth=0.0), seed))
            jobs.append(("legacy_analogue_d0.02", test, dict(mode="legacy", depth=0.02), seed))
        # low inventory: few heavy markers, the core layer holds about one marker weight or less
        for seed in (1, 2, 3):
            jobs.append(("aux_low_inventory", test, dict(mode="aux", nmark=2000), seed))
    S = ("equilibrium", "mode", "mixing")
    sweeps = []
    for v in (1e-4, 2e-5, 5e-6): sweeps.append((f"dtp={v:g}", dict(dtp=v)))
    for v, d in ((1e-3, 2e-5), (2e-4, 2e-5), (5e-5, 1e-5)): sweeps.append((f"tau={v:g}", dict(tau=v, dtp=d)))
    for v in (25, 50, 100, 200): sweeps.append((f"nc={v} (h=dx)", dict(nc=v)))
    for v in (25, 50, 100, 200): sweeps.append((f"nc={v} h=0.02", dict(nc=v, h=0.02)))
    for v in (0.01, 0.02, 0.04, 0.08): sweeps.append((f"h={v:g} nc=100", dict(nc=100, h=v)))
    # joint refinement at fixed D tau/h^2 = 0.5 and sqrt(D dtp)/h = 0.2
    for h in (0.04, 0.02, 0.01): sweeps.append((f"joint h={h:g}", dict(nc=100, h=h, tau=0.5 * h * h, dtp=(0.2 * h) ** 2)))
    for name, kw in sweeps:
        for test in S:
            for seed in (1, 2):
                k = dict(mode="aux"); k.update(kw); jobs.append(("aux " + name, test, k, seed))
    with Pool(10) as p: out = p.map(job, jobs, chunksize=1)
    json.dump(out, open("results.json", "w"), indent=1)

    xr1, u1, _, s1 = H.reference("mode", 0.1, 50, n=1000); xr2, u2, _, s2 = H.reference("mode", 0.1, 50, n=2000)
    refconv = float(np.max(np.abs(s1 - s2)) / (u2[0].sum() * (xr2[1] - xr2[0])))
    print(f"reference convergence (mode, n=1000 vs 2000): {refconv:.1e}")

    names = []
    for o in out:
        if o["name"] not in names: names.append(o["name"])
    print(f"{'case':24s} {'test':12s} {'dens_rms':>15s} {'jump':>16s} {'transfer_err':>15s} {'cons':>8s} {'umin':>9s}  Dtau/hc2 step/hm endpoint jumps")
    for nm in names:
        for test in TESTS:
            g = [q for q in out if q["name"] == nm and q["test"] == test]
            if not g: continue
            cols = " ".join(f"{stat(out, nm, test, k)[0]:+8.4f}±{stat(out, nm, test, k)[1]:.4f}" for k in KEYS)
            um = min((q["umin"] if q["umin"] is not None else 0.0) for q in g)
            print(f"{nm:24s} {test:12s} {cols} {max(q['cons'] for q in g):8.1e} {um:9.2g}  "
                  f"{g[0].get('Dtau_hc2', float('nan')):7.2f} {g[0].get('step_hm', float('nan')):7.3f} "
                  f"{g[0]['endpoint_over_inv']:8.2f} {g[0]['jumps_over_inv']:6.2f}")

    fails = []
    def check(ok, msg):
        print(("  ok   " if ok else "  FAIL ") + msg); fails.append(not ok)
    print("acceptance (aux, baseline):")
    check(refconv < 1e-3, f"reference converged ({refconv:.1e})")
    for nm in ("aux", "aux_low_inventory"):
        g = [q for q in out if q["name"] == nm]
        check(max(q["cons"] for q in g) < 1e-10, f"{nm}: per-label conservation {max(q['cons'] for q in g):.1e}")
        check(all(q["finite"] for q in g) and min(q["umin"] for q in g) >= 0.0,
              f"{nm}: finite and non-negative at every update (min {min(q['umin'] for q in g):.2g})")
        check(max(q["bank_over_w0"] for q in g) < 1.0, f"{nm}: bank below one marker weight")
    for test in TESTS:
        for k in KEYS:
            a, sa = stat(out, "aux", test, k); b, sb = stat(out, "brownian", test, k)
            lim = abs(b) + 3 * sb + TOL
            check(abs(a) <= lim, f"{test:12s} {k:12s} aux {a:+.4f} vs noise floor {b:+.4f}±{sb:.4f} (limit {lim:.4f})")
    print("PASS" if not any(fails) else "FAIL"); sys.exit(1 if any(fails) else 0)
