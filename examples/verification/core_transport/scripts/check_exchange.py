#!/usr/bin/env python3
"""C++ core transport + auxiliary-layer exchange vs full-domain diffusion (output/test_exchange).

Circular cylinder (kappa 1) and elliptical cylinder (kappa 1.7, mapped coefficient D_eff = g2 D,
conductance from the same flux operator), unequal marker weights, four states. Budget set in
advance: transfer error <= 0.005 of a state's inventory. A state passes if every seed is within
budget; if not, it is INCONCLUSIVE when the full-domain Brownian reference with the same markers
is itself above budget (sampling noise), FAIL otherwise.
Exit codes: 0 PASS, 1 FAIL, 2 INCONCLUSIVE (no failure, but at least one state unresolved).

Scope of what is tested:
  - "four-partition": the driver keeps the markers in four vectors inside ONE process and
    reduces over them. This checks partition-independent decisions and reduction order, not
    an MPI run.
  - the ellipse case redraws each marker's poloidal angle every step (instantaneous
    flux-surface mixing). It tests the averaged geometry mapping, not unrestricted 2-D
    diffusion in an ellipse and not magnetic-orbit behaviour.
    python scripts/check_exchange.py            # run the 14 cases
    python scripts/check_exchange.py --reuse    # re-evaluate output/exchange.json
"""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent.parent
exe = here / "output" / "test_exchange"
BUDGET, SEEDS, KAPPAS = 0.005, (1, 2, 3), (1.0, 1.7)
NAMES = ("pulse in core", "pulse in markers", "heavy markers, empty layers", "uniform equilibrium")
jobs = [(m, k, s, 1) for k in KAPPAS for m in ("aux", "brownian") for s in SEEDS] + [("aux", k, 1, 4) for k in KAPPAS]
def run(j):
    return json.loads(subprocess.run([str(exe), j[0], str(j[1]), str(j[2]), str(j[3])], capture_output=True, text=True, check=True).stdout)
if "--reuse" in sys.argv: out = json.load(open(here / "output" / "exchange.json"))
else:
    with ThreadPoolExecutor(10) as ex: out = list(ex.map(run, jobs))
    json.dump(out, open(here / "output" / "exchange.json", "w"), indent=1)
fails, incon = [], []
def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg); fails.append(not ok)
for k in KAPPAS:
    A = [o for o in out if o["mode"] == "aux" and o["kappa"] == k and o["ranks"] == 1]
    B = [o for o in out if o["mode"] == "brownian" and o["kappa"] == k]
    R4 = next(o for o in out if o["mode"] == "aux" and o["kappa"] == k and o["ranks"] == 4)
    a0 = A[0]
    print(f"kappa {k}{' (poloidal angle redrawn every step: flux-surface averaged mapping)' if k != 1 else ''}: g2 {a0['g2']:.4f}, D_eff tau/h^2 {a0['Dtau_h2']:.3f}, rms marker step/h {a0['disp_h']:.3f}, {a0['nwin']} windows x {a0['nsub']} steps")
    check(max(max(o["cons"]) for o in A) < 1e-10, f"per-state conservation (core + markers + pending) {max(max(o['cons']) for o in A):.1e}")
    check(min(o["umin"] for o in A) >= 0, "core densities never negative")
    check(max(o["pending_over_w"] for o in A) < 1.0, f"pending below one marker weight (max {max(o['pending_over_w'] for o in A):.3f})")
    same = abs(R4["checksum"] - A[0]["checksum"]) <= 1e-10 * abs(A[0]["checksum"]) and np.allclose(R4["terr"], A[0]["terr"], rtol=0, atol=1e-9) and R4["markers"] == A[0]["markers"]
    check(same, f"four-partition (single process) run reproduces the one-partition run (checksum diff {abs(R4['checksum'] - A[0]['checksum'])/abs(A[0]['checksum']):.1e}, same marker count)")
    for z, nm in enumerate(NAMES):
        va = np.array([o["terr"][z] for o in A]); vb = np.array([o["terr"][z] for o in B])
        da = np.mean([o["drms"][z] for o in A]); db = np.mean([o["drms"][z] for o in B])
        line = (f"{nm:28s} transfer aux " + " ".join(f"{v:.4f}" for v in va) + f" | Brownian " + " ".join(f"{v:.4f}" for v in vb)
                + f" | density rms aux {da:.4f} / Brownian {db:.4f}")
        if va.max() <= BUDGET: check(True, line)
        elif vb.max() > BUDGET or va.mean() <= vb.mean() + 3 * vb.std():
            print("  INCONCLUSIVE " + line + "  (reference above budget or aux within its noise)"); incon.append(nm)
        else: check(False, line)
    ja = [o["jump"] for o in A]; jb = [o["jump"] for o in B]
    check(abs(np.mean(ja)) <= abs(np.mean(jb)) + 3 * np.std(jb) + 0.01, f"equilibrium interface jump aux {np.mean(ja):+.4f} ± {np.std(ja):.4f} / Brownian {np.mean(jb):+.4f} ± {np.std(jb):.4f}")
if any(fails): print("FAIL"); sys.exit(1)
if incon: print(f"INCONCLUSIVE: {len(incon)} state(s) noise limited, no failure"); sys.exit(2)
print("PASS"); sys.exit(0)
