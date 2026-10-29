#!/usr/bin/env python3
"""PASS/FAIL: a buried B layer erodes through a thin Wd deposit (strata).

Checked at every dump of both cases (fine and coarse ledger folds): the B
column never goes negative and equals the B inventory of the strata stack.
"""
import glob, re, sys
import numpy as np

key = lambda p: int(re.findall(r"\.(\d+)\.dump$", p)[0])
NB0, NWD = 1e15, 2e14
results = []
def check(name, ok, detail):
    results.append(ok); print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}")

def load(f):
    t = open(f).read().split("ITEM: SURFS")[1].split("\n")
    a = np.array([list(map(float, l.split())) for l in t[1:] if l.strip()])
    return a[np.argsort(a[:, 0])]

def stack_inventory(row, m):
    """areal content of material m in a packed strata stack"""
    nsp, nl = int(row[0]), int(row[3])
    tot, k = 0.0, 4
    for _ in range(nl):
        thick, dens = row[k], row[k + 1]
        tot += thick * dens * row[k + 2 + m]
        k += 2 + nsp
    return tot

# columns: id area sB sWd cW cB cWd F strata...
MAT_B, MAT_WD = 1, 4          # material order: W, B, D, O, Wd
for case in ("fine", "coarse"):
    files = sorted(glob.glob(f"output/{case}.surf.*.dump"), key=key)
    dumps = [load(f) for f in files]
    s0, s1 = dumps[0], dumps[-1]
    hot = s0[:, 7] > 1e20
    check(f"{case}: hot coated faces present", hot.sum() >= 5, f"{hot.sum()} faces with F > 1e20")
    colmin = min(d[:, 2].min() for d in dumps)
    check(f"{case}: B column never negative (all dumps)", colmin > -1e-6 * NB0,
          f"min s_adens[B] = {colmin:.3e}")
    gap = max(abs(d[i, 2] - stack_inventory(d[i, 8:], MAT_B)) for d in dumps for i in range(len(d)))
    check(f"{case}: B column equals stack inventory (all dumps, all faces)", gap < 1e-5 * NB0,      # dumps carry 6 significant digits
          f"max |column - stack| = {gap:.3e}")
    gapd = max(abs(d[i, 3] - stack_inventory(d[i, 8:], MAT_WD)) for d in dumps for i in range(len(d)))
    check(f"{case}: Wd column equals stack inventory", gapd < 1e-5 * NWD, f"max = {gapd:.3e}")
    check(f"{case}: buried B depletes", s1[hot, 5].max() < 0.05,
          f"conc_B {s0[hot, 5].mean():.3f} -> max {s1[hot, 5].max():.3e} (pre-fix: frozen)")
    check(f"{case}: W substrate exposed", s1[hot, 4].min() > s0[hot, 4].max(),
          f"conc_W {s0[hot, 4].mean():.3f} -> min {s1[hot, 4].min():.3f}")
    check(f"{case}: un-eroded Wd unchanged", np.allclose(s1[hot, 3], NWD, rtol=1e-9),
          f"s_adens[Wd] {s1[hot, 3].min():.4e}..{s1[hot, 3].max():.4e}")
ok = all(results)
print("PASS | strata buried-layer erosion" if ok else "FAIL | strata buried-layer erosion")
sys.exit(0 if ok else 1)
