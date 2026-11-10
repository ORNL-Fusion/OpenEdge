#!/usr/bin/env python3
"""Compare the adens_nevery 100 ledger dumps with the every-step fold."""
import glob, sys
import numpy as np

RTOL = 1e-9
NP = sys.argv[1] if len(sys.argv) > 1 else "4"

def dump(path):
    lines = open(path).read().splitlines()
    h = next(i for i, l in enumerate(lines) if l.startswith("ITEM: SURFS"))
    cols = lines[h].split()[2:]
    n = int(lines[lines.index("ITEM: NUMBER OF SURFS") + 1])
    data = np.loadtxt(lines[h + 1:h + 1 + n], ndmin=2)
    return cols, data[np.argsort(data[:, cols.index("id")])]

failed = False
def check(ok, msg):
    global failed
    print(("  ok   " if ok else "  FAIL ") + msg); failed |= not ok

for np_ in ("1", NP):
    steps = sorted(int(p.split(".")[-2]) for p in glob.glob(f"output/n1_np{np_}.*.dump"))
    check(steps == [0, 70, 140, 210, 280, 350], f"{np_} ranks: dumps at {steps}")
    for s in steps:
        c1, d1 = dump(f"output/n1_np{np_}.{s}.dump")
        c2, d2 = dump(f"output/n100_np{np_}.{s}.dump")
        for name in ("s_adens_net", "s_adens_dep", "s_adens_ero", "s_adens[1]"):
            a, b = d1[:, c1.index(name)], d2[:, c2.index(name)]
            check(np.allclose(a, b, rtol=RTOL, atol=0), f"{np_} ranks, step {s:3d}: {name:12s} nevery 100 == nevery 1  ({b[0]:.6e})")
    c1, d1 = dump(f"output/n1_np{np_}.350.dump")
    cr, dr = dump(f"output/n100_np{np_}_restart.350.dump")
    for name in ("s_adens_net", "s_adens_dep", "s_adens_ero"):
        check(np.allclose(d1[:, c1.index(name)], dr[:, cr.index(name)], rtol=RTOL, atol=0),
              f"{np_} ranks: {name} read back from the restart == nevery 1")
    check(d1[:, c1.index("s_adens_dep")].max() > 0 and d1[:, c1.index("s_adens_ero")].max() > 0, f"{np_} ranks: ledger non-trivial")
print("PASS" if not failed else "FAIL"); sys.exit(1 if failed else 0)
