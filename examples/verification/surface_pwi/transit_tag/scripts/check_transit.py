#!/usr/bin/env python3
"""Last ehist block: transit column = population B only, at its energy; J sums to the marginals."""
import re, sys
import numpy as np
log = sys.argv[1] if len(sys.argv) > 1 else "output/log.transit"
created = [int(m) for m in re.findall(r"Created (\d+) particles", open(log, errors="ignore").read())]
nA, nB = created[-2], created[-1]
blocks = open("output/ehist.dat").read().split("# step")[1:]
rows = [l.split() for l in blocks[-1].splitlines()[1:] if l.strip()]
E = np.array([[float(x) for x in r[1:]] for r in rows if r[0] == "E"])
A = np.array([[float(x) for x in r[1:]] for r in rows if r[0] == "A"])
J = np.array([[float(x) for x in r[1:]] for r in rows if r[0] == "J"])
w = 1e14                                          # fnum = pweight of every marker
fails = []
def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg); fails.append(not ok)
nall, ntag = E[:, 1].sum() / w, E[:, 3].sum() / w
check(abs(nall - (nA + nB)) <= 10, f"all impacts {nall:.0f} = A + B markers ({nA} + {nB}) within 10 (stragglers)")
check(abs(ntag - nB) <= 10, f"transit impacts {ntag:.0f} = B markers ({nB}) within 10")
lo, hi = E[:, 0] < 70, E[:, 0] > 70
check(E[lo, 3].sum() == 0.0, "no transit weight in the A energy band (< 70 eV): no A marker tagged")
check(np.all(E[hi, 3] == E[hi, 1]), "every impact in the B energy band (> 70 eV) is tagged")
eA = E[np.argmax(E[:, 1] - E[:, 3]), 0]; eB = E[np.argmax(E[:, 3]), 0]
check(abs(eA - 34.5) < 3.0 and abs(eB - 115.7) < 3.0, f"untagged at {eA:.0f} eV (34), tagged at {eB:.0f} eV (116)")
check(abs(A[:, 1].sum() - E[:, 1].sum()) < 1e-6 * w and abs(A[:, 3].sum() - E[:, 3].sum()) < 1e-6 * w, "angle marginals match energy marginals")
check(A[0, 1] == A[:, 1].sum(), "all impacts in the first angle bin (normal incidence)")
check(abs(J[:, 2].sum() - E[:, 1].sum()) < 1e-6 * w and abs(J[:, 3].sum() - E[:, 3].sum()) < 1e-6 * w, "joint histogram sums to the marginals")
check(len(blocks) == 1, f"cumulative mode wrote one block at run end ({len(blocks)})")
print("PASS" if not any(fails) else "FAIL"); sys.exit(1 if any(fails) else 0)
