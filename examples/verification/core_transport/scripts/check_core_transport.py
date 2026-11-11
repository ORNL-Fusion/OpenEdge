#!/usr/bin/env python3
"""C++ CoreTransport1D vs Aurora impden1 reference (bit-level) and particle conservation."""
import os, sys
import numpy as np

here = os.path.join(os.path.dirname(__file__), "..")
fails = []
def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg); fails.append(not ok)

def blocks(path):
    txt = open(path).read().split("# step")[1:]
    return [np.array([[float(x) for x in l.split()] for l in b.splitlines()[1:] if l.strip()]) for b in txt]

safe = "--safe" in sys.argv
prefix = "safe_" if safe else ""
for case in (() if safe else ("neutral", "evolve", "closed")):
    ref = blocks(os.path.join(here, "reference", f"ref_{case}.dat"))
    cpp = blocks(os.path.join(here, "output", f"dens_{case}.dat"))
    worst = max(np.max(np.abs(c - r)) / np.max(np.abs(r)) for c, r in zip(cpp, ref))
    check(worst < 1e-12, f"{case}: max |cpp - aurora| / max|n| = {worst:.1e} over 3 snapshots")

for case in ("evolve", "closed", "source", "source_fixed_n0"):   # neutral: fixed n0 with recombination to it = sink
    L = np.loadtxt(os.path.join(here, "output", f"{prefix}ledger_{case}.dat"))
    dt = float(open(os.path.join(here, "input", f"case_{case}.dat")).read().split("\n")[1].split()[0])
    budget = L[0, 1] + dt * np.sum(L[1:, 2] - L[1:, 3] - L[1:, 4])
    rel = abs(L[-1, 1] - budget) / L[-1, 1]
    check(rel < 1e-10, f"{case}: N(end) = N(0) + dt*sum(source - outflux - parloss) to {rel:.1e}"
          + (" (closed: outflux %.1e)" % np.abs(L[1:, 3]).max() if case == "closed" else ""))
    if case == "closed":
        check(np.all(L[1:, 3] == 0.0), "closed boundary: zero outflux")
    if case == "source":
        check(L[-1, 3] > 0.3 * L[-1, 2], f"source case: outflux {L[-1,3]:.3e} is a major share of source {L[-1,2]:.3e} at the end")

if safe:
    for case in ("neutral", "evolve", "closed", "source", "source_fixed_n0"):
        ns = blocks(os.path.join(here, "output", f"safe_dens_{case}.dat"))
        check(all(np.all(np.isfinite(n) & (n >= 0)) for n in ns), f"{case}: subcycled densities nonnegative")
print("PASS" if not any(fails) else "FAIL"); sys.exit(1 if any(fails) else 0)
