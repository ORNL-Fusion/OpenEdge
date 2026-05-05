#!/usr/bin/env python3
"""
Verify constant-flux source: total emitted = flux * area * dt * nsteps,
independent of nevery (after the dt*=nevery scaling fix).
"""
import re
import os
import sys
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLUX = 1.0e6
AREA = 1.0
DT   = 1.0e-5
NSTEPS = 1000
EXPECTED = FLUX * AREA * DT * NSTEPS

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = sorted(glob.glob(os.path.join(HERE, "output", "log.n*")),
              key=lambda p: int(re.search(r"log\.n(\d+)", p).group(1)))

def parse_log(path):
    """Return (steps[], nsingle[], ntotal[]) from stats lines."""
    steps, nsingle, ntotal = [], [], []
    in_run = False
    with open(path) as f:
        for line in f:
            if line.startswith("Step"):
                in_run = True
                continue
            if in_run:
                m = re.match(r"\s*(\d+)\s+\d+\s+(\d+)\s+(\d+)\s*$", line)
                if not m:
                    if line.strip().startswith("Loop"):
                        break
                    continue
                steps.append(int(m.group(1)))
                nsingle.append(int(m.group(2)))
                ntotal.append(int(m.group(3)))
    return steps, nsingle, ntotal

results = []
for log in LOGS:
    n = int(re.search(r"log\.n(\d+)", log).group(1))
    s, ns, nt = parse_log(log)
    if not nt:
        print(f"[skip] {log}: no stats parsed")
        continue
    final = nt[-1]
    rel_err = (final - EXPECTED) / EXPECTED
    results.append((n, final, rel_err, s, nt))
    print(f"nevery={n:>3d}  final ntotal={final:>6d}  expected={EXPECTED:.0f}  "
          f"rel_err={rel_err*100:+.2f}%")

if not results:
    sys.exit("No logs parsed.")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for n, _, _, s, nt in results:
    ax.plot(s, nt, label=fr"$n_{{\mathrm{{every}}}}={n}$", lw=2)
ax.axhline(EXPECTED, color="k", ls="--", lw=1.5,
           label=fr"expected $\Gamma A \Delta t N$ = {EXPECTED:.0f}")
ax.set_xlabel("step", fontsize=14)
ax.set_ylabel("cumulative particles emitted", fontsize=14)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)

ax = axes[1]
ns = [r[0] for r in results]
err = [r[2] * 100 for r in results]
ax.bar([str(n) for n in ns], err, color="steelblue", edgecolor="k")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel(r"$n_{\mathrm{every}}$", fontsize=14)
ax.set_ylabel(r"relative error in total (\%)", fontsize=14)
ax.tick_params(labelsize=12)

plt.tight_layout()
out = os.path.join(HERE, "constant_flux_validation.png")
plt.savefig(out, dpi=130)
print(f"\nwrote {out}")
