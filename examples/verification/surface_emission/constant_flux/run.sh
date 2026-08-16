#!/usr/bin/env bash
# Verify (1) constant-flux mode and (2) nevery flux scaling.
# Theoretical: total emitted = flux * area * dt * nsteps  (independent of nevery).
# Exits 0 on PASS, 1 on FAIL.

set -e
cd "$(dirname "$0")"
mkdir -p output

SPA=${SPA:-$HOME/build_oe/src/spa_mac_mpi}
NP=${NP:-4}

for n in 1 5 10 50; do
  echo "=== nevery = $n ==="
  mpirun -np "$NP" "$SPA" -in in.constant_flux -var nevery_in $n -log output/log.n$n > /dev/null
done

python3 - <<'EOF'
import re, os, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLUX, AREA, DT, NSTEPS = 1.0e6, 1.0, 1.0e-5, 1000
EXPECTED = FLUX * AREA * DT * NSTEPS
REL_TOL = 0.05
NEVERY_SET = {1, 5, 10, 50}

logs = sorted(glob.glob("output/log.n*"),
              key=lambda p: int(re.search(r"log\.n(\d+)", p).group(1)))

results = []
for log in logs:
    n = int(re.search(r"log\.n(\d+)", log).group(1))
    steps, ntotal, in_run = [], [], False
    for line in open(log):
        if line.startswith("Step"):
            in_run = True
            continue
        if in_run:
            m = re.match(r"\s*(\d+)\s+\d+\s+(\d+)\s+(\d+)\s*$", line)
            if m:
                steps.append(int(m.group(1))); ntotal.append(int(m.group(3)))
            elif line.strip().startswith("Loop"):
                break
    if not ntotal:
        print(f"[skip] {log}: no stats parsed")
        continue
    rel_err = (ntotal[-1] - EXPECTED) / EXPECTED
    results.append((n, ntotal[-1], rel_err, steps, ntotal))
    print(f"nevery={n:>3d}  final ntotal={ntotal[-1]:>6d}  "
          f"expected={EXPECTED:.0f}  rel_err={rel_err*100:+.2f}%")

if not results:
    sys.exit("No logs parsed.")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for n, _, _, s, nt in results:
    axes[0].plot(s, nt, label=fr"$n_{{\mathrm{{every}}}}={n}$", lw=2)
axes[0].axhline(EXPECTED, color="k", ls="--", lw=1.5,
                label=fr"expected $\Gamma A \Delta t N$ = {EXPECTED:.0f}")
axes[0].set_xlabel("step"); axes[0].set_ylabel("cumulative particles emitted")
axes[0].legend()
axes[1].bar([str(r[0]) for r in results], [r[2]*100 for r in results],
            color="steelblue", edgecolor="k")
axes[1].axhline(0, color="k", lw=1)
axes[1].set_xlabel(r"$n_{\mathrm{every}}$")
axes[1].set_ylabel("relative error in total (%)")
plt.tight_layout()
plt.savefig("output/constant_flux_verification.png", dpi=130)
print("wrote output/constant_flux_verification.png")

failed = {r[0] for r in results} != NEVERY_SET
if failed:
    print(f"  nevery sweep incomplete: {sorted(r[0] for r in results)} FAIL")
for n, _, rel_err, _, _ in results:
    ok = abs(rel_err) < REL_TOL
    failed |= not ok
    print(f"  nevery={n}: rel_err {rel_err*100:+.2f}% (tol {REL_TOL*100:.0f}%) "
          f"{'ok' if ok else 'FAIL'}")
print("PASS" if not failed else "FAIL")
sys.exit(1 if failed else 0)
EOF
