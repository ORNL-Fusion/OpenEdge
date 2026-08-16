#!/bin/sh
# MPI rank-invariance gate: the same single-particle deck on 1 and 4 ranks
# must agree to dump precision. Runs 150k steps — inside the window where
# ulp-level migration-retrace roundoff has not yet been amplified by the
# marginally trapped orbit (divergence onset ~150k steps is expected and
# benign; field/decomposition bugs show up by step ~300, e.g. the missing
# btf/rtf equilibrium broadcast gave 3e-2 m there).
# Usage: ./check_mpi.sh [/path/to/spa binary]
set -e
BIN=${1:-$HOME/build_oe/src/spa_mac_mpi}
PY=${PYTHON:-python3}
cd "$(dirname "$0")"
# own dump names: never clobbers the canonical output/traj.boris
mpirun -np 1 "$BIN" -var numStep 150000 \
  -var dumpFile output/traj.boris.np1 -in in.boris > /dev/null 2>&1
mpirun -np 4 "$BIN" -var numStep 150000 \
  -var dumpFile output/traj.boris.np4 -in in.boris > /dev/null 2>&1
$PY - <<'EOF'
import numpy as np, sys
def read(path):
    rows = []
    with open(path) as f:
        for line in f:
            if "ITEM: TIMESTEP" in line:
                ts = int(next(f)); next(f); n = int(next(f)); next(f)
                for _ in range(3): next(f)
                next(f)
                a = next(f).split()
                rows.append([ts] + [float(q) for q in a[2:8]])
                for _ in range(n - 1): next(f)
    return np.array(rows)
a = read("output/traj.boris.np1")
b = read("output/traj.boris.np4")
n = min(len(a), len(b))
sep  = np.sqrt(((a[:n,1:4] - b[:n,1:4])**2).sum(1)).max()
sepv = np.sqrt(((a[:n,4:7] - b[:n,4:7])**2).sum(1)).max()
# tolerances = dump precision (6 significant digits): ~1e-6 at R~1,
# ~1 m/s at v~2e5; single last-digit flips are benign, real bugs give >>.
ok = len(a) == len(b) and sep < 2e-6 and sepv < 2.0
print(f"  np1 frames {len(a)}  np4 frames {len(b)}  "
      f"max sep {sep:.3e} m, {sepv:.3e} m/s")
print(("PASS" if ok else "FAIL") +
      ": MPI rank invariance (150k steps, tol 2e-6 m / 2 m/s)")
sys.exit(0 if ok else 1)
EOF
