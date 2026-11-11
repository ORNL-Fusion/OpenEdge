#!/usr/bin/env bash
# fix particle/tag + surface/pwi ehist transit split.  Exits 0 on PASS, 1 on FAIL.
set -e
cd "$(dirname "$0")"
mkdir -p output
SPA=${SPA:-$HOME/build_oe/src/spa_mac_mpi}
NP=${NP:-4}
LAUNCH=${LAUNCH:-mpirun -np $NP}
SPA_ARGS=${SPA_ARGS:-}
PYTHON=${PYTHON:-python3}
$LAUNCH "$SPA" $SPA_ARGS -in in.transit -log output/log.transit > output/screen.transit 2>&1 \
  || { echo "run failed:"; tail -5 output/screen.transit; exit 1; }
$PYTHON scripts/check_transit.py output/log.transit
