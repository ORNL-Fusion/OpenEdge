#!/usr/bin/env bash
# surface/pwi ledger fold before dumps, run end and restarts: adens_nevery 100
# must reproduce the every-step ledger at every dump step (70, 140, ... 350),
# at 1 and NP ranks.  Exits 0 on PASS, 1 on FAIL.
set -e
cd "$(dirname "$0")"
mkdir -p output
SPA=${SPA:-$HOME/build_oe/src/spa_mac_mpi}
NP=${NP:-4}
LAUNCH=${LAUNCH:-mpirun -np}
SPA_ARGS=${SPA_ARGS:-}
PYTHON=${PYTHON:-python3}
run() {  # name ranks [extra -var args]
  local name=$1 np=$2; shift 2
  echo "=== $name ($np ranks)"
  $LAUNCH $np "$SPA" $SPA_ARGS -in in.fold -var case "$name" "$@" -log "output/log.$name" > "output/screen.$name" 2>&1 \
    || { echo "run $name failed:"; tail -5 "output/screen.$name"; exit 1; }
}
for np in 1 $NP; do
  run n1_np$np   $np -var nevery 1
  run n100_np$np $np -var nevery 100
  run n100_np$np $np -var nevery 100 -var restart 1
done
$PYTHON scripts/check_fold.py $NP
