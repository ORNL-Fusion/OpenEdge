#!/usr/bin/env bash
# strata erosion reaches a species buried under a thin layer of another
# material inside the reaction zone, and the column ledger stays equal to the
# stack (fine folds, and coarse folds whose debits exceed what is left).
# Exits 0 on PASS, 1 on FAIL.
set -e
cd "$(dirname "$0")"
rm -rf output && mkdir -p output
SPA=${SPA:-$HOME/build_oe/src/spa_mac_mpi}
NP=${NP:-4}
LAUNCH=${LAUNCH:-mpirun -np $NP}
PYTHON=${PYTHON:-python3}   # needs numpy
for c in fine:10 coarse:500; do
  $LAUNCH "$SPA" -in in.buried -var case ${c%%:*} -var nevery ${c##*:} \
    -log output/log.${c%%:*} > output/screen.${c%%:*} 2>&1 \
    || { echo "run ${c%%:*} failed:"; tail -5 output/screen.${c%%:*}; exit 1; }
done
$PYTHON check.py
