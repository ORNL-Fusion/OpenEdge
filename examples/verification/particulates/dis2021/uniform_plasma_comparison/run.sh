#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${1:-${OPENEDGE_BIN:-$HOME/build_oe/src/spa_mac_mpi}}
PYTHON=${PYTHON:-python3}
NP=${NP:-1}

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: OpenEdge executable is not runnable: $BIN" >&2
  exit 2
fi

cd "$HERE"
mkdir -p output
rm -f output/state.dustt2005 output/state.dis2021 \
      output/log.dustt2005 output/log.dis2021 output/comparison.csv

for model in dustt2005 dis2021; do
  echo "== $model"
  mpirun -np "$NP" "$BIN" \
    -var physics_model "$model" -var tag "$model" \
    -log "output/log.$model" -in in.compare
done

"$PYTHON" scripts/check_models.py
