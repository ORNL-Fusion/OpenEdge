#!/usr/bin/env bash
# Verify (1) constant-flux mode and (2) nevery flux scaling.
# Theoretical: total emitted = flux * area * dt * nsteps  (independent of nevery).

set -e
cd "$(dirname "$0")"
mkdir -p output

SPA=${SPA:-/home/cloud/buildOpenEdge/src/spa_mpi}

for n in 1 5 10 50; do
  echo "=== nevery = $n ==="
  $SPA -in in.constant_flux -var nevery_in $n -log output/log.n$n
done

python3 analyze.py
