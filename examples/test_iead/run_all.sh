#!/bin/bash
set -e

EXE=/home/cloud/buildOpenEdge/src/spa_mpi
NP=4

for alpha in 0 45 85; do
    echo "=== Running alpha = ${alpha} deg ==="
    mpirun -np $NP $EXE -in in.iead_alpha${alpha}
done

echo "=== All done ==="
