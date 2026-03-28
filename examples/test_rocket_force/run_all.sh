#!/bin/bash
# Run all rocket force test cases
# Usage: ./run_all.sh [path_to_spa_mpi]

SPA=${1:-~/buildOpenEdge/src/spa_mpi}

echo "=== Rocket Force Test ==="
echo "Using: $SPA"

# Generate input files and HDF5 data
python3 create_case.py

# Run three eta cases
for eta in 0p0 0p5 1p0; do
    echo ""
    echo "--- Running eta=$eta ---"
    mpirun -np 1 $SPA -in in.eta_${eta} 2>&1 | tail -5
done

echo ""
echo "--- Comparing results ---"
python3 compare_rocket.py
