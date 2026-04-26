#!/usr/bin/env bash
# Launch all 4 test_neutral_slab cases in parallel and plot each result.
# Each case is mpirun -np 4 → 16 ranks total. Logs in log.<case>, dumps
# in output/slab_<case>.grid, plots <case>.png in this directory.

set -e
cd "$(dirname "$0")"
mkdir -p output

SPA=${SPA:-$HOME/buildOpenEdge/src/spa_mpi}
NPROC=${NPROC:-4}

for case in iz recycle diss cx; do
  ( mpirun -np "$NPROC" "$SPA" -in "in.slab_$case" > "log.$case" 2>&1 \
      && python3 plot_slab.py "$case" ) &
done

wait
python3 plot_slab.py composite
python3 plot_slab.py thermalization
echo "all 4 cases done"
ls -l output/slab_*.grid *.png 2>/dev/null
