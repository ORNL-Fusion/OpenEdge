#!/bin/sh
# Full orbit verification: Boris reference + GCA in 3D/2D-cart/2D-axi at rk2
# and rk4, gated by plot_trajectories.py, plus the MPI rank-invariance
# check. Single-particle decks run at -np 1 (see ../README.md).
# Usage: ./run.sh [/path/to/spa binary]     (PYTHON=... to pick interpreter)
BIN=${1:-$HOME/build_oe/src/spa_mac_mpi}
PY=${PYTHON:-python3}
cd "$(dirname "$0")"
mkdir -p output
fail=0

[ -f khan_plasma.h5 ] || $PY input/make_khan_plasma_h5.py

run() {  # run <deck> [-var ...] — abort on solver error
  echo "== mpirun -np 1 $BIN -in $*"
  mpirun -np 1 "$BIN" -in "$@" > /dev/null 2>&1 || { echo "FAIL: solver $*"; fail=1; }
}

run in.boris
for rk in rk2 rk4; do
  run in.gca     -var gcaIntegrator $rk
  run in.gca.2d  -var gcaIntegrator $rk
  run in.gca.axi -var gcaIntegrator $rk
done

log=output/run_gates.log
gate() {  # gate <dump> <tag> <mode>
  if $PY plot_trajectories.py --gca-dump "$1" --tag "$2" --mode "$3" \
       > "$log" 2>&1; then tail -1 "$log"
  else tail -1 "$log"; fail=1; fi
}

for rk in rk2 rk4; do
  gate traj.gca.$rk    $rk        3d
  gate traj.gca2d.$rk  2d.$rk     2dcart
  gate traj.gcaaxi.$rk axi.$rk    axi
done

echo "== MPI rank invariance"
if sh check_mpi.sh "$BIN" > "$log" 2>&1; then tail -1 "$log"
else tail -1 "$log"; fail=1; fi

if [ $fail -eq 0 ]; then echo "PASS: verification/pushers/orbit"; else echo "FAIL: verification/pushers/orbit"; fi
exit $fail
