#!/bin/sh
# One-to-one grain-chain benchmark vs DUSTT (Pigarov PoP 12, 122508):
#   drag   : OML charging + ion friction chain vs python integration
#   efield : F_E = Z_d e E vs analytic
#   free   : phi_unwrap + centrifugal R(t) vs exact kinematics
BIN=${1:-$HOME/build_oe/src/spa_mac_mpi}
PY=${PYTHON:-python3}
cd "$(dirname "$0")"
mkdir -p output
rm -f output/traj.*
fail=0

[ -f input/grain.rest ] || $PY make_input.py

run() {
  echo "== $*"
  mpirun -np 1 "$BIN" "$@" -in in.grain > /dev/null 2>&1 || { echo "FAIL: solver $*"; fail=1; }
}

run -var tag drag   -var src grain.rest -var upar 2.0e4 -var dofix 1
run -var tag efield -var src grain.rest -var ez -100.0  -var dofix 1
run -var tag free   -var src grain.vphi                 -var dofix 0
run -var tag neut   -var src grain.v100 -var nn 1.0e19 -var tn 2.0 -var nel 1.0e6 -var dofix 1
run -var tag see    -var src grain.rest -var tev 30.0 -var dosee yes -var dofix 1
# vacuum: fixed charge, zero plasma/neutrals, E only, gravity off ->
# v(t) = Zd e E t / m exactly (gates the no-drag fallback branch)
run -var tag vac -var src grain.rest -var nel 0.0 -var ez -100.0 -var zfix -2.0e4 -var dofix 1
# MPI invariance: same drag case on 2 ranks must reproduce 1-rank output
echo "== np2 invariance"
mpirun -np 2 "$BIN" -var tag drag2 -var src grain.rest -var upar 2.0e4 -var dofix 1 -in in.grain > /dev/null 2>&1 || { echo "FAIL: solver np2"; fail=1; }

$PY check_dustt.py || fail=1
exit $fail
