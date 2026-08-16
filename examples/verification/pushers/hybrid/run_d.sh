#!/bin/sh
# Stage D1: A1 in production geometry — axi ring target, 45-deg B.
#   ref/ref2 : axi kick-drift Boris at gyro-resolving dt (+dt/2 band)
#   a0       : pure GCA, chord impact (must FAIL the distribution gates —
#              the axi mover collides GCA particles at chord velocity)
#   a1       : pure GCA + gc_wall flux -> collision-site materialization
BIN=${1:-$HOME/build_oe/src/spa_mac_mpi}
PY=${PYTHON:-python3}
cd "$(dirname "$0")"
mkdir -p output
rm -f output/traj.d_* output/impacts.d_* output/switch.d_*
fail=0

[ -f input/source.axi256 ] || $PY make_inputs.py

run() {
  echo "== $*"
  mpirun -np 1 "$BIN" "$@" -in in.axiring > /dev/null 2>&1 || { echo "FAIL: solver $*"; fail=1; }
}

run -var pmode hybrid -var gswitch 1e30 -var dt 5e-10   -var nsteps 12000 -var dfreq 20 -var tag d_ref
run -var pmode hybrid -var gswitch 1e30 -var dt 2.5e-10 -var nsteps 24000 -var dfreq 40 -var tag d_ref2
run -var pmode gca -var gswitch 2.5 -var gcwall a0   -var dt 1e-8 -var nsteps 800 -var dfreq 1 -var tag d_a0
run -var pmode gca -var gswitch 2.5 -var gcwall flux -var dt 1e-8 -var nsteps 800 -var dfreq 1 -var tag d_a1

$PY plot_axi.py || fail=1
exit $fail
