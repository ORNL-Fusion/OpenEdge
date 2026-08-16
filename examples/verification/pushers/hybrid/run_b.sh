#!/bin/sh
# Stage B1: sheath boundary barrier at alpha = 0 (B || n).
#   b_in0  inbound, sheath OFF  — baseline impact energies
#   b_in   inbound, sheath ON   — gate the +e*phi inbound gain
#   b_sub  outbound E_par = 2 eV  < e*phi — must redeposit (0 escapes)
#   b_sup  outbound E_par = 40 eV > e*phi — must escape minus e*phi,
#          exactly once (transit-state accounting)
# Usage: ./run_b.sh [/path/to/spa binary]   (PYTHON=... for interpreter)
BIN=${1:-$HOME/build_oe/src/spa_mac_mpi}
PY=${PYTHON:-python3}
cd "$(dirname "$0")"
mkdir -p output
rm -f output/impacts.b_* output/escapes.b_* output/traj.b_* output/switch.b_*
fail=0

[ -f input/source.b_in ] || $PY make_inputs.py

run() {
  echo "== $*"
  mpirun -np 1 "$BIN" "$@" -in in.sheath > /dev/null 2>&1 || { echo "FAIL: solver $*"; fail=1; }
}

run -var shmode off      -var src source.b_in  -var tag b_in0 -var nsteps 1200
run -var shmode boundary -var src source.b_in  -var tag b_in  -var nsteps 1200
run -var shmode boundary -var src source.b_sub -var tag b_sub -var nsteps 1200
run -var shmode boundary -var src source.b_sup -var tag b_sup -var nsteps 1200

$PY plot_sheath.py || fail=1
exit $fail
