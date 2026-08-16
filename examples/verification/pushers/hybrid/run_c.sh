#!/bin/sh
# Stage C: operator coupling to the GC state. Each operator runs twice —
# Boris reference (hybrid, gswitch 1e30) and pure GCA — and the gate is
# agreement of the operator's OWN observable:
#   force  : parallel drift rate       (fix force/thermal, grad_te_z)
#   cd     : perpendicular MSD growth  (fix cross_field_diffusion)
#   coul   : var(v_par) growth         (fix coulomb/background via ni)
# Before the API hooks, GCA runs showed ZERO operator response (kicks
# discarded at reconstruction) — the ratio gates fail loudly on that.
BIN=${1:-$HOME/build_oe/src/spa_mac_mpi}
PY=${PYTHON:-python3}
cd "$(dirname "$0")"
mkdir -p output
rm -f output/traj.c_*
fail=0

[ -f input/source.b_in ] || $PY make_inputs.py

run() {
  echo "== $*"
  mpirun -np 1 "$BIN" "$@" -in in.coupling > /dev/null 2>&1 || { echo "FAIL: solver $*"; fail=1; }
}

run -var pmode hybrid -var gswitch 1e30 -var gradte 50.0  -var tag c_force_ref
run -var pmode gca    -var gswitch 2.5  -var gradte 50.0  -var tag c_force_gca
run -var pmode hybrid -var gswitch 1e30 -var dperp 0.5    -var tag c_cd_ref
run -var pmode gca    -var gswitch 2.5  -var dperp 0.5    -var tag c_cd_gca
run -var pmode hybrid -var gswitch 1e30 -var nicoul 1e20  -var tag c_coul_ref
run -var pmode gca    -var gswitch 2.5  -var nicoul 1e20  -var tag c_coul_gca

$PY plot_coupling.py || fail=1
exit $fail
