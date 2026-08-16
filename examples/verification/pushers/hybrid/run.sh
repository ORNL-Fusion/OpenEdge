#!/bin/sh
# Stage A of PLAN.md: Boris reference (+dt/2 band), hybrid at 1x/20x/400x
# dt, pure GCA at 20x dt; gates in plot_impact.py. -np 1 (single-orbit
# reproducibility; MPI invariance is covered by ../orbit).
# Usage: ./run.sh [/path/to/spa binary]     (PYTHON=... to pick interpreter)
BIN=${1:-$HOME/build_oe/src/spa_mac_mpi}
PY=${PYTHON:-python3}
cd "$(dirname "$0")"
mkdir -p output
rm -f output/impacts.* output/traj.* output/switch.*
fail=0

[ -f input/source.n256 ] && [ -f input/source.hyst ] || $PY make_inputs.py

run() {
  echo "== $*"
  mpirun -np 1 "$BIN" "$@" -in in.wall > /dev/null 2>&1 || { echo "FAIL: solver $*"; fail=1; }
}

# Boris reference at gyro-resolving dt, and dt/2 for the tolerance band
run -var pmode hybrid -var gswitch 1e30 -var knear 0 -var dt 5e-10   -var nsteps 12000 -var dfreq 20 -var tag ref
run -var pmode hybrid -var gswitch 1e30 -var knear 0 -var dt 2.5e-10 -var nsteps 24000 -var dfreq 40 -var tag ref2
run -var pmode hybrid -var gswitch 1e30 -var knear 0 -var dt 1.25e-10 -var nsteps 48000 -var dfreq 80 -var tag ref4
# hybrid at the reference dt (shell = 2.5 rho_L)
run -var pmode hybrid -var gswitch 2.5  -var knear 2.5 -var dt 5e-10 -var nsteps 12000 -var dfreq 20 -var tag hyb_small
# hybrid at 20x dt — the payoff case; per-step dump for transition gates
run -var pmode hybrid -var gswitch 2.5  -var knear 2.5 -var dt 1e-8  -var nsteps 800 -var dfreq 1 -var subc 20 -var tag hyb
# shell-skipping case: one GC step spans > the shell; trial/replay must switch
run -var pmode hybrid -var gswitch 2.5  -var knear 2.5 -var dt 2e-7  -var nsteps 4000 -var dfreq 1 -var subc 400 -var tag skip
# complete leap-through: ONE GC step crosses the whole shell AND the wall
# plane (endpoint-only distance tests cannot see this; signed-crossing
# detection must switch every particle to Boris before impact)
run -var pmode hybrid -var gswitch 2.5  -var knear 2.5 -var dt 2.5e-6 -var nsteps 800 -var dfreq 1 -var subc 5000 -var tag leap
# k*rho_L calibration sweep (PLAN: k = 1 / 2.5 / 5)
run -var pmode hybrid -var gswitch 2.5  -var knear 1   -var dt 1e-8  -var nsteps 800 -var dfreq 4 -var tag k1
run -var pmode hybrid -var gswitch 2.5  -var knear 5   -var dt 1e-8  -var nsteps 800 -var dfreq 4 -var tag k5
# pure GCA to the wall (A0: uniform reconstruction phase — rejected ref)
run -var pmode gca    -var gswitch 2.5  -var knear 0   -var dt 1e-8  -var nsteps 800 -var dfreq 1 -var tag gca
# A1 experiment: flux-weighted first-passage phase at GC wall arrival
run -var pmode gca    -var gswitch 2.5  -var knear 0   -var dt 1e-8  -var nsteps 800 -var dfreq 1 -var gcwall flux -var tag a1
# hysteresis excursion: launched inside the shell moving AWAY; must stay
# Boris until chi >= 4*pi AND d > 2*d_sw, then re-enter GCA exactly once
run -var pmode hybrid -var gswitch 2.5  -var knear 2.5 -var dt 1e-8  -var nsteps 300 -var dfreq 1 -var src source.hyst -var tag hyst

$PY plot_impact.py || fail=1
exit $fail
