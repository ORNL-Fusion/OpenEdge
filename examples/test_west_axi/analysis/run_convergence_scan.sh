#!/usr/bin/env bash
# Convergence/sensitivity scan for test_west_axi.
#
# Runs three variants of in.west with different (nlaunchTotalW, Nphase)
# overrides. Each run re-uses the same PMI compute, plasma, and grid --
# only the macroparticle-emission count and run length change.
#
# Produces:
#   output/grid.dens.{A,B,C}.west   -- per-cell n(W0..W20+) every Ndump steps
#   output/state.{A,B,C}.west.final -- particle dump at end, with pweight
#   log.{A,B,C}.openedge            -- full stats log for time-series parsing

set -e
cd "$(dirname "$0")/.."

: "${SPARTA:=$HOME/buildOpenEdge/src/spa_mpi}"
: "${NPROC:=64}"

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1

run() {
    local tag=$1 nlaunch=$2 nphase=$3 ndump=$4
    echo "=== run $tag : nlaunchTotalW=$nlaunch Nphase=$nphase Ndump=$ndump ==="
    mpirun -np "$NPROC" "$SPARTA" \
        -var nlaunchTotalW "$nlaunch" \
        -var Nphase "$nphase" \
        -var Ndump "$ndump" \
        -var tag "$tag" \
        -in in.west \
        2>&1 | tee "log.${tag}.openedge" > /dev/null
    echo "    log -> log.${tag}.openedge   grid -> output/grid.dens.${tag}.west"
}

# A = baseline (nlaunch=10), longest time to check plateau
run A  10  100000 10000

# B = same time, 10x sampling -- MC-convergence check
run B  100 100000 10000

# C = same time, 30x sampling
run C  300 100000 10000

echo "=== all three runs done ==="
ls -la output/grid.dens.*.west log.*.openedge
