#!/bin/bash
# Run all four 2-way coupling cases sequentially. Each case is re-staged FRESH
# from the canonical config.json (clean campaign tree + clean convergence trace),
# then looped. Values (max_iters, solps_ntim, oe max_steps, droplet rates) all
# come from canonical config.json -> edit there before launching.
#
#   nohup bash run_ensemble.sh > ensemble.log 2>&1 &
#
# Per-case output also lands in ensemble_runs/<case>/loop_<case>/.
set -u
export OPENEDGE_ROOT="${OPENEDGE_ROOT:-/home/cloud/OpenEdge}"
export OPENEDGE_BUILD="${OPENEDGE_BUILD:-/home/cloud/buildOpenEdge}"
COUPLING="$OPENEDGE_ROOT/tools/coupling"
ROOT="$OPENEDGE_ROOT/examples/test_solps_coupling/coupled_test"
CASES="detached__mist detached__no_mist attached__no_mist attached__mist"

cd "$ROOT"
for c in $CASES; do
    echo "==================== $c  stage+run  $(date '+%F %T') ===================="
    python3 -u "$COUPLING/setup_coupling.py" config.json --case "$c" --fresh
    python3 -u "$COUPLING/loop_driver.py" "ensemble_runs/$c/config.json" --case "$c"
    echo "==================== $c  done rc=$?  $(date '+%F %T') ===================="
done
echo "ENSEMBLE COMPLETE $(date '+%F %T')"
