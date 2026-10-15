#!/bin/bash
# ST40 <-> SOLPS coupling: stage the SOLPS case once, then run the loop.
#   ./run_coupling.sh            stage if needed, run loop.max_iters iterations
#   ./run_coupling.sh --fresh    re-stage coupling/solps/st40 from ../solps/run_dir
#   ./run_coupling.sh --dry      print the plan, launch nothing
# Needs: OPENEDGE_ROOT (repo), OPENEDGE_BUILD (cmake build with src/spa_mpi),
#        OPENEDGE_SOLPS_DIR (SOLPS-ITER tree, used by run_solps.csh).
# The staged SOLPS case must already contain Li species (Li, Li+, Li2+, Li3+)
# and a converged b2fstate with zero Li source; the loop restarts from it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CASE_DIR="$(dirname "$HERE")"
export OPENEDGE_ROOT="${OPENEDGE_ROOT:-$(cd "$CASE_DIR/../../../.." && pwd)}"
export OPENEDGE_BUILD="${OPENEDGE_BUILD:-$HOME/build_oe}"
export OPENEDGE_SOLPS_DIR="${OPENEDGE_SOLPS_DIR:-/path/to/solps-iter}"
COUPLING="$OPENEDGE_ROOT/tools/coupling"
SRC_SOLPS="${SOLPS_SRC:-$CASE_DIR/solps/run_dir}"      # converged SOLPS case to start from
STAGED="$HERE/solps/st40"

FRESH=0; DRY=(); export COUPLING_ARGS="$*"
for a in "$@"; do case "$a" in --fresh) FRESH=1 ;; --dry) DRY=(--dry) ;; *) echo "unknown arg $a" >&2; exit 2 ;; esac; done

if [[ ! -f "$STAGED/b2fstate" || $FRESH == 1 ]]; then
    echo "## staging SOLPS case: $SRC_SOLPS -> $STAGED"
    rm -rf "$STAGED"; mkdir -p "$HERE/solps"
    rsync -a --exclude 'b2mn.exe.dir' --exclude 'output' --exclude 'tracing' --exclude '*.npz' \
          --exclude '*.h5' --exclude 'b2fmovie' --exclude 'b2ftrace' --exclude 'b2ftrack' \
          --exclude 'run.log' "$SRC_SOLPS/" "$STAGED/"
    ln -sfn "$(cd "$SRC_SOLPS/.." && pwd)/baserun" "$HERE/solps/baserun"
    # keep one active b2mndr_ntim line; the driver sets its value from config solps_ntim
    sed -i.openedge.bak -e "/^#'b2mndr_ntim'/d" "$STAGED/b2mn.dat"
    awk "/'b2mndr_ntim'/{n++; if(n>1) next} {print}" "$STAGED/b2mn.dat" > "$STAGED/b2mn.dat.tmp" && mv "$STAGED/b2mn.dat.tmp" "$STAGED/b2mn.dat"
    python3 - "$STAGED" <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.environ["OPENEDGE_ROOT"], "tools/coupling"))
from solps_interface import SolpsInterface
si = SolpsInterface(sys.argv[1], baserun_dir=os.path.join(os.path.dirname(sys.argv[1]), "baserun"))
si.load_geometry(); si.load_plasma_state()
zn = [int(z) for z in si.state["zn"].flatten()]
print(f"   staged: nx={si.nx} ny={si.ny} ns={si.ns} nuclear charges={zn}")
if 3 not in zn:
    msg = "no lithium species in the staged b2fstate; SOLPS must be set up with Li before coupling"
    if "--dry" in os.environ.get("COUPLING_ARGS", ""):
        print("   WARNING (dry run): " + msg)
    else:
        sys.exit("ERROR: " + msg)
PY
fi

echo "## loop: $(python3 -c "import json;print(json.load(open('$HERE/config.json'))['loop'])")"
python3 -u "$COUPLING/loop_driver.py" "$HERE/config.json" --case st40 "${DRY[@]}"
