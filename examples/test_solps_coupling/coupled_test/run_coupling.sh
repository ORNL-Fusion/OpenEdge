#!/bin/bash
# Stage a self-contained campaign, then run the iterative OpenEdge<->SOLPS
# coupling loop to convergence. The validated standalone inputs are never
# mutated -- the loop runs entirely under ensemble_runs/<case>/.
#
# Usage:
#   ./run_coupling.sh [case]            stage if needed, then run the loop
#   ./run_coupling.sh [case] --fresh    rebuild the campaign from clean inputs
#   ./run_coupling.sh [case] --dry      print the plan, launch nothing

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export OPENEDGE_ROOT="${OPENEDGE_ROOT:-/home/cloud/OpenEdge}"
export OPENEDGE_BUILD="${OPENEDGE_BUILD:-/home/cloud/buildOpenEdge}"
COUPLING="$OPENEDGE_ROOT/tools/coupling"
SETUP="$COUPLING/setup_coupling.py"
LOOP="$COUPLING/loop_driver.py"

CASE="$(python3 -c "import json;print(json.load(open('config.json')).get('loop',{}).get('case','attached__mist'))")"
FRESH=()
DRY=()
for a in "$@"; do
    case "$a" in
        --fresh) FRESH=(--fresh) ;;
        --dry)   DRY=(--dry) ;;
        *)       CASE="$a" ;;
    esac
done

CAMPAIGN="$SCRIPT_DIR/ensemble_runs/$CASE"
CAMP_CFG="$CAMPAIGN/config.json"

echo "## case=$CASE  campaign=$CAMPAIGN"

# Stage the campaign when missing or when --fresh is requested.
if [[ ! -f "$CAMP_CFG" || ${#FRESH[@]} -gt 0 ]]; then
    echo "## staging campaign..."
    python3 -u "$SETUP" config.json --case "$CASE" "${FRESH[@]}" "${DRY[@]}"
fi

if [[ ${#DRY[@]} -gt 0 ]]; then
    echo "## --dry: loop plan against staged config"
    [[ -f "$CAMP_CFG" ]] && python3 -u "$LOOP" "$CAMP_CFG" --dry || \
        echo "  (campaign not staged in --dry mode; run without --dry to stage)"
    exit 0
fi

echo "## launching loop..."
python3 -u "$LOOP" "$CAMP_CFG"
