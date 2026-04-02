#!/bin/bash
# Run OpenEdge-SOLPS coupling from clean state
# Usage: ./run_coupling.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DRIVER=/home/cloud/OpenEdge/tools/coupling/openedge_solps_driver.py

echo "Cleaning previous run outputs..."
rm -f openedge/in.chunk_* openedge/error_*.log
rm -f openedge/output/particles.txt openedge/restart.dat
rm -rf step_*

echo "Launching coupling driver..."
python3 "$DRIVER" config.json
