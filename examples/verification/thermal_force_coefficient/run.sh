#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BIN=${1:-/Users/42d/build_oe/src/spa_mac_mpi}
PYTHON=${PYTHON:-python3}
cd "$HERE"
mkdir -p output
rm -f output/particles.*
mpirun -np 1 "$BIN" -in in.coefficient > output/screen 2>&1
"$PYTHON" check.py
