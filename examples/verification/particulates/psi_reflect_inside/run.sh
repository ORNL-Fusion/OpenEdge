#!/bin/sh
# Regression for fix reflect/psi: particles that start inside the contour in
# reflect mode must be rejected and tallied, not abort the run.
# Usage: ./run.sh [/path/to/spa binary]   (MPIRUN="" to launch without mpirun)
BIN=${1:-$HOME/build_oe/src/spa_mac_mpi}
NP=${NP:-1}
MPIRUN=${MPIRUN-mpirun -np $NP}
PY=${PYTHON:-python3}
cd "$(dirname "$0")"
mkdir -p output
rm -f output/log.* output/screen.* output/exit.*
$MPIRUN "$BIN" -in in.inside -log output/log.normal -echo none > output/screen.normal 2>&1
echo "normal run exit $?" > output/exit.normal
OE_PSI_STRICT=1 $MPIRUN "$BIN" -in in.inside -log output/log.strict -echo none > output/screen.strict 2>&1
echo "strict run exit $?" > output/exit.strict
$PY scripts/check_inside.py
