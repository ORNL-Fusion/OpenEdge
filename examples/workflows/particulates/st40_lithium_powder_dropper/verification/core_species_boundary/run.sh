#!/usr/bin/env bash
set -euo pipefail

OPENEDGE_BIN=${OPENEDGE_BIN:-${HOME}/build_oe/src/spa_mac_mpi}
NP=${NP:-2}
PYTHON=${PYTHON:-python3}

rm -rf output_absorb output_reflect log.openedge
mkdir -p output_absorb output_reflect
mpirun -np "${NP}" "${OPENEDGE_BIN}" -in in.openedge
"${PYTHON}" scripts/check_result.py
mpirun -np "${NP}" "${OPENEDGE_BIN}" -in in.reflect
"${PYTHON}" scripts/check_reflection.py
