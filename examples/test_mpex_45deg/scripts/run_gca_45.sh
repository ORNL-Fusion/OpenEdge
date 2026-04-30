#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENEDGE_BIN="${OPENEDGE_BIN:-}"
NP="${NP:-1}"

if [[ -z "${OPENEDGE_BIN}" ]]; then
  echo "Set OPENEDGE_BIN to your spa_mpi executable before running this script." >&2
  exit 1
fi

cd "${ROOT}"
exec mpirun -np "${NP}" "${OPENEDGE_BIN}" -in input/in.mpex_45_gca
