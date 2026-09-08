#!/usr/bin/env bash
# Run the native-VTU example and verify that both ParaView files were written.

set -euo pipefail

cd "$(dirname "$0")"

bin=${1:-${OPENEDGE_BIN:-$HOME/build_oe/src/spa_mac_mpi}}
np=${NP:-1}

if [[ ! -x "$bin" ]]; then
  printf 'ERROR: OpenEdge executable not found: %s\n' "$bin" >&2
  printf 'Pass its path as the first argument or set OPENEDGE_BIN.\n' >&2
  exit 2
fi

# Disable the startup log in this directory; the input opens output/log.openedge.
mpirun -np "$np" "$bin" -log none -in in.openedge

for file in output/grid_0.vtu output/surface_0.vtu; do
  if [[ ! -s "$file" ]]; then
    printf 'FAIL: missing or empty %s\n' "$file" >&2
    exit 1
  fi
  if ! grep -q '<VTKFile type="UnstructuredGrid"' "$file"; then
    printf 'FAIL: %s is not a VTK XML UnstructuredGrid file\n' "$file" >&2
    exit 1
  fi
  if command -v xmllint >/dev/null 2>&1; then
    xmllint --noout "$file"
  fi
done

printf 'PASS: native VTU files are in %s/output\n' "$PWD"
