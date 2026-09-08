#!/usr/bin/env bash
# Run the native-VTU example and verify that both ParaView files were written.

set -euo pipefail

cd "$(dirname "$0")"

np=${NP:-1}

supports_native_vtk() {
  local help
  # SPARTA prints valid help text but exits nonzero after handling -help.
  help=$("$1" -help 2>&1) || true
  [[ "$help" == *"grid/vtk"* && "$help" == *"surf/vtk"* ]]
}

requested_bin=${1:-${OPENEDGE_BIN:-}}
if [[ -n "$requested_bin" ]]; then
  if [[ ! -x "$requested_bin" ]]; then
    printf 'ERROR: OpenEdge executable not found: %s\n' "$requested_bin" >&2
    exit 2
  fi
  if ! supports_native_vtk "$requested_bin"; then
    printf 'ERROR: %s was built without the native VTK dump styles.\n' "$requested_bin" >&2
    printf 'Reconfigure with -DPKG_VTK=ON or pass a VTK-enabled executable.\n' >&2
    exit 2
  fi
  bin=$requested_bin
else
  bin=
  candidates=(
    "$HOME"/build_oe/src/spa_*
    "$HOME"/build_oe_vtk*/src/spa_*
    "$HOME"/buildOpenEdge*/src/spa_*
  )
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && supports_native_vtk "$candidate"; then
      bin=$candidate
      break
    fi
  done
  if [[ -z "$bin" ]]; then
    printf 'ERROR: no VTK-enabled OpenEdge executable was found.\n' >&2
    printf 'Build with -DPKG_VTK=ON, then pass the executable to ./run.sh.\n' >&2
    exit 2
  fi
fi

printf 'Using VTK-enabled executable: %s\n' "$bin"

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
