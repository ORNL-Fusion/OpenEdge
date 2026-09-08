#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
openedge_bin="${OPENEDGE_BIN:-${HOME}/build_oe/src/spa_mac_mpi}"
np="${NP:-2}"

cd "$root"
for case in hs000 hs025 hs050 hs075 hs100 hs150 hs200; do
  case "$case" in
    hs000) scale=0.0 ;;
    hs025) scale=0.25 ;;
    hs050) scale=0.50 ;;
    hs075) scale=0.75 ;;
    hs100) scale=1.00 ;;
    hs150) scale=1.50 ;;
    hs200) scale=2.00 ;;
  esac
  outdir="output_core_transit_heat_scan_${case}"
  echo "=== OpenEdge OML heat scale ${scale} -> ${outdir}"
  mpirun -np "$np" "$openedge_bin" \
    -var heat_scale "$scale" -var outdir "$outdir" \
    -in in.core_transit_size_scan
done
