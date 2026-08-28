#!/bin/bash
# D2 chemistry GPU verification (chem A1b): device run + host control,
# with automatic PASS/FAIL summary. Usage:
#   bash run_gpu_check.sh [/path/to/spa_kokkos_cuda]
set -u
cd "$(dirname "$0")"
EXE=${1:-}
if [ -z "$EXE" ]; then
  for c in $HOME/openedge-build-cuda/src/spa_kokkos_cuda \
           $PSCRATCH/openedge-build-gpu2/src/spa_kokkos_cuda_perlmutter 2>/dev/null; do
    [ -x "$c" ] && EXE=$c && break
  done
fi
[ -x "${EXE:-}" ] || { echo "ERROR: CUDA binary not found; pass it as arg 1"; exit 1; }
export OPENEDGE_ROOT=$(cd ../../.. && pwd)
echo "exe: $EXE"

run() {
  env $2 "$EXE" -k on g 1 -sf kk -pk kokkos react/retry yes \
      -in in.d2_chem -log log.$1 > out.$1 2>&1
  local np=$(awk '/^Step/{on=1;next} on && /^ *[0-9]+ /{last=$2} END{print last}' log.$1)
  local frac=$(awk '/^Step/{on=1;next} on && /^ *[0-9]+ /{d=$4;dp=$5} END{if(d+dp>0) printf "%.4f", dp/(d+dp)}' log.$1)
  local done=$(grep -ac 'D2CHEM_COMPLETE' out.$1)
  local path=$(grep -a 'volume/chem/adas/kk' out.$1 | head -1)
  echo "== $1: complete=$done final_np=$np ionized_frac=$frac"
  echo "   $path"
  [ "$done" = 1 ] && [ "$np" = 4000 ] && echo "   PASS" || echo "   FAIL (want complete=1, np=4000)"
}
run device _=1
run hostctl OE_CHEM_HOST=1
echo "Device vs control ionized fractions should agree within a few percent."
