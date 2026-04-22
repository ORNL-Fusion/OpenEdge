#!/bin/bash
# Run OpenEdge on the DIII-D plasma-neutral test case.
#
# Usage:
#   ./run_openedge.sh                      # in.diii_d_neutrals, 16 ranks
#   NP=32 ./run_openedge.sh                # change rank count
set -eu

INPUT=${1:-in.diii_d_neutrals}
NP=${NP:-16}
SPA_BIN=${SPA_BIN:-/home/cloud/buildOpenEdge/src/spa_mpi}

if [ ! -f "$INPUT" ]; then
  echo "error: input file '$INPUT' not found" >&2
  exit 1
fi

mkdir -p ../output

# setvars.sh references unset vars; relax -eu across the source so it
# doesn't silently abort the script.
set +eu
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1 || true
set -eu
LOG=../output/run_$(basename "$INPUT" .diii_d_neutrals | sed 's/^in\.//').log

echo "running: mpirun -np $NP $SPA_BIN -in $INPUT  > $LOG"
mpirun -np "$NP" "$SPA_BIN" -in "$INPUT" > "$LOG" 2>&1
echo "done. tail -f $LOG"
