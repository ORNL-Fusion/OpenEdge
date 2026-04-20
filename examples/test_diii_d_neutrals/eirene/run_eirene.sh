#!/bin/bash
# Run standalone EIRENE against the input deck in this directory.
# The binary lives under /home/cloud/eirene_standalone/ (ITER Org repo,
# commit f8f63fa0, built with Intel ifx + MPI).
#
# Usage:
#   ./run_eirene.sh             # uses ./fort.1
#   ./run_eirene.sh fort.1.foo  # uses the file you pass
#
# The binary needs Intel MPI runtime libs in LD_LIBRARY_PATH. We source
# oneAPI here.
set -eu

INPUT=${1:-fort.1}
EIRENE_BIN=/home/cloud/eirene_standalone/EIRENE/binRelease/eirene

if [ ! -f "$INPUT" ]; then
  echo "error: input file '$INPUT' not found" >&2
  echo "  try: $0 fort.1.solps_ref" >&2
  exit 1
fi

if [ ! -L Database ]; then
  ln -sf /home/cloud/eirene_standalone/EIRENE/Database Database
fi

# The standalone binary reads fort.1 from the working directory, so make
# sure the file we want is at ./fort.1 (stage via symlink if needed).
if [ "$INPUT" != "fort.1" ]; then
  ln -sf "$INPUT" fort.1
fi

source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1

NP=${NP:-4}
echo "running: mpirun -np $NP $EIRENE_BIN  (input=$INPUT)"
mpirun -np "$NP" "$EIRENE_BIN" 2>&1 | tee run.log
