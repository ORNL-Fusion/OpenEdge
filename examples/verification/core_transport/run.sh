#!/usr/bin/env bash
# CoreTransport1D (src/core_transport_1d.cpp) vs Aurora's impden1 + conservation.  Exits 0 on PASS.
# The Aurora reference outputs are committed in reference/; set AURORA=/path/to/Aurora to regenerate.
set -e
cd "$(dirname "$0")"
mkdir -p output input
PYTHON=${PYTHON:-python3}
CXX=${CXX:-c++}
SRC=${SRC:-../../../src}
$PYTHON scripts/make_inputs.py input > /dev/null
$CXX -O2 -std=c++11 -I"$SRC" -o output/test_driver scripts/test_driver.cpp "$SRC/core_transport_1d.cpp"
if [ -n "$AURORA" ]; then
  gfortran -O2 -o output/aurora_ref "$AURORA/aurora/math.f90" "$AURORA/aurora/impden.f90" ref_aurora/aurora_ref.f90
  for c in neutral evolve closed; do output/aurora_ref input/case_$c.dat reference/ref_$c.dat; done
fi
for c in neutral evolve closed source source_fixed_n0; do
  output/test_driver input/case_$c.dat output/dens_$c.dat output/ledger_$c.dat
done
$PYTHON scripts/check_core_transport.py
$CXX -O2 -std=c++11 -I"$SRC" -o output/test_safety scripts/test_safety.cpp "$SRC/core_transport_1d.cpp"
output/test_safety
for c in neutral evolve closed source source_fixed_n0; do
  output/test_driver input/case_$c.dat output/safe_dens_$c.dat output/safe_ledger_$c.dat safe
done
$PYTHON scripts/check_core_transport.py --safe
$CXX -O2 -std=c++17 -I"$SRC" -o output/test_split scripts/test_split.cpp "$SRC/core_transport_1d.cpp"
output/test_split relax
output/test_split conv input/case_source.dat
