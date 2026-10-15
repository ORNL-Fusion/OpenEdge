#!/usr/bin/env bash
# Rebuild the SOLPS-13589_2 background for OpenEdge.
#
# One output, plasma_13589.h5: native B2 cells + EIRENE triangles, total heat
# flux rebuilt as in SOLPS-routines calc_fht, exponential extension into the
# wall gap.  The central hole inside the SOLPS core boundary stays vacuum; the
# decks absorb at that contour (input/st40_core.surf, psi_N 0.615 = inner B2 row).
# ../solps/baserun is Erin's Oct 14 baserun.zip (the regridded b2fgmtry that the
# run used); the zip's original baserun carried the stale August grid.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/Users/42d/venvs/openedge-env/bin/python}   # needs h5py, scipy, freeqdsk
CONV=../../../../../tools/converters/convert_solps_plasma.py
SOLPS=../solps
RUN=$SOLPS/run_dir
B2FGMTRY=${B2FGMTRY:-$SOLPS/baserun/b2fgmtry}
GFILE=$SOLPS/run_dir/gfiles/g013589.00176.geqdsk
LAMBDA_N=${LAMBDA_N:-fit}
LAMBDA_T=${LAMBDA_T:-fit}

FINAL=${OUT:-input/plasma_13589.h5}             # what in.openedge reads (gitignored)
"$PY" "$CONV" "$RUN" --b2fgmtry "$B2FGMTRY" --gfile "$GFILE" \
  --geometry axi --wall-in input/st40_wall_axi.surf \
  --heatflux total \
  --gap-extrapolate --gap-lambda-n "$LAMBDA_N" --gap-lambda-t "$LAMBDA_T" \
  --plasma-out "$FINAL"

"$PY" scripts/check_equilibrium_orientation.py "$FINAL"
echo "background: $FINAL (native SOLPS cells + wall-gap extension; core hole left as vacuum)"

