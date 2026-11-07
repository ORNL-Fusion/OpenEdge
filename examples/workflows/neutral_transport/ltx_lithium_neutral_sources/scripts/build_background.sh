#!/usr/bin/env bash
# Rebuild the LTX-beta background for OpenEdge from the SOLEDGE3X-EIRENE run
# (shot 103955, pure H plasma, Li wall; converged plasmaFinal.h5 of Sep 2024).
#
# Inputs : input/s3x/{refParam_raptorX,mesh,meshEIRENE,mesh_raptorX,plasmaFinal}.h5
#          copied from ~/Projects/LTX/ltx/run_dir (gitignored)
# Outputs: input/plasma_ltx.h5     plasma + B on the EIRENE triangles (what in.openedge reads)
#          input/ltx_wall_axi.surf S3X wall as a SPARTA axi surface, (x, y) = (Z, R)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/Users/42d/venvs/openedge-env/bin/python}   # needs h5py, scipy, matplotlib
CONV=../../../../tools/converters/convert_s3x_plasma.py
S3X=${S3X:-input/s3x}
SRC=${S3X_SRC:-$HOME/Projects/LTX/ltx/run_dir}

mkdir -p "$S3X"
for f in refParam_raptorX.h5 mesh.h5 meshEIRENE.h5 mesh_raptorX.h5 plasmaFinal.h5; do
  [ -f "$S3X/$f" ] || cp "$SRC/$f" "$S3X/$f"
done

"$PY" "$CONV" "$S3X" --plasma-snapshot plasmaFinal.h5 \
  --main-ion-spec 1 --main-ion-name H+ --main-ion-mass 1.00794 \
  --geometry axi \
  --plasma-out input/plasma_ltx.h5 --wall-out input/ltx_wall_axi.surf
echo "background: input/plasma_ltx.h5, wall: input/ltx_wall_axi.surf"
