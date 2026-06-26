#!/bin/bash
# End-to-end sputter validation pipeline:
#   1. Sync the freshly-built binary (in case ~/buildOpenEdge_clean/ has the
#      latest compute_surface_physical_sputter changes but ~/buildOpenEdge/
#      is stale).
#   2. Populate /mesh/wall_surf_cell in plasma.h5 from the actual wall.surf,
#      so the OE compute can use the SOLPS target-adjacent cell directly.
#   3. Diagnose: for each lower-divertor segment, print the cell it maps to
#      and the Te/ni at that cell. If those are all zero, the wall_surf_cell
#      mapping is hitting PFR/vacuum cells and we need a smarter mapper.
#   4. Run mpirun -> sputter.dump.
#   5. Plot.
#
# Usage:  cd coupled_test && ./run_sputter_validation.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OE_DIR="openedge"
ANALYSIS="../analysis"
TOOLS="/home/cloud/OpenEdge/tools/converters"

OLD_BIN="$HOME/buildOpenEdge/src/spa_mpi"
NEW_BIN="$HOME/buildOpenEdge_clean/src/spa_mpi"

# 1. Rebuild + sync binary
BUILD_DIR="$HOME/buildOpenEdge_clean"
if [[ -d "$BUILD_DIR" ]]; then
    echo "[1/5] rebuilding $BUILD_DIR"
    ( cd "$BUILD_DIR" && LD_LIBRARY_PATH= make -j$(nproc) spa_mpi 2>&1 \
        | tail -5 )
fi
if [[ -f "$NEW_BIN" ]]; then
    if ! cmp -s "$NEW_BIN" "$OLD_BIN" 2>/dev/null; then
        echo "  syncing $NEW_BIN -> $OLD_BIN"
        cp "$NEW_BIN" "$OLD_BIN"
    fi
else
    echo "  WARNING: $NEW_BIN not found; using $OLD_BIN as-is"
fi
md5sum "$OLD_BIN"

# 2. Populate wall_surf_cell in plasma.h5
echo
echo "[2/5] populating plasma.h5 /mesh/wall_surf_cell"
python3 "$TOOLS/add_wall_surf_cell.py" \
    "$OE_DIR/plasma.h5" "../input/wall.surf"

# 3. Diagnose: print each lower-divertor segment's mapped cell + Te/ni
echo
echo "[3/5] diagnostic: cells the wall_surf_cell map points to (lower divertor)"
python3 << 'PY'
import h5py, numpy as np
from pathlib import Path
plasma_h5 = Path('openedge/plasma.h5')
wall = Path('../input/wall.surf')
with h5py.File(plasma_h5, 'r') as f:
    wsc = f['mesh/wall_surf_cell'][:]
    Te = f['mesh/temp_e'][:]
    Ne = f['mesh/dens_e'][:]
    Ni = f['mesh/dens_i'][:]
# parse wall.surf
text = wall.read_text().splitlines()
pts, lns, sec = {}, [], None
for line in text:
    s = line.strip()
    if s == 'Points': sec='p'; continue
    if s == 'Lines':  sec='l'; continue
    if not s or not s[0].isdigit(): continue
    tok = s.split()
    if sec=='p': pts[int(tok[0])] = (float(tok[1]), float(tok[2]))
    elif sec=='l': lns.append((int(tok[0]), int(tok[1]), int(tok[2])))
print(f"  {'sid':>4} {'R':>7} {'Z':>7} {'cell':>5}  {'Te':>9} {'ne':>10} {'ni':>10}")
for sid, p1, p2 in lns:
    z = 0.5*(pts[p1][0]+pts[p2][0]); r = 0.5*(pts[p1][1]+pts[p2][1])
    if z >= -3.0: continue
    c = int(wsc[sid-1])
    if 0 <= c < len(Te):
        print(f"  {sid:>4d} {r:>7.3f} {z:>7.3f} {c:>5d}  "
              f"{Te[c]:>9.3e} {Ne[c]:>10.3e} {Ni[c]:>10.3e}")
    else:
        print(f"  {sid:>4d} {r:>7.3f} {z:>7.3f} {c:>5d}  out of range")
PY

# 4. Run OpenEdge
echo
echo "[4/5] mpirun -np 4 $OLD_BIN -in $OE_DIR/in.solo"
( cd "$OE_DIR" && mpirun -np 4 "$OLD_BIN" -in in.solo > /tmp/sputter_run.log 2>&1 )
tail -3 /tmp/sputter_run.log
ls -la "$OE_DIR/sputter.dump"

# 5. Plot
echo
echo "[5/5] plotting"
python3 "$ANALYSIS/plot_sputter_along_wall.py"
