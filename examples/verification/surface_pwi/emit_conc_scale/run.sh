#!/usr/bin/env bash
# fix surface/emit/source conc_scale on a depleting B layer (WEST wedge).
#   (a) emitted B per event = sum F_i c_i A_i dt nevery with c from the previous
#       surface update, for a live (static no) and a frozen (static yes) compute
#   (b) the hottest coated faces deplete; bare faces switch on from c = 0
#   (c) static yes and static no emit the same B over the first 400 steps
#   (d) two surface/pwi writers of one concentration array: every event follows
# Exits 0 on PASS, 1 on FAIL.

set -e
cd "$(dirname "$0")"
rm -rf output && mkdir -p output

SPA=${SPA:-$HOME/build_oe/src/spa_mac_mpi}
NP=${NP:-4}
LAUNCH=${LAUNCH:-mpirun -np $NP}
SPA_ARGS=${SPA_ARGS:-}
PYTHON=${PYTHON:-python3}   # needs numpy

for c in no yes; do
  echo "=== static_$c"
  $LAUNCH "$SPA" $SPA_ARGS -in in.emit_conc -var case static_$c -var cstatic $c \
    -log output/log.static_$c > output/screen.static_$c 2>&1 \
    || { echo "run static_$c failed:"; tail -5 output/screen.static_$c; exit 1; }
done

echo "=== two_writers"
$LAUNCH "$SPA" $SPA_ARGS -in in.two_writers -log output/log.two_writers \
  > output/screen.two_writers 2>&1 \
  || { echo "run two_writers failed:"; tail -5 output/screen.two_writers; exit 1; }

$PYTHON check.py
