#!/bin/zsh
# Naujoks 1996 (NF 36 671) Fig. 7 scans. Usage: ./run_naujoks.sh [a0|a2|b|all]
#   a0 : alpha=0, single-ioniz, no E   -> must land on eq. (3)   [tag nj_a0]
#   a2 : alpha=2, single-ioniz, no E   -> Fig. 7(a)              [tag nj_a2]
#   b  : Fig. 7(b) curves               A: single-ioniz + sheath E
#                                       B: multi-ioniz, no E
#                                       C: multi-ioniz + sheath E
# All runs emit every step (nemit 1). Analysis: naujoks.ipynb.
set -e
cd "$(dirname "$0")"

SPA=~/build_oe/src/spa_mac_mpi
NE_LIST=(8.8e17 1.3e18 2.65e18 5.3e18 1.06e19 2.6e19 5.3e19 1.06e20)
PHYS=(-var Bmag 2.0 -var TeV 20.0 -var blat p -var nemit 1)
what=${1:-all}

run() { echo "=== $4 ne=$3 ==="; $SPA -in $1 $PHYS -var alpha $2 -var ne $3 \
        -var tag nj_$4_ne$3 "${@:5}" -log none >> naujoks.log; }

: > naujoks.log
for ne in $NE_LIST; do
  case $what in
    a0|all)  run in.naujoks_single 0.0 $ne a0 -var smode off ;|
    a2|all)  run in.naujoks_single 2.0 $ne a2 -var smode off ;|
    b|all)   run in.naujoks_single 2.0 $ne A -var smode spatial
             run in.naujoks_multi  2.0 $ne B -var smode off     -var dcad 10
             run in.naujoks_multi  2.0 $ne C -var smode spatial -var dcad 10 ;;
  esac
done
echo "done -> state/redepo.nj_*"
