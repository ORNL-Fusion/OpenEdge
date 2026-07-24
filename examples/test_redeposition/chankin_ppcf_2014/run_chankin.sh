#!/bin/zsh
# Chankin 2014 (PPCF 56 025003) scans. Usage: ./run_chankin.sh [single|multi|full|orbits|all]
#   single : single-ioniz, no E   -> Fussman reference p^2/(1+p^2)  [tag single]
#   multi  : multi-ioniz,  no E   -> stranded high-Z ions           [tag multi]
#   full   : multi-ioniz + sheath E (MPS/DS)                        [tag full]
#   orbits : Fig-3 launch of the atoms in particles.dat             [tag fan]
# Deck defaults: B=5.3 T, alpha=3 deg, Te=100 eV. Emission every step (nemit 1).
# Periodic laterals (infinite target); fate classified per particle from the
# dumps by analysis.ipynb.
set -e
cd "$(dirname "$0")"

SPA=~/build_oe/src/spa_mac_mpi
NE_LIST=(3e18 1e19 3e19 1e20 3e20 1e21 3e21 1e22)
what=${1:-all}

run() { echo "=== $3 ne=$2 ==="; $SPA -in $1 -var nemit 1 -var ne $2 \
        -var tag $3_ne$2 "${@:4}" -log none >> chankin.log; }

: > chankin.log
if [[ $what == orbits || $what == all ]]; then
  echo "=== fan (Fig-3 orbits) ==="
  $SPA -in in.chankin_multi -var orbit 1 -var alpha 0 -var ne 1e21 \
       -var smode off -var y0 -1000 -var nrun 700 -var tag fan \
       -log none >> chankin.log
fi
for ne in $NE_LIST; do
  case $what in
    single|all) run in.chankin_single $ne single -var blat p -var smode off ;|
    multi|all)  run in.chankin_multi  $ne multi  -var smode off     -var dcad 10 ;|
    full|all)   run in.chankin_multi  $ne full   -var smode spatial -var dcad 10 ;;
  esac
done

# fig6: Chankin Fig. 6 / tables 3-4 conditions. Monoenergetic W0 at
# E0 = R_E * (3/2)kTi (Te=Ti=100 eV), multi-ioniz + sheath E, p scanned via ne.
if [[ $what == fig6 ]]; then
  RE_TAGS=(01 03 1 3)          # R_E = 0.1 0.3 1 3
  RE_EV=(15 45 150 450)        # E0 [eV]
  NE6_LIST=(1e18 3e18 1e19 3e19 1e20 3e20 1e21)
  for i in {1..4}; do
    for ne in $NE6_LIST; do
      run in.chankin_multi $ne re${RE_TAGS[i]} -var smode spatial -var dcad 10 \
          -var emodel fixed_energy -var epar ${RE_EV[i]}
    done
  done
fi
echo "done -> state/redepo.*"
