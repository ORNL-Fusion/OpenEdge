#!/bin/bash
#SBATCH --job-name=openedge_cpu_match
#SBATCH --account=m5186
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/cpu_match_%j.out

# CPU baseline matched to submit_gpu.sbatch (4 ranks/node, same problem
# size). Pair this with the GPU run to (a) validate Phase D correctness
# by diffing the W density dump (compare_outputs.py) and (b) measure
# the GPU speedup at the actual rank count, not against single-rank.

NLAUNCH=10
NPHASE=30000
NWARM=2000

REPO=/global/u1/d/diawa/OpenEdge
BENCH=$REPO/bench/openedge/test_west_axi
DECK=$REPO/examples/test_west_axi/in.west
SPA=/global/u1/d/diawa/build_gpu/src/spa_mpi

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Run from the bench dir; symlink input/ from examples; output/ is
# bench-local so dumps land alongside the GPU run for compare_outputs.py.
cd $BENCH
[ -e input ]  || ln -sf $REPO/examples/test_west_axi/input input
mkdir -p output

srun --ntasks=4 --cpus-per-task=32 $SPA \
     -in $DECK \
     -var nlaunchTotalW $NLAUNCH \
     -var Nwarm  $NWARM \
     -var Nphase $NPHASE \
     -var tag    "cpu_match_4" \
   | tee $BENCH/logs/cpu_match_4.log
