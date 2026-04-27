#!/bin/bash
#SBATCH --job-name=openedge_cpu_match
#SBATCH --account=<PROJECT>
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

REPO=$(realpath "$(dirname "$0")/../../..")
DECK=$REPO/examples/test_west_axi/in.west
SPA=$REPO/../buildOpenEdge/src/spa_mpi

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

cd $REPO/examples/test_west_axi

srun --ntasks=4 --cpus-per-task=32 $SPA \
     -in $DECK \
     -var nlaunchTotalW $NLAUNCH \
     -var Nwarm  $NWARM \
     -var Nphase $NPHASE \
     -var tag    "cpu_match_4" \
   | tee $REPO/bench/openedge/test_west_axi/logs/cpu_match_4.log
