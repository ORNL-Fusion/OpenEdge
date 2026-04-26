#!/bin/bash
#SBATCH --job-name=openedge_gpu
#SBATCH --account=<PROJECT>
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/gpu_%j.out

# Perlmutter A100 single-node GPU run.
# 4 ranks, 1 A100 each. With current Kokkos coverage (see
# GPU_PORT_STATUS.md), several per-particle fixes still run on host;
# expect host<->device shuffle overhead to dominate. The benchmark
# documents this baseline.

NLAUNCH=10
NPHASE=30000
NWARM=2000

REPO=$(realpath "$(dirname "$0")/../../..")
DECK=$REPO/examples/test_west_axi/in.west
SPA=$REPO/../buildOpenEdge/src/spa_kokkos_cuda

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

cd $REPO/examples/test_west_axi

srun --ntasks=4 --cpus-per-task=32 --gpu-bind=single:1 $SPA \
     -k on g 4 -sf kk \
     -in $DECK \
     -var nlaunchTotalW $NLAUNCH \
     -var Nwarm  $NWARM \
     -var Nphase $NPHASE \
     -var tag    "gpu_a100_4" \
   | tee $REPO/bench/openedge/test_west_axi/logs/gpu_a100_4.log
