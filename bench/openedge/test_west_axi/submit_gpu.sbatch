#!/bin/bash
#SBATCH --job-name=openedge_gpu
#SBATCH --account=m5186
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

REPO=/global/u1/d/diawa/OpenEdge
BENCH=$REPO/bench/openedge/test_west_axi
DECK=$REPO/examples/test_west_axi/in.west
# Same binary as CPU runs — Kokkos+CUDA build with SPARTA_MACHINE=mpi.
# `-k on g 1 -sf kk` in srun activates the Kokkos backend.
# Note: `g 1` is *GPUs per rank*, NOT total GPUs in job. With 4 ranks
# and --gpu-bind=single:1, each rank sees exactly 1 GPU. Setting g >1
# tries to allocate non-existent GPU ids and crashes Kokkos::initialize.
SPA=/global/u1/d/diawa/build_gpu/src/spa_mpi

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

cd $BENCH
[ -e input ]  || ln -sf $REPO/examples/test_west_axi/input input
mkdir -p output

srun --ntasks=4 --cpus-per-task=32 --gpu-bind=single:1 $SPA \
     -k on g 1 -sf kk \
     -in $DECK \
     -var nlaunchTotalW $NLAUNCH \
     -var Nwarm  $NWARM \
     -var Nphase $NPHASE \
     -var tag    "gpu_a100_4" \
   | tee $BENCH/logs/gpu_a100_4.log
