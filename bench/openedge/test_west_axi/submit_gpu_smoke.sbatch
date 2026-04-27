#!/bin/bash
#SBATCH --job-name=openedge_gpu_smoke
#SBATCH --account=<PROJECT>
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/gpu_smoke_%j.out

# Small smoke run for Phase D correctness validation on GPU. Pairs with
# submit_cpu_smoke.sbatch — diff the W density dumps with
# compare_outputs.py to confirm the GPU sheath spatial port matches CPU.

NLAUNCH=2
NPHASE=2000
NWARM=200

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
     -var tag    "gpu_smoke_4" \
   | tee $REPO/bench/openedge/test_west_axi/logs/gpu_smoke_4.log
