#!/bin/bash
#SBATCH --job-name=openedge_strong
#SBATCH --account=m5186
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --array=0-5
#SBATCH --output=logs/strong_n%a_%j.out

# Strong scaling on Perlmutter Milan CPU.
# Array index -> rank count; same problem size for all.
RANKS_LIST=(16 32 64 128 256 512)
NRANKS=${RANKS_LIST[$SLURM_ARRAY_TASK_ID]}

# Sizes for production-quality strong scaling. Adjust if memory or wall
# time bites at the high-rank end.
NLAUNCH=10           # nlaunchTotalW (per-PMI W launches)
NPHASE=30000         # main-phase steps
NWARM=2000           # adaptation + equilibration before timed phase

REPO=/global/u1/d/diawa/OpenEdge
BENCH=$REPO/bench/openedge/test_west_axi
DECK=$REPO/examples/test_west_axi/in.west
# Single binary: Kokkos+CUDA preset with SPARTA_MACHINE=mpi keeps the
# `spa_mpi` name. CPU runs use it without `-sf kk`; GPU runs add it.
SPA=/global/u1/d/diawa/build_gpu/src/spa_mpi

# Pin to one rank per core; nothing fancy on AMD Milan.
export OMP_NUM_THREADS=1

# Run from the bench dir, NOT from examples/. Symlink input/ from
# examples (single source of truth) so `read_surf input/wall.surf`
# resolves; output/ stays bench-local so dumps don't pollute the
# examples tree and compare_outputs.py finds them in one place.
cd $BENCH
[ -e input ]  || ln -sf $REPO/examples/test_west_axi/input input
mkdir -p output

srun --ntasks=$NRANKS --cpus-per-task=1 $SPA -in $DECK \
     -var nlaunchTotalW $NLAUNCH \
     -var Nwarm  $NWARM \
     -var Nphase $NPHASE \
     -var tag    "strong_n${NRANKS}" \
   | tee $BENCH/logs/strong_n${NRANKS}.log
