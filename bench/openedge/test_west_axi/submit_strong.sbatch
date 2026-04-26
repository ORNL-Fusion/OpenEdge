#!/bin/bash
#SBATCH --job-name=openedge_strong
#SBATCH --account=<PROJECT>
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

REPO=$(realpath "$(dirname "$0")/../../..")
DECK=$REPO/examples/test_west_axi/in.west
SPA=$REPO/../buildOpenEdge/src/spa_mpi   # adjust if your build is elsewhere

# Pin to one rank per core; nothing fancy on AMD Milan.
export OMP_NUM_THREADS=1

cd $REPO/examples/test_west_axi   # so `read_surf input/wall.surf` resolves

srun --ntasks=$NRANKS --cpus-per-task=1 $SPA -in $DECK \
     -var nlaunchTotalW $NLAUNCH \
     -var Nwarm  $NWARM \
     -var Nphase $NPHASE \
     -var tag    "strong_n${NRANKS}" \
   | tee $REPO/bench/openedge/test_west_axi/logs/strong_n${NRANKS}.log
