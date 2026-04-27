#!/bin/bash
#SBATCH --job-name=openedge_cpu_smoke
#SBATCH --account=<PROJECT>
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/cpu_smoke_%j.out

# Small smoke run for Phase D correctness validation. Pairs with
# submit_gpu_smoke.sbatch — same problem size, same rank count, same
# tag-prefixed output. Runs in ~5 minutes on Perlmutter Milan.

NLAUNCH=2          # 2 W launches per PMI tally (vs 10 in match)
NPHASE=2000        # 2k phase steps (vs 30k in match)
NWARM=200          # 200 warm steps (vs 2k in match)

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
     -var tag    "cpu_smoke_4" \
   | tee $REPO/bench/openedge/test_west_axi/logs/cpu_smoke_4.log
