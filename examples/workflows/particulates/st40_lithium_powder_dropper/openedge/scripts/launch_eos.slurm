#!/bin/bash
#SBATCH --job-name=st40_li
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-st40-%j.out
# usage (from the case dir on EOS):
#   sbatch --ntasks=16 scripts/launch_eos.slurm            # nsteps from the deck
#   sbatch --ntasks=16 --export=ALL,STEPS=2500000 scripts/launch_eos.slurm
# Output: runs/s<STEPS>_n<ranks>_<jobid>/ (+ .log); pull with scripts/sync_eos.sh pull
set -euo pipefail
module purge; module load openmpi gcc hdf5/1.14.5_cpp
cd "$SLURM_SUBMIT_DIR"
export OPENEDGE_ROOT=${OPENEDGE_ROOT:-/scratch/home_eos/42d/OpenEdge_slag}
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=vader,self OMPI_MCA_btl_vader_single_copy_mechanism=none
SPA=${SPA:-/scratch/home_eos/42d/build_oe_slag/src/spa_mpi}
DECK_STEPS=$(awk '$1=="variable" && $2=="nsteps" {print $4}' in.openedge)
STEPS=${STEPS:-$DECK_STEPS}
OUT=${OUT:-runs/s${STEPS}_n${SLURM_NTASKS}_${SLURM_JOB_ID}}
mkdir -p runs
sha256sum "$SPA" > "$OUT.binary_sha256"
mpirun -np "$SLURM_NTASKS" --bind-to none "$SPA" -in in.openedge \
  -var nsteps "$STEPS" -var outdir "$OUT" -log "$OUT.log" && touch "$OUT/COMPLETED"
