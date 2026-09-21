#!/bin/bash -l
# Saturated z-ramped rfpie case on one Perlmutter GPU node (4 x A100, 1 MPI rank per GPU).
#   sbatch scripts/run_gpu_perlmutter.sbatch            (from the case folder)
# Env: RUNDIR (default $PSCRATCH/openedge-runs/rfpie_fix) receives runs/gpu_zramp_hi;
#      EXE    (default from scripts/perlmutter_gpu_env.sh: the GTL binary with OE_GPU_MPI=ipc,
#              openedge-build-gpu2 with OE_GPU_MPI=host)
# Needs input/plasma_he_zramp.h5 (python scripts/build_plasma_zramp.py).
#SBATCH -J rfpie_gpu_zramp_hi -q regular -t 00:30:00 -N 1 -A m5186_g -C gpu --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=32
#SBATCH -o slurm-%x-%j.out -e slurm-%x-%j.err
source ${OPENEDGE_ROOT:-$HOME/OpenEdge}/scripts/perlmutter_gpu_env.sh   # toolchain pin, MPI mode (OE_GPU_MPI=ipc|host), EXE, srun/Kokkos options
CASE=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
RUN=${RUNDIR:-$PSCRATCH/openedge-runs/rfpie_fix}/gpu_zramp_hi
mkdir -p "$RUN" && cp "$CASE/in.openedge" "$RUN/" && rm -rf "$RUN/input" && cp -r "$CASE/input" "$RUN/input"
mkdir -p "$CASE/runs" && ln -sfn "$RUN" "$CASE/runs/gpu_zramp_hi"
cd "$RUN" && rm -rf output
# 50x markers, 2x finer grid in every direction, roulette cap raised, particle state dump only at the end
V="-var plasmafile input/plasma_he_zramp.h5 -var nlaunch 5000 -var popnmax 5000 -var nx 128 -var ny 128 -var nz 192 -var stateevery 10000 -var diagevery 2000 -var dumpevery 10000 -var outdir output"
srun -n 4 --cpu-bind=cores $OE_SRUN_GPU $EXE $OE_KK -sf kk -pk kokkos react/retry yes gpu/aware $OE_PK_AWARE comm threaded -in in.openedge $V -log output_log.gpu > screen.gpu 2>&1; echo "gpu_zramp_hi rc=$?"
grep -E 'ERROR|Segmentation|Loop time|Performance' screen.gpu | tail -4 | cut -c1-120
