#!/bin/bash -l
# Saturated z-ramped rfpie case on one Perlmutter GPU node (4 x A100, 1 MPI rank per GPU).
#   sbatch scripts/run_gpu_perlmutter.sbatch            (from the case folder)
# Env: RUNDIR (default $PSCRATCH/openedge-runs/rfpie_fix) receives runs/gpu_zramp_hi;
#      EXE    (default $PSCRATCH/openedge-build-gpu2/src/spa_kokkos_cuda_perlmutter)
# Needs input/plasma_he_zramp.h5 (python scripts/build_plasma_zramp.py).
#SBATCH -J rfpie_gpu_zramp_hi -q regular -t 00:30:00 -N 1 -A m5186_g -C gpu --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=32 --gpus-per-task=1
#SBATCH -o slurm-%x-%j.out -e slurm-%x-%j.err
module reset >/dev/null 2>&1; module load cpe/25.09 PrgEnv-gnu cudatoolkit/12.9 cray-hdf5 python >/dev/null 2>&1; module unload craype-accel-nvidia80 >/dev/null 2>&1
export OMP_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 OPENEDGE_ROOT=${OPENEDGE_ROOT:-$HOME/OpenEdge}; unset CRAY_ACCEL_TARGET
EXE=${EXE:-$PSCRATCH/openedge-build-gpu2/src/spa_kokkos_cuda_perlmutter}
CASE=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
RUN=${RUNDIR:-$PSCRATCH/openedge-runs/rfpie_fix}/gpu_zramp_hi
mkdir -p "$RUN" && cp "$CASE/in.openedge" "$RUN/" && rm -rf "$RUN/input" && cp -r "$CASE/input" "$RUN/input"
mkdir -p "$CASE/runs" && ln -sfn "$RUN" "$CASE/runs/gpu_zramp_hi"
cd "$RUN" && rm -rf output
# 50x markers, 2x finer grid in every direction, roulette cap raised, particle state dump only at the end
V="-var plasmafile input/plasma_he_zramp.h5 -var nlaunch 5000 -var popnmax 5000 -var nx 128 -var ny 128 -var nz 192 -var stateevery 10000 -var diagevery 2000 -var dumpevery 10000 -var outdir output"
srun -n 4 --cpu-bind=cores --gpu-bind=single:1 $EXE -k on g 1 -sf kk -pk kokkos react/retry yes gpu/aware no comm threaded -in in.openedge $V -log output_log.gpu > screen.gpu 2>&1; echo "gpu_zramp_hi rc=$?"
grep -E 'ERROR|Segmentation|Loop time|Performance' screen.gpu | tail -4 | cut -c1-120
