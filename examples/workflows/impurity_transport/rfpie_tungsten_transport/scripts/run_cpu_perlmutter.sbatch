#!/bin/bash -l
# CPU twin of run_gpu_perlmutter.sbatch: same deck and settings on one CPU node (128 MPI ranks).
#   sbatch scripts/run_cpu_perlmutter.sbatch            (from the case folder)
# Env: RUNDIR (default $PSCRATCH/openedge-runs/rfpie_fix) receives runs/cpu_zramp_hi;
#      EXE    (default $PSCRATCH/openedge-build-kkhost4/src/spa_kokkos_omp, run on the pure CPU path)
#SBATCH -J rfpie_cpu_zramp_hi -q regular -t 00:30:00 -N 1 -A m5186 -C cpu --ntasks-per-node=128 --cpus-per-task=2
#SBATCH -o slurm-%x-%j.out -e slurm-%x-%j.err
module reset >/dev/null 2>&1; module load cpe/25.09 PrgEnv-gnu cray-hdf5 python >/dev/null 2>&1; module unload craype-accel-nvidia80 >/dev/null 2>&1
export OMP_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 OPENEDGE_ROOT=${OPENEDGE_ROOT:-$HOME/OpenEdge}
EXE=${EXE:-$PSCRATCH/openedge-build-kkhost4/src/spa_kokkos_omp}
CASE=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
RUN=${RUNDIR:-$PSCRATCH/openedge-runs/rfpie_fix}/cpu_zramp_hi
mkdir -p "$RUN" && cp "$CASE/in.openedge" "$RUN/" && rm -rf "$RUN/input" && cp -r "$CASE/input" "$RUN/input"
mkdir -p "$CASE/runs" && ln -sfn "$RUN" "$CASE/runs/cpu_zramp_hi"
cd "$RUN" && rm -rf output
V="-var plasmafile input/plasma_he_zramp.h5 -var nlaunch 5000 -var popnmax 5000 -var nx 128 -var ny 128 -var nz 192 -var stateevery 10000 -var diagevery 2000 -var dumpevery 10000 -var outdir output"
srun -n 128 --cpu-bind=cores $EXE -in in.openedge $V -log output_log.cpu > screen.cpu 2>&1; echo "cpu_zramp_hi rc=$?"
grep -E 'ERROR|Segmentation|Loop time|Performance' screen.cpu | tail -3 | cut -c1-120
