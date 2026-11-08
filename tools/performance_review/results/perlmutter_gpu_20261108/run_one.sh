#!/bin/bash -l
# usage: run_one.sh MODE TAG NRANKS [STEPS]   MODE = cpu | gpu | gpu-nsys | gpu-counts
# Runs the WEST prevention_fixed_on case through the production run_case.sh (unchanged) inside the
# interactive allocation (SLURM_JOB_ID must be exported). Output: $PSCRATCH/openedge-runs/perf_review_20261108/TAG/
set -euo pipefail
mode=$1; tag=$2; nranks=$3; steps=${4:-20000}
S=/tmp/claude-99462/-pscratch-sd-d-diawa/61bde2e5-a5f0-4c23-81ae-bc15afe47d89/scratchpad
BUNDLE=/pscratch/sd/d/diawa/slag_mb_pilot_source_on_matched/west_mb8_continuous_ops_20260922_bundle
PILOT=$BUNDLE/examples/wip/slag/slag_mb_pilot
CASE=$PILOT/production/west_mb8_continuous_ops_20260922
RUN_ROOT=$PSCRATCH/openedge-runs/perf_review_20261108/$tag
export OPENEDGE_ROOT=$BUNDLE
jobid=${OE_JOBID:?export OE_JOBID (allocation job id)}
for v in $(env | grep -o "^SLURM_[A-Za-z_]*"); do unset $v; done   # this shell lives inside another Slurm job
export SLURM_JOB_ID=$jobid SLURM_JOBID=$jobid
case $mode in
  cpu)
    module reset >/dev/null 2>&1; module load cpe/25.09 PrgEnv-gnu/8.6.0 cray-hdf5 python >/dev/null 2>&1
    module unload craype-accel-nvidia80 >/dev/null 2>&1 || true
    export MPICH_GPU_SUPPORT_ENABLED=0 OMP_NUM_THREADS=1
    export OE_PROFILE_BINARY=${OE_BIN:-$PSCRATCH/openedge-build-a508aff-cpu/src/spa_mpi}
    unset OE_KOKKOS_ARGS ;;
  gpu|gpu-nsys|gpu-counts)
    EXE=${OE_BIN:-$PSCRATCH/openedge-build-a508aff-gpu/src/spa_kokkos_cuda_perlmutter}
    export EXE; source $HOME/OpenEdge/scripts/perlmutter_gpu_env.sh   # ipc line: GTL + CUDA IPC, -k on g 4
    module load python >/dev/null 2>&1
    export OE_PROFILE_BINARY=$EXE
    export OE_KOKKOS_ARGS="${OE_KK} -sf kk -pk kokkos react/retry yes gpu/aware ${OE_PK_AWARE} comm threaded"
    export SLURM_GPUS_PER_NODE=4 SLURM_GPU_BIND=none
    [[ $mode == gpu-nsys ]] && export KOKKOS_TOOLS_LIBS=$PSCRATCH/oe-perf-tools/kp_nvtx.so OE_NSYS_RANKS=${OE_NSYS_RANKS:-0} NSYS=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys
    [[ $mode == gpu-counts ]] && export KOKKOS_TOOLS_LIBS=$PSCRATCH/oe-perf-tools/kp_counts.so OE_KOKKOS_COUNTS_DIR=$RUN_ROOT ;;
  *) echo bad mode; exit 2 ;;
esac
export OE_PROFILE_DIR=$RUN_ROOT/profile_gen
mkdir -p $RUN_ROOT
env | grep -E '^(OE_|SLURM_GPU|MPICH|FI_|KOKKOS|EXE|OMP)' > $RUN_ROOT/launch_env.txt
cd $PILOT
OPENEDGE_LAUNCHER=srun OPENEDGE_BIN=/pscratch/sd/d/diawa/oe-perf-tools/oe_wrap.py PYTHON_BIN=python3 RUN_ROOT=$RUN_ROOT \
RUN_STEPS_OVERRIDE=$steps STATE_EVERY=$steps SURFACE_EVERY=$steps DENSITY_EVERY=$steps RESTART_EVERY=$steps \
DENSITY_REPEAT=$(( steps/250 > 0 ? steps/250 : 1 )) STATS_EVERY=1000 CHECKPOINT_STEPS_OVERRIDE=0 \
  $CASE/run_case.sh prevention_fixed_on $nranks 2>&1 | tee $RUN_ROOT/run_case.out
