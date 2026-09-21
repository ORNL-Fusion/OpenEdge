# Perlmutter GPU launch environment for OpenEdge Kokkos/CUDA runs. Source it (bash -l) in
# every sbatch / interactive launcher; it pins the toolchain and selects the MPI mode.
#   OE_GPU_MPI=ipc  (default) GPU-aware Cray MPICH with CUDA IPC, GTL-linked binary, all 4 GPUs
#                   visible per rank and Kokkos picking by rank: Comm -0.8 s per 1000 steps on the
#                   rfpie perf deck vs host staging (GPU_PORTING.md sections 13 and 16, 2026-09-14)
#   OE_GPU_MPI=host host-staged particle exchange, non-GTL binary, one GPU bound per rank
# Exports: EXE (unless preset), OE_SRUN_GPU (srun GPU options), OE_KK ("-k on g N"), OE_PK_AWARE.
# Sbatch headers must request --gpus-per-node=4 without --gpus-per-task (per-task GPU binding
# hides the peer GPUs and breaks IPC: "cuIpcOpenMemHandle: invalid argument").
module reset >/dev/null 2>&1
module load cpe/25.09 >/dev/null 2>&1                    # site default moved to cpe/26.03 + CUDA 13.2 on 2026-09-12
module load PrgEnv-gnu/8.6.0 cudatoolkit/12.9 cray-hdf5 python >/dev/null 2>&1
export OMP_NUM_THREADS=1 OPENEDGE_ROOT=${OPENEDGE_ROOT:-$HOME/OpenEdge}
export FI_MR_CACHE_MONITOR=memhooks   # multi-node: the site default userfaultfd monitor aborts the first fix balance cell migration (cxil_map write error)
export MPICH_GPU_IPC_ENABLED=1
case "${OE_GPU_MPI:-ipc}" in
  ipc)
    module load craype-accel-nvidia80 >/dev/null 2>&1
    export MPICH_GPU_SUPPORT_ENABLED=1
    # the cpe/25.09 GTL wants libcudart.so.13 next to the application's .so.12
    export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64:$LD_LIBRARY_PATH
    export EXE=${EXE:-$PSCRATCH/openedge-build-gpu2gtl/src/spa_kokkos_cuda_perlmutter}
    export OE_SRUN_GPU="--gpus-per-node=4 --gpu-bind=none" OE_KK="-k on g 4" OE_PK_AWARE=yes ;;
  host)
    module unload craype-accel-nvidia80 >/dev/null 2>&1; unset CRAY_ACCEL_TARGET
    export MPICH_GPU_SUPPORT_ENABLED=0
    export EXE=${EXE:-$PSCRATCH/openedge-build-gpu2/src/spa_kokkos_cuda_perlmutter}
    export OE_SRUN_GPU="--gpus-per-task=1 --gpu-bind=single:1" OE_KK="-k on g 1" OE_PK_AWARE=no ;;
  *) echo "perlmutter_gpu_env.sh: OE_GPU_MPI must be ipc or host" >&2; return 1 2>/dev/null || exit 1 ;;
esac
