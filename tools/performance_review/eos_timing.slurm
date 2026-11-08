#!/bin/bash
#SBATCH -J oe-perf-timing
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --exclude=eos01
#SBATCH -n 48
#SBATCH --mem=96G
#SBATCH -t 02:00:00
set -euo pipefail
module purge
module load gcc/15.2.0 openmpi/4.1.7-gcc15.2.0 cmake hdf5/1.14.5_cpp python/3.12.3
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=vader,self
export OMPI_MCA_btl_vader_single_copy_mechanism=none UCX_TLS=sm,self HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=1
root=${OE_REVIEW_ROOT:-/scratch/home_eos/42d/oe_perf_review_20261107}
mkdir -p "$root/timing_evidence"
hostname > "$root/timing_evidence/host.txt"
for variant in baseline patched; do
  src=$root/source
  [[ $variant != patched ]] || src=$root/source_patched
  cmake -S "$src/cmake" -B "$root/build_release_$variant" -DPKG_OPENEDGE=ON -DPKG_VTK=ON \
    -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
    -DVTK_DIR=/scratch/home_eos/42d/sw/vtk-9.3.1-minimal-gcc15/lib64/cmake/vtk-9.3 \
    -DBUILD_MPI=ON -DSPARTA_MACHINE=perf -DCMAKE_BUILD_TYPE=Release \
    > "$root/timing_evidence/configure_$variant.log" 2>&1
  cmake --build "$root/build_release_$variant" -j 24 > "$root/timing_evidence/build_$variant.log" 2>&1
done
export OPENEDGE_BIN=$root/source_patched/tools/performance_review/profile_wrapper.py
export PYTHON_BIN=$(command -v python3)
export RUN_STEPS_OVERRIDE=5000 STATE_EVERY=5000 SURFACE_EVERY=5000 DENSITY_REPEAT=20
export DENSITY_EVERY=5000 RESTART_EVERY=5000 STATS_EVERY=1000
for ranks in 8 48; do
  export OE_PROFILE_WARM_RESTART=$root/baseline_np$ranks/teprobe_fixed_off/restart.final
  for rep in 1 2 3; do
    # Alternate ordering to reduce bias from temperature/frequency drift.
    order="baseline patched"
    [[ $rep != 2 ]] || order="patched baseline"
    for variant in $order; do
      src=$root/source
      [[ $variant != patched ]] || src=$root/source_patched
      export OPENEDGE_ROOT=$src OE_PROFILE_BINARY=$root/build_release_$variant/src/spa_perf
      export RUN_ROOT=$root/timing_${variant}_np${ranks}_rep${rep}
      export OE_PROFILE_DIR=$root/timing_evidence/${variant}_np${ranks}_rep${rep}
      cd "$src/examples/wip/slag/slag_mb_pilot"
      production/west_mb8_continuous_ops_20260922/run_case.sh teprobe_fixed_off "$ranks"
    done
  done
done
echo OE_TIMING_COMPLETE
