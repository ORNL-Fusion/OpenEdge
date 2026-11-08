#!/bin/bash
#SBATCH -J oe-perf-patch
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
src=$root/source_patched
ev=$root/patched_evidence
mkdir -p "$ev"
cmake -S "$src/cmake" -B "$root/build_patched" -DPKG_OPENEDGE=ON -DPKG_VTK=ON \
  -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
  -DVTK_DIR=/scratch/home_eos/42d/sw/vtk-9.3.1-minimal-gcc15/lib64/cmake/vtk-9.3 \
  -DBUILD_MPI=ON -DSPARTA_MACHINE=perf -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO='-O3 -g -pg -DNDEBUG' -DCMAKE_EXE_LINKER_FLAGS=-pg \
  > "$ev/configure.log" 2>&1
cmake --build "$root/build_patched" -j 24 > "$ev/build.log" 2>&1
export OPENEDGE_ROOT=$src PYTHON_BIN=$(command -v python3)
export OE_PROFILE_BINARY=$root/build_patched/src/spa_perf
export OPENEDGE_BIN=$src/tools/performance_review/profile_wrapper.py
export RUN_STEPS_OVERRIDE=20000 STATE_EVERY=20000 SURFACE_EVERY=20000 DENSITY_REPEAT=80
export DENSITY_EVERY=20000 RESTART_EVERY=20000 STATS_EVERY=1000
cd "$src/examples/wip/slag/slag_mb_pilot"
for ranks in 8 48; do
  export RUN_ROOT=$root/patched_np$ranks OE_PROFILE_DIR=$ev/np$ranks
  mkdir -p "$OE_PROFILE_DIR"
  export GMON_OUT_PREFIX=$OE_PROFILE_DIR/gmon
  production/west_mb8_continuous_ops_20260922/run_case.sh teprobe_fixed_off "$ranks"
  gprof -b "$OE_PROFILE_BINARY" "$OE_PROFILE_DIR"/gmon.* > "$OE_PROFILE_DIR/gprof.txt"
done
unset GMON_OUT_PREFIX
cd "$src"
bash regression/run_regression.sh --np 4 --exe "$OE_PROFILE_BINARY" --verbose \
  > "$ev/regression_patched.txt" 2>&1 || echo regression_failed
find examples -name 'regression_metrics_*_cpu.json' -exec cp --parents '{}' "$ev/" \;
bash tools/check_package_parity.sh > "$ev/parity_patched.txt" 2>&1 || true
for ranks in 8 48; do
  python3 tools/performance_review/compare_west.py "$root/baseline_np$ranks/teprobe_fixed_off" \
    "$root/patched_np$ranks/teprobe_fixed_off" > "$ev/west_comparison_np$ranks.json"
done
echo OE_PATCH_COMPLETE
