#!/bin/bash
#SBATCH -J oe-perf-repeat
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --exclude=eos01
#SBATCH -n 48
#SBATCH --mem=96G
#SBATCH -t 00:30:00
set -euo pipefail
module purge
module load gcc/15.2.0 openmpi/4.1.7-gcc15.2.0 cmake hdf5/1.14.5_cpp python/3.12.3
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=vader,self
export OMPI_MCA_btl_vader_single_copy_mechanism=none UCX_TLS=sm,self HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=1
root=${OE_REVIEW_ROOT:-/scratch/home_eos/42d/oe_perf_review_20261107}
export OPENEDGE_ROOT=$root/source PYTHON_BIN=$(command -v python3)
export OE_PROFILE_BINARY=$root/build_gprof/src/spa_perf
export OPENEDGE_BIN=$root/source_patched/tools/performance_review/profile_wrapper.py
export RUN_ROOT=$root/baseline_repeat_np48 OE_PROFILE_DIR=$root/evidence/repeat_np48
export GMON_OUT_PREFIX=$OE_PROFILE_DIR/gmon
export RUN_STEPS_OVERRIDE=20000 STATE_EVERY=20000 SURFACE_EVERY=20000 DENSITY_REPEAT=80
export DENSITY_EVERY=20000 RESTART_EVERY=20000 STATS_EVERY=1000
cd "$root/source/examples/wip/slag/slag_mb_pilot"
production/west_mb8_continuous_ops_20260922/run_case.sh teprobe_fixed_off 48
python3 "$root/source_patched/tools/performance_review/compare_west.py" \
  "$root/baseline_np48/teprobe_fixed_off" "$root/baseline_repeat_np48/teprobe_fixed_off" \
  > "$root/evidence/west_baseline_repeat_np48.json"
echo OE_REPEAT_COMPLETE
