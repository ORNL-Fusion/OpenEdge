#!/bin/bash
#SBATCH -J oe-perf-audit
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
export OMPI_MCA_btl_vader_single_copy_mechanism=none UCX_TLS=sm,self HWLOC_COMPONENTS=-gl
export OMP_NUM_THREADS=1
root=${OE_REVIEW_ROOT:-/scratch/home_eos/42d/oe_perf_review_20261107}
src=$root/source
export OPENEDGE_ROOT=$src
python3 "$src/tools/performance_review/build_audit.py" "$src" "$root/build_gprof" \
  > "$root/evidence/audit_build.log" 2>&1
cd "$src/examples/wip/slag/slag_mb_pilot"
python3 - "$root" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
deck=(root/'evidence/np8/profile.in').read_text()
# Keep the full initialized setup, omit the advancing schedule and final restart.
deck=deck.split('include '+str(root/'evidence/np8/schedule.inc'))[0]
(root/'evidence/audit.in').write_text(deck)
PY
out=$root/audit_output
mkdir -p "$out"
export OE_AUDIT_CSV=$root/evidence/background_accuracy.csv
export GMON_OUT_PREFIX=$root/evidence/audit_gmon
mpirun -np 1 "$root/build_gprof/src/background_audit" -in "$root/evidence/audit.in" \
  -var outdir "$out" -var sweepwork "$root/baseline_np8/teprobe_fixed_off/sweep_runtime" \
  -var runsteps 20000 -var external_injection 0 -var dt 1e-8 -var dperp 0.1 \
  -var source_rate 1e17 -var stateevery 20000 -var surfaceevery 20000 \
  -var densitysample 250 -var densityrepeat 80 -var densityevery 20000 \
  -var restartevery 20000 -var statsevery 1000 -log none > "$root/evidence/audit.log" 2>&1
unset GMON_OUT_PREFIX
cmake -S "$src/cmake" -B "$root/build_kkhost" -DPKG_OPENEDGE=ON -DPKG_KOKKOS=ON \
  -DKokkos_ENABLE_OPENMP=ON -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
  -DPKG_VTK=ON -DVTK_DIR=/scratch/home_eos/42d/sw/vtk-9.3.1-minimal-gcc15/lib64/cmake/vtk-9.3 \
  -DSPARTA_MACHINE=kkhost -DCMAKE_BUILD_TYPE=Release > "$root/evidence/kk_configure.log" 2>&1
cmake --build "$root/build_kkhost" -j 24 > "$root/evidence/kk_build.log" 2>&1
g++ -shared -fPIC -O2 -std=c++17 -I "$src/lib/kokkos/core/src" \
  "$src/tools/performance_review/kokkos_counts.cpp" -o "$root/counts.so"
export OE_PROFILE_BINARY=$root/build_kkhost/src/spa_kkhost OE_KK_PROFILE=1
export OPENEDGE_BIN=$src/tools/performance_review/profile_wrapper.py PYTHON_BIN=$(command -v python3)
export RUN_STEPS_OVERRIDE=20000 STATE_EVERY=20000 SURFACE_EVERY=20000
export DENSITY_REPEAT=80 DENSITY_EVERY=20000 RESTART_EVERY=20000 STATS_EVERY=1000
export RUN_ROOT=$root/kkhost_np8 OE_PROFILE_DIR=$root/evidence/kkhost_np8
mkdir -p "$OE_PROFILE_DIR"
export OE_KOKKOS_COUNTS_DIR=$OE_PROFILE_DIR KOKKOS_TOOLS_LIBS=$root/counts.so
production/west_mb8_continuous_ops_20260922/run_case.sh teprobe_fixed_off 8
echo OE_AUDIT_COMPLETE
