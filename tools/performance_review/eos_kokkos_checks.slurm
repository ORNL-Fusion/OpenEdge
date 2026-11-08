#!/bin/bash
#SBATCH -J oe-kk-checks
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --exclude=eos01
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 00:40:00
set -euo pipefail
module purge
module load gcc/15.2.0 openmpi/4.1.7-gcc15.2.0 cmake hdf5/1.14.5_cpp python/3.12.3
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=vader,self
export OMPI_MCA_btl_vader_single_copy_mechanism=none UCX_TLS=sm,self HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=1
root=${OE_REVIEW_ROOT:-/scratch/home_eos/42d/oe_perf_review_20261107}
# The profiled executable is complete before this job starts. Install the
# validated CPU query shortcut into its source and rebuild the host backend.
cp "$root/source_patched/src/fix_background.cpp" "$root/source/src/fix_background.cpp"
cp "$root/source_patched/src/OPENEDGE/fix_background.cpp" "$root/source/src/OPENEDGE/fix_background.cpp"
cmake --build "$root/build_kkhost" -j 24 > "$root/evidence/kk_checks_build.log" 2>&1
cd "$root/source"
export PYTHON=$(command -v python3)
bash regression/run_regression.sh --kk --np 4 --exe "$root/build_kkhost/src/spa_kkhost" \
  --verbose \
  > "$root/evidence/regression_kkhost.txt" 2>&1
# The parity runner hardcodes CUDA -k on g; use a private fixture copy
# with host-backend flags to exercise the same comparisons on EOS.
cp -aL examples/verification/kokkos_parity "$root/parity_host_fixture"
python3 - "$root/parity_host_fixture/run.sh" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
p.write_text('\n'.join('KK="-k on t 1 -sf kk -pk kokkos react/retry yes comm threaded"'
                        if line.startswith('KK=') else line
                        for line in p.read_text().splitlines())+'\n')
PY
export EXE_CPU=$root/build_release_baseline/src/spa_perf EXE_GPU=$root/build_kkhost/src/spa_kkhost
bash "$root/parity_host_fixture/run.sh" > "$root/evidence/parity_kkhost.txt" 2>&1
mkdir -p "$root/parity_snapshot/src" "$root/parity_snapshot/tools"
rsync -a --exclude='fix_diag_parallel_noise.cpp' --exclude='fix_diag_parallel_noise.h' \
  "$root/source_patched/src/" "$root/parity_snapshot/src/"
cp "$root/source/tools/check_package_parity.sh" "$root/parity_snapshot/tools/"
bash "$root/parity_snapshot/tools/check_package_parity.sh" --install > "$root/evidence/parity_tracked_install.txt" 2>&1
echo OE_KOKKOS_CHECKS_PASS
