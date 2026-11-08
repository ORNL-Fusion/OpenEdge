#!/bin/bash
#SBATCH -J oe-final-checks
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
cd "$root/source"
export PYTHON=$(command -v python3)
cmake --build "$root/build_kkhost" -j 16 > "$root/evidence/final_kk_build.log" 2>&1
bash regression/run_regression.sh --np 4 --exe "$root/build_release_baseline/src/spa_perf" --verbose \
  > "$root/evidence/regression_final_cpu.txt" 2>&1
bash regression/run_regression.sh --kk --np 4 --exe "$root/build_kkhost/src/spa_kkhost" --verbose \
  > "$root/evidence/regression_final_kkhost.txt" 2>&1
cp src/fix_background.cpp "$root/parity_snapshot/src/fix_background.cpp"
cp src/OPENEDGE/fix_background.cpp "$root/parity_snapshot/src/OPENEDGE/fix_background.cpp"
bash "$root/parity_snapshot/tools/check_package_parity.sh" --install > "$root/evidence/parity_final_tracked_install.txt" 2>&1
echo OE_FINAL_CHECKS_PASS
