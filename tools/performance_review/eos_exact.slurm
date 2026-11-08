#!/bin/bash
#SBATCH -J oe-query-exact
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --exclude=eos01
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 00:20:00
set -euo pipefail
module purge
module load gcc/15.2.0 openmpi/4.1.7-gcc15.2.0 cmake hdf5/1.14.5_cpp python/3.12.3
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=vader,self HWLOC_COMPONENTS=-gl OMP_NUM_THREADS=1
root=${OE_REVIEW_ROOT:-/scratch/home_eos/42d/oe_perf_review_20261107}
for variant in baseline patched; do
  build=$root/build_gprof
  [[ $variant != patched ]] || build=$root/build_patched
  python3 "$root/source_patched/tools/performance_review/build_audit.py" \
    "$root/source_patched" "$build" query_fingerprint > "$root/evidence/query_build_$variant.log" 2>&1
  export OPENEDGE_ROOT=$root/source
  export OE_FINGERPRINT_OUTPUT=$root/evidence/query_fingerprint_$variant.txt
  export GMON_OUT_PREFIX=$root/evidence/query_gmon_$variant
  out=$root/query_$variant
  mkdir -p "$out"
  cd "$root/source/examples/wip/slag/slag_mb_pilot"
  mpirun -np 1 "$build/src/query_fingerprint" -in "$root/evidence/audit.in" \
    -var outdir "$out" -var sweepwork "$root/baseline_np8/teprobe_fixed_off/sweep_runtime" \
    -var runsteps 20000 -var external_injection 0 -var dt 1e-8 -var dperp 0.1 \
    -var source_rate 1e17 -var stateevery 20000 -var surfaceevery 20000 \
    -var densitysample 250 -var densityrepeat 80 -var densityevery 20000 \
    -var restartevery 20000 -var statsevery 1000 -log none > "$root/evidence/query_$variant.log" 2>&1
done
diff "$root/evidence/query_fingerprint_baseline.txt" "$root/evidence/query_fingerprint_patched.txt"
echo OE_QUERY_EXACT_PASS
