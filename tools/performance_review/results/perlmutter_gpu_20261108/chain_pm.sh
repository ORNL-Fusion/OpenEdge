#!/bin/bash -l
export OE_JOBID=61089693
S=/pscratch/sd/d/diawa/oe-perf-tools; L=$S/build_logs
until grep -q REBUILD_DONE $L/rebuild_pm.log; do sleep 15; done
grep -q 'GPU_REBUILD_RC=0' $L/rebuild_pm.log || { echo GPU_REBUILD_FAILED; exit 1; }
cp $PSCRATCH/openedge-build-a508aff-gpu/src/spa_kokkos_cuda_perlmutter $S/spa_kokkos_cuda_item7_wip
for r in 1 2 3; do
  OE_SKIP_POST_MIGRATE=0 $S/run_one.sh gpu b11_gpu1_r$r 1 20000 > $L/run_b11_gpu1_r$r.log 2>&1
  OE_SKIP_POST_MIGRATE=1 $S/run_one.sh gpu i11_gpu1_r$r 1 20000 > $L/run_i11_gpu1_r$r.log 2>&1
  OE_SKIP_POST_MIGRATE=0 $S/run_one.sh gpu b11_gpu4_r$r 4 20000 > $L/run_b11_gpu4_r$r.log 2>&1
  OE_SKIP_POST_MIGRATE=1 $S/run_one.sh gpu i11_gpu4_r$r 4 20000 > $L/run_i11_gpu4_r$r.log 2>&1
done
echo RUNS_I11_DONE
