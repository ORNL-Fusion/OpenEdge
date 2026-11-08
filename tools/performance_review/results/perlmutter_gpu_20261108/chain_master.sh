#!/bin/bash
export OE_JOBID=61003229
S=/tmp/claude-99462/-pscratch-sd-d-diawa/61bde2e5-a5f0-4c23-81ae-bc15afe47d89/scratchpad
L=$S/build_logs
until grep -q CHAIN_CPU_DONE $L/chain_cpu.log; do sleep 20; done
$S/run_one.sh cpu cpu48_100k 48 100000 > $L/run_cpu48_100k.log 2>&1
until grep -q GPU_BUILD_OK $L/gpu.log; do if grep -q -E 'Error|error:' $L/gpu.log && ! pgrep -f build_gpu.sh >/dev/null; then echo GPU_BUILD_FAILED; exit 1; fi; sleep 30; done
$S/run_one.sh gpu gpu4 4 20000 > $L/run_gpu4.log 2>&1
$S/run_one.sh gpu gpu1 1 20000 > $L/run_gpu1.log 2>&1
OE_NSYS_RANKS=0,1,2,3 $S/run_one.sh gpu-nsys gpu4_nsys 4 5000 > $L/run_gpu4_nsys.log 2>&1
$S/run_one.sh gpu-counts gpu4_counts 4 5000 > $L/run_gpu4_counts.log 2>&1
OE_NSYS_RANKS=0 $S/run_one.sh gpu-nsys gpu1_nsys 1 5000 > $L/run_gpu1_nsys.log 2>&1
$S/run_one.sh gpu gpu4_100k 4 100000 > $L/run_gpu4_100k.log 2>&1
echo CHAIN_MASTER_DONE
