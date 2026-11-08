#!/bin/bash
# Stack-sample rank 0 of a running GPU job on the allocated node every ~0.5 s for DUR seconds.
# usage: gdb_sample.sh DUR OUTFILE   (needs OE_JOBID)
dur=$1; out=$2
for v in $(env | grep -o '^SLURM_[A-Za-z_]*'); do unset $v; done
export SLURM_JOB_ID=$OE_JOBID SLURM_JOBID=$OE_JOBID
srun --jobid $OE_JOBID --overlap -N1 -n1 bash -c '
  pid=""; for i in $(seq 1 60); do pid=$(pgrep -u $USER -f "spa_kokkos_cuda_perlmutter" | sort -n | head -1); [ -n "$pid" ] && break; sleep 1; done
  echo "sampling pid $pid"
  end=$((SECONDS+'"$dur"'))
  while [ $SECONDS -lt $end ]; do
    echo "=== SAMPLE $(date +%s.%N)"
    gdb -p $pid -batch -ex "bt 25" 2>/dev/null | grep -E "^#" | head -25
    sleep 0.3
  done' > $out 2>&1
echo SAMPLING_DONE
