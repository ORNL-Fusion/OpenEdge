#!/bin/bash
# usage: finish_table.sh TAG [every]  -> loop time, timing breakdown, per-1000-step series
t=$1; ev=${2:-4}; R=$PSCRATCH/openedge-runs/perf_review_20261108/$t/prevention_fixed_on
echo "== $t"; grep -E 'Loop time|Performance' $R/run.log | tail -2
grep -A14 'MPI task timing' $R/run.log | tail -12
grep -E 'Particle moves|Particles: |lost at periodic caps|Particle comm iterations' $R/run.log | tail -4
echo "step cpu_s np | s_per_1000"
awk '/^Step CPU Np/{f=1;p=0;next} f&&/^ *[0-9]+ /{printf "%6d %9.2f %6d | %6.2f\n",$1,$2,$3,$2-p; p=$2} /^Loop time/{f=0}' $R/run.log | awk -v e=$ev 'NR==1||NR%e==0'
