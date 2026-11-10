#!/bin/bash -l
export OE_JOBID=61089693
S=/pscratch/sd/d/diawa/oe-perf-tools; L=$S/build_logs
until grep -q -E 'RUNS_I11_DONE|FAILED' $L/chain_pm.log; do sleep 20; done
grep -q RUNS_I11_DONE $L/chain_pm.log || { echo CHAIN_PM_FAILED; exit 1; }
( for v in $(env | grep -o '^SLURM_[A-Za-z_]*'); do unset $v; done
  export SLURM_JOB_ID=$OE_JOBID SLURM_JOBID=$OE_JOBID
  export EXE=$PSCRATCH/openedge-build-a508aff-gpu/src/spa_kokkos_cuda_perlmutter
  source $HOME/OpenEdge/scripts/perlmutter_gpu_env.sh; export OE_KK_NGPU=4
  cd $HOME/OpenEdge
  ./regression/run_regression.sh --kk --exe $EXE --parity-cpu-exe $PSCRATCH/openedge-build-a508aff-cpu/src/spa_mpi \
     --launcher "srun --jobid=$OE_JOBID -N1 --ntasks-per-node=4 $OE_SRUN_GPU --overlap -n 4 --cpu-bind=cores" > $L/regression_gpu_i11.log 2>&1
  echo "GPU_REGRESSION_I11_EXIT $?" )
echo CHAIN_REG11_DONE
