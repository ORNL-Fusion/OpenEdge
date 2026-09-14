#!/bin/bash
# usage: EXE_CPU=<cpu binary> EXE_GPU=<cuda kokkos binary> LAUNCH_CPU="srun -n 4" LAUNCH_GPU="srun -n 4 --gpus-per-task=1 ..." bash run.sh
set -u; cd "$(dirname "$0")"
: "${EXE_CPU:?}" "${EXE_GPU:?}"; LAUNCH_CPU=${LAUNCH_CPU:-mpirun -np 4}; LAUNCH_GPU=${LAUNCH_GPU:-mpirun -np 4}
rm -rf out_cpu out_gpu state.restart state_surf.restart state_gpu.restart; mkdir -p out_cpu out_gpu
$LAUNCH_CPU $EXE_CPU -in in.make_state -log log.make_state > screen.make_state 2>&1 || { echo "make_state failed"; exit 2; }
$LAUNCH_CPU $EXE_CPU -in in.parity -var tag cpu -log log.cpu > screen.cpu 2>&1 || { echo "cpu run failed"; exit 2; }
$LAUNCH_GPU $EXE_GPU -k on g 1 -sf kk -pk kokkos react/retry yes gpu/aware no comm threaded -in in.parity -var tag gpu -log log.gpu > screen.gpu 2>&1 || { echo "gpu run failed"; exit 2; }
grep -h -i 'fallback' screen.cpu screen.gpu | head -3
python3 compare.py || exit 1
# surface-tally check (compute surf/weighted on an absorbing target)
$LAUNCH_CPU $EXE_CPU -in in.make_state_surf -log log.make_state_surf > screen.make_state_surf 2>&1 || { echo "make_state_surf failed"; exit 2; }
rm -rf out_cpu out_gpu; mkdir -p out_cpu out_gpu
$LAUNCH_CPU $EXE_CPU -in in.parity_surf -var tag cpu -log log.cpu_surf > screen.cpu_surf 2>&1 || { echo "cpu surf run failed"; exit 2; }
$LAUNCH_GPU $EXE_GPU -k on g 1 -sf kk -pk kokkos react/retry yes gpu/aware no comm threaded -in in.parity_surf -var tag gpu -log log.gpu_surf > screen.gpu_surf 2>&1 || { echo "gpu surf run failed"; exit 2; }
grep -h -A2 'Kokkos host fallbacks' screen.gpu_surf | head -3
python3 compare_surf.py || exit 1
# population-control check (thinning on the frozen state)
rm -rf out_cpu out_gpu; mkdir -p out_cpu out_gpu
$LAUNCH_CPU $EXE_CPU -in in.parity_pop -var tag cpu -log log.cpu_pop > screen.cpu_pop 2>&1 || { echo "cpu pop run failed"; exit 2; }
$LAUNCH_GPU $EXE_GPU -k on g 1 -sf kk -pk kokkos react/retry yes gpu/aware no comm threaded -in in.parity_pop -var tag gpu -log log.gpu_pop > screen.gpu_pop 2>&1 || { echo "gpu pop run failed"; exit 2; }
grep -h -A2 'Kokkos host fallbacks' screen.gpu_pop | head -3
python3 compare_pop.py || exit 1
# GPU write_restart round trip (writer on the GPU, then both binaries read state_gpu.restart)
rm -rf out_cpu out_gpu; mkdir -p out_cpu out_gpu
$LAUNCH_GPU $EXE_GPU -k on g 1 -sf kk -pk kokkos react/retry yes gpu/aware no comm threaded -in in.parity_write -log log.gpu_write > screen.gpu_write 2>&1 || { echo "gpu write_restart run failed"; exit 2; }
$LAUNCH_CPU $EXE_CPU -in in.parity -var state state_gpu.restart -var tag cpu -log log.cpu_rst > screen.cpu_rst 2>&1 || { echo "cpu read of the GPU restart failed"; exit 2; }
$LAUNCH_GPU $EXE_GPU -k on g 1 -sf kk -pk kokkos react/retry yes gpu/aware no comm threaded -in in.parity -var state state_gpu.restart -var tag gpu -log log.gpu_rst > screen.gpu_rst 2>&1 || { echo "gpu read of the GPU restart failed"; exit 2; }
grep -h -A2 'Kokkos host fallbacks' screen.gpu_write | head -3
python3 compare_restart.py || exit 1
python3 compare.py || exit 1
