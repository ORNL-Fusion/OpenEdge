# Production pair: saturated z-ramped rfpie case (2026-09-12)

Same deck (`../in.openedge`), inputs and variables for both runs:
`-var plasmafile input/plasma_he_zramp.h5 -var nlaunch 5000 -var popnmax 5000 -var nx 128 -var ny 128 -var nz 192 -var stateevery 10000 -var diagevery 2000 -var dumpevery 10000`
(10 000 steps of 20 ns, 128 x 128 x 192 grid = 3.1 M cells, ~4.47 M markers in flight at the end).

| Run | Machine | OpenEdge commit | Binary | Main loop | First 1000 steps | Last 9000 steps |
|---|---|---|---|---|---|---|
| `gpu_zramp_hi` | Perlmutter, 1 GPU node (4 x A100, 4 MPI ranks) | 1258901b (gpu branch) | `spa_kokkos_cuda_perlmutter` (`-k on g 1 -sf kk -pk kokkos react/retry yes gpu/aware no comm threaded`) | 160 s | 31 s | 129 s (313 Mparticle-steps/s) |
| `cpu_zramp_hi` | Perlmutter, 1 CPU node (128 MPI ranks) | d7ffc5af (gpu branch) | `spa_kokkos_omp`, pure CPU path (no `-k on`) | 309 s | 25 s | 284 s (142 Mparticle-steps/s) |

Synthetic W I 498 nm profiles (chord / axis / density) of the two runs agree within 0.008 peak-normalized
(`wi498_profiles.csv`, plotted in `wi498_gpu_cpu_experiment.png` with the digitized experiment).
The full runs (density dumps, 1.8 GB each) are on NERSC scratch:
`/pscratch/sd/d/diawa/openedge-runs/rfpie_fix/{gpu_zramp_hi,cpu_zramp_hi}`.
Screens and logs of both runs are in `gpu_zramp_hi/` and `cpu_zramp_hi/` here.
