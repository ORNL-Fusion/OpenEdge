# gpu_parity — deterministic CPU vs GPU check of OpenEdge device paths

Stage 1 (`in.make_state`, CPU binary) creates 200 000 W ions in the rfpie box with
the rfpie raster plasma file (`rfpie_input` -> the rfpie case inputs) and writes
`state.restart`. Stage 2 (`in.parity`) is run once with the CPU binary and once
with the CUDA Kokkos binary from that same restart, with positions frozen
(`global move no`), so both paths see identical particles:

- step 0: every `compute grid/weighted` keyword (per-group and cell-level
  ratios/densities) through `fix ave/grid`, compared cell by cell (rel 1e-9);
- step 1: the `fix force/thermal` velocity kick (tilted constant B, raster
  gradients) compared particle by particle (rel 1e-6), and the particle set must
  be identical (a lost particle is a failure).

```bash
EXE_CPU=<cpu or kk-host binary> EXE_GPU=<spa_kokkos_cuda_*> \
LAUNCH_CPU="srun -n 4" LAUNCH_GPU="srun -n 4 --gpus-per-task=1" bash run.sh
```
`compare.py` prints one line per column and exits 1 on any failure.
History: this case found the Kokkos update loop ignoring `global move no`
(particles moved and left the open box on GPU) on 2026-09-12.
