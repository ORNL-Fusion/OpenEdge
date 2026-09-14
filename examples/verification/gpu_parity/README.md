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

Stage 3 (`in.make_state_surf` + `in.parity_surf`, run by the same `run.sh`): W ions
stream onto the rfpie target puck (absorbing) in a tilted constant B; `compute
surf/weighted` (all six keywords) through `fix ave/surf` is dumped every step and
compared surf by surf between the CPU and the GPU (rel 1e-9, `compare_surf.py`).

Stage 5 (`in.parity_write` + `in.parity` with `-var state state_gpu.restart`): the GPU
binary reads `state.restart`, moves the particles for 5 steps (Kokkos mover, rank
migration, thermal kicks), dumps them with their custom `pweight`, resets the step to 0
and writes `state_gpu.restart`. Both binaries then read that file: the readers' step-0
particle dumps must reproduce the writer's dump bit for bit (`compare_restart.py`: id
set, species, x, v, pweight) and the stage-2 grid/weighted + kick check must pass on
the GPU-written state (`compare.py`). This is the only test of `write_restart` from a
device-resident particle set.

```bash
EXE_CPU=<cpu or kk-host binary> EXE_GPU=<spa_kokkos_cuda_*> \
LAUNCH_CPU="srun -n 4" LAUNCH_GPU="srun -n 4 --gpus-per-task=1" bash run.sh
```
`compare.py` prints one line per column and exits 1 on any failure.
History: this case found the Kokkos update loop ignoring `global move no`
(particles moved and left the open box on GPU) on 2026-09-12.
