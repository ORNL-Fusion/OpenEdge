# OpenEdge benchmarks

Performance benchmarks for OpenEdge. The legacy `bench/` directory at the
repo root carries the SPARTA-stock benchmarks (`in.sphere`, `in.collide`,
`in.free`); this subtree adds OpenEdge-specific cases that exercise the
full plasma-edge physics stack.

## Layout

```
bench/openedge/
  README.md                     this file
  GPU_PORT_STATUS.md            Kokkos coverage table per fix/compute
  test_west_axi/
    submit_strong.sbatch        CPU strong-scaling SLURM array
    submit_gpu.sbatch           Perlmutter A100 GPU run
    collect_timings.py          parse SPARTA log timing -> CSV
    plot_scaling.py             strong-scaling speedup figure
    logs/                       SLURM stdout/stderr (gitignored)
    results/                    CSV + plots (gitignored)
```

The bench decks themselves are not duplicated. They reuse
`examples/test_west_axi/in.west` (and `examples/test_diii_d_neutrals/in.diii_d_neutrals`
in a future addition). The submit scripts call those decks with
`-var` overrides for run length, particle count, and adapt frequency.

## What test_west_axi exercises

Full impurity-transport stack:

- Boris pusher with sheath kick (in update.cpp / sheath_models.cpp).
- `fix force/thermal` --- Braginskii alpha,beta forces.
- `fix cross_field_diffusion` --- anomalous D_perp + pinch.
- `fix coulomb/binary, coulomb/background` --- Nanbu binary scattering.
- `surf_react surface/pwi` --- TRIM reflection + absorb-reemit.
- `compute surface/physical/sputter` --- Eckstein analytic yield (Kokkos-ready).
- `fix surface/emit/source` --- W sputter emission.
- `fix particle/weight` --- variable per-particle weight bookkeeping.
- `fix background` --- mesh plasma + B-field.

## What we measure

- **Strong scaling**: fixed problem (`nlaunchTotalW=10`, `Nphase=30000`),
  vary `-np` across {16, 32, 64, 128, 256, 512, 1024}. Speedup vs. ranks.
- **GPU vs CPU**: same problem on Perlmutter A100 nodes (4 GPUs each)
  vs. AMD Milan CPU nodes (128 cores each). Per `GPU_PORT_STATUS.md`,
  most OpenEdge fixes are CPU-only today --- a GPU run currently shuttles
  particles between host and device every step. The benchmark serves
  partly as a *measurement* of how much that mixed-execution penalty
  costs, which guides which fix to port next.

## Running on Perlmutter

```bash
# Build (already validated; see CLAUDE.md):
module load PrgEnv-gnu cmake cray-hdf5
mkdir build && cd build
cmake -C ../OpenEdge/cmake/presets/mpi.cmake ../OpenEdge/cmake/ \
      -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
      -DPKG_OPENEDGE=ON
make -j$(nproc)

# CPU strong-scaling sweep:
cd ../OpenEdge/bench/openedge/test_west_axi
sbatch submit_strong.sbatch       # one SLURM array

# GPU run:
sbatch submit_gpu.sbatch
```

Both submit scripts have `<PROJECT>` placeholders --- fill in your
NERSC allocation ID before submission.

## Post-processing

```bash
python3 collect_timings.py logs/ -o results/timings.csv
python3 plot_scaling.py results/timings.csv -o results/strong_scaling.png
```
