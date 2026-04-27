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
    submit_gpu.sbatch           Perlmutter A100 GPU production run
    submit_cpu_match.sbatch     CPU baseline matched to submit_gpu (4 ranks/node)
    submit_cpu_smoke.sbatch     small CPU smoke run (~5 min) — Phase D check
    submit_gpu_smoke.sbatch     small GPU smoke run (~5 min) — Phase D check
    collect_timings.py          parse SPARTA log timing -> CSV
    plot_scaling.py             strong-scaling speedup figure
    compare_outputs.py          diff CPU vs GPU grid-density dumps
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

# GPU production run:
sbatch submit_gpu.sbatch

# Phase D correctness pair (smoke, ~5 min each):
sbatch submit_cpu_smoke.sbatch    # writes output/grid.dens.cpu_smoke_4.west
sbatch submit_gpu_smoke.sbatch    # writes output/grid.dens.gpu_smoke_4.west

# Same problem, matched 4-rank layout (production-size, ~30 min):
sbatch submit_cpu_match.sbatch
sbatch submit_gpu.sbatch
```

All submit scripts have `<PROJECT>` placeholders --- fill in your
NERSC allocation ID before submission.

## Post-processing

Strong-scaling sweep:

```bash
python3 collect_timings.py logs/ -o results/timings.csv
python3 plot_scaling.py results/timings.csv -o results/strong_scaling.png
```

CPU vs GPU correctness diff (Phase D validation):

```bash
cd test_west_axi
python3 compare_outputs.py \
    output/grid.dens.cpu_smoke_4.west \
    output/grid.dens.gpu_smoke_4.west
```

The script reads the last frame of each grid-density dump, aligns by
cell id (handles MPI rank shuffling), and reports per-column relative
diff (RMS, max, p99) plus volume-integrated mass agreement. Default
tolerance is 5%% peak / 1%% RMS — loosen for small-sample stochastic
noise via `--tol`. Pass criterion is what we need to lock in before
calling Phase D done.

## Layout

All bench runs execute from `bench/openedge/test_west_axi/` itself so
the deck's relative paths resolve against a single, predictable cwd:

- `input/` is a symlink (auto-created on first sbatch) to
  `examples/test_west_axi/input/`. The deck (in `examples/`) is the
  single source of truth for `wall.surf`, `plasma.h5`, `*.species`.
- `output/` is bench-local — every dump (`grid.dens.<tag>.west`) lands
  here, so `compare_outputs.py` and analysis scripts read from one
  place regardless of which submit script wrote them.
- `logs/` and `results/` are bench-local for SLURM stdout / CSVs.

This keeps the `examples/test_west_axi/` tree clean of bench artefacts
and avoids the previous confusion where dumps from `submit_*.sbatch`
runs ended up scattered across both directories.
