# RFPIE tungsten sputtering and transport

He plasma on a biased W target in the RFPIE geometry, built from the RFPIE
CAD and Langmuir-probe profiles.

- Background: measured He radial profiles (density shape-preserving, Te an
  even-polynomial fit), quasineutral He+, uniform in z.
- Source: He-on-W RustBCA yield from `database/processes.h5`, Thompson W
  emission (Us = 8.68 eV, Emax = 80 eV).
- Target: one watertight surface; the plasma-facing top is retiled into
  radial rings so wall-flux sampling resolves the peaked profile. Each tile
  carries `sheath = [Vdc, Vrf_peak, phase]`. The sheath is a thin potential
  sheet, not a resolved Debye sheath.
- Transport: W to W4+ with ADAS charge-state evolution; pweight-aware
  interval-averaged density per charge state in `output/rfpie_w_density.*.dump`.
- Sampling: 64 x 64 x 96 cells, 100 markers per step, roulette at 200
  markers per cell and species, two-stage RCB balancing.

The chamber STL is kept for plotting only; the deck uses the open Cartesian
box as its outer boundary.

## Files

| File | Purpose |
|---|---|
| `in.openedge` | Deck; defaults Vdc = -500 V, Vrf = 0, 64 phase samples |
| `input/config.json` | Geometry and voltage parameters |
| `input/target_face.surf`, `input/target.surf` | Retiled target surfaces |
| `input/plasma_he.h5` | Reconstructed He background |
| `scripts/build_geometry.py`, `build_plasma.py` | Rebuild the inputs from CAD and probe data |
| `scripts/check_case.py` | Case consistency check |
| `scripts/analysis.ipynb` | Plasma, geometry, yield and W density diagnostics |
| `scripts/build_he_on_w.py`, `install_he_on_w.py` | Regenerate the He-on-W yield table in `processes.h5` |

## Run

```bash
mpirun -np 4 /path/to/spa_mpi -in in.openedge
python scripts/check_case.py
jupyter nbconvert --to notebook --execute --inplace scripts/analysis.ipynb
```

Defaults: dt = 20 ns, 10 000 steps (200 us), about 16 plume transit
times. Changing frequency: edit `input/config.json`, rebuild
`target_face.surf`, and keep `rffreq` consistent in both input files. For
time-resolved RF at 13.56 MHz use one phase sample, disable the PMI cache,
and set dt to about 2 ns. The DC case is the cleaner starting point.

## W I 498 nm comparison with experiment (added 2026-09-11)

| File | Purpose |
|---|---|
| `scripts/wi498_compare.py` | Peak-normalized W I 498 nm axial profile (chord / axis / density) from the density dumps, ADAS PEC weighting, Curt-style plot; used by notebook section 8 |
| `scripts/build_plasma_zramp.py` | z-ramped background variant (`input/plasma_he_zramp.h5`): LP density at the target face rising to `1/f0` × aloft with e-folding `L` (defaults 0.5, 3 mm) |
| `input/experiment_wi498_axial_digitized.csv` | Experiment profile digitized from the reference plot (replace with measured data) |

Run the ramped case with `-var plasmafile input/plasma_he_zramp.h5`; higher statistics /
finer grid via `-var nlaunch 5000 -var popnmax 5000 -var nx 128 -var ny 128 -var nz 192`
(≈4.5 M markers). Production pair (2026-09-12, GPU at commit 1258901b, CPU at d7ffc5af, identical
settings): `$PSCRATCH/openedge-runs/rfpie_fix/{gpu_zramp_hi,cpu_zramp_hi}` — one GPU node
(4 × A100) vs one CPU node (128 MPI ranks); profiles identical (max diff 0.008 peak-normalized);
main loop 160 s vs 309 s (1.9× on wall), steady state 129 s vs 284 s for the last 9000 steps
(2.2×, 313 vs 142 Mparticle-steps/s). The GPU's first 1000 steps take 31 s (were 110 s before
1258901b): the first `fix balance` moves ~600 k cells per rank and the per-cell arrays of
`fix ave/grid` grew 1024 cells at a time with a full host/device round trip per grow (O(n²));
they now grow by 25 % at a time and the migration pre-grows the cell arrays once. The W I PEC used for the synthetic emission is the
open-ADAS `pec40#w_ic#w0.dat` line at 5000.73 Å (theoretical), the closest to the 498 nm
line; `compute volume/emissivity/grid` reproduces the post-processed emission exactly when the
table is stored in the loader's convention (log10 coefficient on log10 Te × log10 ne[m^-3]).

**GPU note (2026-09-11):** `nlaunch_total >= 2000` on the Kokkos builds crashed at the first
capacity crossing (device emission appended past the particle array because the pre-grow
passed a deficit to `grow(nextra)`); fixed in commit d7ffc5af. With the fix the saturated
settings above run on 4 A100s.

## Folder layout and how to reproduce the GPU / CPU pair

| Path | What |
|---|---|
| `in.openedge`, `input/` | Deck and inputs (`input/plasma_he_zramp.h5` is generated: `python scripts/build_plasma_zramp.py`) |
| `scripts/run_gpu_perlmutter.sbatch` | Saturated z-ramped case on one GPU node (4 × A100); writes `$RUNDIR/gpu_zramp_hi` |
| `scripts/run_cpu_perlmutter.sbatch` | Same deck and settings on one CPU node (128 MPI ranks); writes `$RUNDIR/cpu_zramp_hi` |
| `runs/{gpu,cpu}_zramp_hi` | Symlinks to the run directories on scratch (made by the sbatch scripts; not in git) |
| `results/{gpu,cpu}_zramp_hi/` | Run screens and logs of the production pair (timings, settings) |
| `results/wi498_gpu_cpu_experiment.png` | The GPU / CPU / experiment figure |
| `results/wi498_profiles.csv`, `results/PROVENANCE.md` | Peak-normalized profiles of the pair (the notebook plots these when the dumps are absent) and where/how they were run |
| `input/w_i_pec_adf15_pec40_w_ic_w0_isel49_5000p73A.npz` | Open-ADAS W I PEC used for the synthetic emission |
| `scripts/wi498_compare.py` | Synthetic W I 498 nm profiles from the density dumps (`python scripts/wi498_compare.py label=runs/gpu_zramp_hi/output ...`) |
| `scripts/analysis.ipynb` | Full analysis; sections 8 and 9 read `runs/` (falls back to scratch, then `results/`) |

Prerequisites: an OpenEdge build (GPU: Kokkos/CUDA; CPU: any), `OPENEDGE_ROOT` pointing at a
tree whose `database/processes.h5` holds the ADAS W tables (see `database/ingest/`), and Python
with numpy/scipy/h5py/matplotlib for the analysis. The sbatch scripts are written for NERSC
Perlmutter (modules, account, queue); on another machine edit the `#SBATCH` lines and the module
block, and set `EXE`, `RUNDIR` as needed. The runs land in `$RUNDIR` (default
`$PSCRATCH/openedge-runs/rfpie_fix`) and `runs/` in this folder links to them.

```bash
cd examples/workflows/impurity_transport/rfpie_tungsten_transport
python scripts/build_plasma_zramp.py                 # once: input/plasma_he_zramp.h5
sbatch scripts/run_gpu_perlmutter.sbatch             # RUNDIR=... EXE=... to override defaults
sbatch scripts/run_cpu_perlmutter.sbatch
python scripts/wi498_compare.py "GPU=runs/gpu_zramp_hi/output" "CPU=runs/cpu_zramp_hi/output"
jupyter nbconvert --to notebook --execute --inplace scripts/analysis.ipynb
```
