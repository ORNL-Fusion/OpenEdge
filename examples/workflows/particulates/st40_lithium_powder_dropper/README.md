# ST40 lithium powder dropper

Axisymmetric injection of the measured shot-14449 Li powder source into a
static SOLPS-13589 background. Grains charge, feel plasma drag and gravity,
ablate into Li, and feed Li charge-state transport. Source timing follows
shot 14449 while the background is shot 13589, so the case supports
qualitative plume and trajectory comparison, not frame-by-frame comparison
with the Photron video.

## Layout

| Path | Purpose |
|---|---|
| `in.openedge` | Canonical deck; `st40_core.surf` absorbs grains and Li at the inner mesh edge (core-loss control) |
| `in.core_transit` | Grains cross the core contour; Li atoms and ions are absorbed there |
| `in.core_transit_cohort`, `in.core_transit_ballistic` | Deterministic 30-grain transit smoke and its gravity-only oracle |
| `in.core_transit_size_scan` | Deterministic 40 to 300 um diameter response and timestep scan |
| `in.core_transit_background_probe` | Zero-step audit of Te/ne/q on the run grid |
| `input/*.inc` | Setup, physics and diagnostics includes per deck |
| `input/plasma_st40_solps_corefill.h5` | Runtime plasma: native SOLPS-13589 datasets with `q_par/q_perp`, plus a labelled bounded continuation across the central mesh hole (`core_fill_mask`) |
| `scripts/build_*.py` | Rebuild geometry, source deck, core-fill background, cohort and size-scan inputs, and the review notebook |
| `scripts/analyze_core_transit_*.py` | Machine checks: trajectories, fates, gravity oracle, Li closure, size response, heat-scan fates |
| `scripts/run_core_transit_heat_scan.sh` | 0 to 2x OML-heating bracket |
| `scripts/check_equilibrium_orientation.py` | Fails on a transposed equilibrium |
| `scripts/plot_lpd.py`, `scripts/analysis.ipynb` | Plume plots and coupled-plume analysis |
| `notebooks/st40_core_transit_setup_and_smoke.ipynb` | Executed setup and smoke review |
| `verification/core_species_boundary/` | Two-particle regression: Li absorbed, grain transparent at the same psi contour |

## Run

```bash
mpirun -np 4 /path/to/spa_mpi -in in.openedge                      # 0.5 s slice
mpirun -np 4 /path/to/spa_mpi -var nsteps 2500000 -in in.openedge  # full 5 s history
```

## Core-transit variant

`in.core_transit` removes the artificial core surface and uses

```text
fix fcore reflect/psi background pd psi_norm 0.51 action absorb mixture allLi
```

Li through Li3+ keep a core-loss boundary; grains are not in `allLi`, cross
the contour, and keep heating and ablating. `output_core_transit/core_flux`
holds the species-resolved sink ledger: cumulative marker events, cumulative
physical particles from `pweight`, and mean removal rate. Difference the
cumulative `pweight` columns over a time window for a core-code source rate.

Deterministic decks use `global weight cell none`, since axisymmetric cell
weights clone and delete markers and cannot keep one launch ID per grain.
The size scan gives each diameter its own species because `read_particles`
takes radius, mass and temperature from the species. It is a diameter
response map, not a measured size distribution.

```bash
(cd verification/core_species_boundary && NP=2 ./run.sh)
python scripts/build_core_transit_cohort.py
mpirun -np 2 $SPA -in in.core_transit_ballistic
mpirun -np 2 $SPA -in in.core_transit_cohort
python scripts/analyze_core_transit_smoke.py
python scripts/build_core_transit_size_scan.py
mpirun -np 2 $SPA -in in.core_transit_size_scan
mpirun -np 2 $SPA -var dt 1e-6 -var nsteps 750000 -var ndump 1000 \
  -var outdir output_core_transit_size_scan_dt1us -in in.core_transit_size_scan
python scripts/analyze_core_transit_size_scan.py
mpirun -np 2 $SPA -in in.core_transit_background_probe
python scripts/analyze_core_transit_maps.py
NP=2 OPENEDGE_BIN=$SPA scripts/run_core_transit_heat_scan.sh
python scripts/analyze_core_transit_heat_scan.py
python scripts/build_core_transit_notebook.py
```

The heat scan changes only the OML surface-heating term (`heatflux/scale`).
Core penetration appears only below 0.25x heating (16 of 270 grains at zero
heating, none at 0.25x and above), and most grains hit the upper or side
wall at any heating. The core fill is a numerical extension and needs the
vacuum and hot/dense background brackets before a firm bottom-arrival claim.

## Regenerating the background

```bash
python scripts/restore_st40_heatflux.py --source <run_dir_converted_with_q.h5> \
  --target /tmp/plasma_st40_solps_corrected.h5
python scripts/build_core_transit_background.py --base /tmp/plasma_st40_solps_corrected.h5
```
