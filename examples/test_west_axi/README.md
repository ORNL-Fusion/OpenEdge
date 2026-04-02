# WEST test case (OpenEdge)

**Data download required:** `./download_data.sh test_west_axi` (from repo root)

Primary input:
- `in.west`

Current workflow in `in.west`:
1. Load WEST wall surface (`input/wall.txt`).
2. Apply static `adapt_grid` refinement near lower divertor, upper divertor, and limiter.
3. Compute plasma/background fields with `compute plasma/fields`.
4. Compute sheath geometry/fields with `compute sheath/geometry/grid` and `compute sheath/fields/grid`.
5. Apply static B and sheath E fields with `fix bfield/grid` and `fix efield/grid`.
6. Compute ADAS chemistry with sheath-corrected electron density (`fix chem/adas`).
7. Compute incident plasma flux with `compute incident/plasma/flux`.
8. Compute sputter source with `compute pmi/surf/data` (`sputter_flux_total`).
9. Emit sputtered particles with `fix emit/surf/pmi`.
10. Diagnose wall flux and grid density (`compute grid ... species nrho` + `fix ave/grid`).

## Run
```bash
cd examples/test_west
mpirun -np 4 ../../src/spa_mpi < in.west
```

## Main outputs
- `output/gamma.only` (incident plasma flux by species on wall)
- `output/state.west` (particle state)
- `output/tmp.grid.density` (time-averaged per-species grid density from tracked sputtered particles)

## Plot helpers
- `plot_west_rz.py` for trajectory R-Z visualization.
- `plot_gamma_species_vs_surfid.py` for wall flux by species.
- `plot_grid_density_west.py` for WEST-style R-Z density maps.

Example:
```bash
python3 plot_grid_density_west.py \
  --dump output/tmp.grid.density \
  --wall input/wall.txt \
  --out output/grid_density.west.png \
  --show --log
```

## Input files
- `input/plasma.h5`
- `input/bfield.h5`
- `input/wall.txt`
- `input/8_on_74.h5` (PMI sputter table)
- `input/plasma.adas`
- `input/plasma.species`

`plasma.h5` supports multi-ion layout (`ions/*`, `ion_species/*`) and legacy single-ion fields.

## SOLEDGE conversion utility
- `tools/converters/soledge2openedge.py` writes `plasma.h5`, `bfield.h5`, and `wall.txt` from SOLEDGE HDF5 geometry.
- Core export is disabled for now (can be re-enabled later once core contour source is finalized).

Example:
```bash
cd tools/converters
python3 soledge2openedge.py
```
