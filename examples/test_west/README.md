# WEST test case (OpenEdge)

Primary input:
- `in.west`

Current workflow in `in.west`:
1. Load WEST wall/core surfaces.
2. Apply static `adapt_grid` refinement near lower divertor, upper divertor, limiter, and core.
3. Compute plasma/background fields with `compute plasma/fields`.
4. Compute incident plasma flux with `compute incident/plasma/flux`.
5. Compute sputter source with `compute pmi/surf/data` (`sputter_flux_total`).
6. Emit sputtered particles with `fix emit/surf/pmi`.
7. Diagnose wall flux and grid density (`compute grid ... nrho` + `fix ave/grid`).

## Run
```bash
cd examples/test_west
mpirun -np 4 ../../src/spa_mpi < in.west
```

## Main outputs
- `output/gamma.only` (incident plasma flux by species on wall)
- `output/state.west` (particle state)
- `output/tmp.grid.density` (time-averaged grid density from tracked sputtered particles)

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
- `input/core.txt`
- `input/8_on_74.h5` (PMI sputter table)

`plasma.h5` supports multi-ion layout (`ions/*`, `ion_species/*`) and legacy single-ion fields.

## SOLEDGE conversion utility
- `input/soledge2openedge.py` writes `plasma.h5` + `bfield.h5` and debug plots.
