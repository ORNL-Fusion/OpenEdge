# WEST test case (OpenEdge)

> **Note (2026-04-20):** Despite the `_axi` name, this test currently runs
> in 2D Cartesian mode (`boundary p p p`, `x=R`, `y=Z`) with a per-radian
> wedge convention — every `compute grid nrho` and `compute surf flux`
> needs a `2*pi*R̄` post-multiply for full-3D values. Migration to true
> SPARTA-native axisymmetric (`boundary o ao p`, `x=Z`, `y=R`) is queued
> as the next step after `test_diii_d_neutrals` (the pilot). Underlying
> physics fixes (`thermal_force`, `cross_diffusion`, etc.) are already
> axi-aware via `openedge_geom`; only the converter rerun + input deck
> flip + `fnum` re-tune are pending. See `CLAUDE.md` § "Migration cookbook".

Input data for this example is included under `input/`.

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
9. Emit sputtered particles with `fix surface/emit/sputter`.
10. Diagnose wall flux and grid density (`compute grid ... species nrho` + `fix ave/grid`).

## Run
```bash
cd examples/test_west_axi
mpirun -np 4 ../../src/spa_mpi < in.west
```

## Main outputs
- `output/gamma.only` (incident plasma flux by species on wall)
- `output/state.west.o_sputter` (particle state)
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
- `input/74_on_74_pmi.h5` (PMI sputter table)
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
