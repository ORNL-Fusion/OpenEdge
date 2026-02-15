# WEST test case (OpenEdge)

This case runs axisymmetric WEST particle pushing using `compute plasma/fields`
with split plasma and magnetic input files:

- `input/plasma.h5`
- `input/bfield.h5`

The main input script is:

- `in.west`

## Run

```bash
cd examples/test_west
mpirun -np 4 ../../src/spa_mpi < in.west
```

Outputs are written to:

- `output/state.west` (particle trajectories)
- `output/epar.west.grid` (grid diagnostics, if enabled in `in.west`)

## Plasma/B-file convention

`compute cwest plasma/fields all file input/plasma.h5 input/bfield.h5 ...`
expects 2D `(r,z)` plasma and B-field tables.

`plasma.h5` can include both:

- legacy main-ion fields (`dens_i`, `temp_i`, `parr_flow`, etc.)
- optional multi-ion groups (`ion_species/*`, `ions/*`)
- optional per-ion metadata used by PMI models:
  - `ion_species/mass_amu`
  - `ion_species/charge_state_z`

The ion list is not fixed to O charge states. It can include any set of
species (for example `D+`, `O+..O8+`, `Ne+`, `Li+`, mixed impurity cases).

For current `compute plasma/fields` use in this test, requested fields are read
by dataset name and are independent of how many optional ion species are stored.

## SOLEDGE conversion

The conversion script is:

- `input/soledge2openedge.py`

It writes `plasma.h5` and `bfield.h5` separately and can also generate debug
plots for field sanity checks and wall-flux diagnostics.

## Incident Gamma check (1 step)

`in.west` includes `compute incident/flux` and `dump surf` for wall Gamma.
For a quick validation run:

```bash
cd examples/test_west
mpirun -np 4 ../../src/spa_mpi < in.west
```

Gamma output:
- `output/gamma.only`

This can be run with `run 0` or `run 1` depending on whether only field/surface
interpolation check is desired.
