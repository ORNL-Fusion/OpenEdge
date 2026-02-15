# test_boris_grid

Grid-based Boris pusher test using `compute plasma/fields` as field source.

For 3D `global ... grid` fields, Boris now samples `E/B` via trilinear
interpolation on uniform grids (with fallback to cell-centered values
if neighbor cells are unavailable).

## Files
- `in.constant`: `compute plasma/fields ... constant ...`
- `plasma.species`: one charged species (H+), with `mass` in kg and `charge` in units of `e`
- `source.single`: one test particle
- `plot_trajectory.py`: trajectory plotting script
  - plots `R-Z` trajectory and kinetic energy vs time
  - reports relative KE drift range in stdout

## Charge/mass convention used by Boris

The pusher uses:

`q/m = charge[e] * echarge[C] / mass[kg]`

So in species files:
- `charge` must be integer-like charge state in units of `e` (`0`, `1`, `2`, ...)
- `mass` must be physical mass in kg

## Boris debug/tuning flags

You can tune Boris from the input script via `global`:

- `global boris_subcycles N`
  - splits each move step into `N` Boris substeps
  - use this to improve gyro-orbit accuracy when `dt` is large
- `global boris_dump yes|no`
  - enables/disables runtime Boris field dumps
- `global boris_dump_every M`
  - print cadence in timesteps when `boris_dump yes` is enabled
  - prints one line from rank 0 for particle index `i=0`
- `global boris_bad_dt_check yes|no`
  - enables/disables warning when Boris substep is too large
- `global boris_bad_dt_limit X`
  - warning threshold for `|q/m|*|B|*dt_sub` (default `0.1`)

## Run constant case
```bash
mpirun -np 4 ../../src/spa_mpi < in.constant
MPLCONFIGDIR=/tmp /usr/bin/python3 plot_trajectory.py --case constant
```

## Output
- `output/state.constant`
- `output/trajectory_ke.constant.png`
