# ST40 <-> SOLPS-ITER coupling

Each iteration:

1. OpenEdge runs `openedge/in.openedge` (neutral mode) against
   `openedge/input/plasma_13589.h5` to `nsteps`, or until the stats scalar
   `c_cSp_tot` plateaus (`oe_convergence`).
2. `tools/coupling/source_spse_to_solps.py` averages the last
   `source_export.average_last_windows` 20 ms source dumps, maps Sp and Qe
   onto the B2 cells of the staged SOLPS case (the axi layout, dump x = Z and
   y = R, follows from `plasma_build.geometry`), writes `source2d.00001` (sna0, Li+) and
   `she2d.00001` (she0), `b2.sources.profile`, and sets `b2sral_inputfile`
   and `b2mndr_ntim` in `b2mn.dat`.
3. `run_solps.csh` restarts SOLPS from the previous `b2fstate` for
   `solps_ntim` steps.
4. The plasma file is rebuilt from the new `b2fstate` with the same recipe
   as `openedge/scripts/build_background_13589_2.sh` (`plasma_build`).

`tools/coupling/loop_driver.py` repeats this `loop.max_iters` times and
archives each iteration under `loop_st40/iter_NN/` with `convergence.csv`.

## Files

| File | Role |
|---|---|
| `config.json` | every path and knob; `${OPENEDGE_ROOT}` and `${OPENEDGE_BUILD}` expand from the environment |
| `run_coupling.sh` | stages `solps/st40` from `../solps/run_dir` (lean copy, baserun symlinked), checks the case has Li species, launches the loop |
| `run_solps.csh` | SOLPS wrapper the driver calls; needs `OPENEDGE_SOLPS_DIR` |
| `solps/`, `loop_st40/` | staged case and iteration archives, not tracked |

## Prerequisites on the SOLPS machine

- `OPENEDGE_ROOT`, `OPENEDGE_BUILD` (with `src/spa_mpi`), `OPENEDGE_SOLPS_DIR`.
- Python with numpy, scipy, pandas, h5py, freeqdsk (the converter reads the gfile).
- A SOLPS case for 13589_2 that already carries lithium species (Li, Li+,
  Li2+, Li3+ in `b2fstate`) and is converged with zero Li source. Erin's
  `solps/run_dir` has only D and C: it cannot receive an Li+ source. Point
  `SOLPS_SRC` at the Li-enabled case when staging:
  `SOLPS_SRC=/path/to/li_case ./run_coupling.sh --fresh`.

## Run

```bash
export OPENEDGE_ROOT=/path/to/OpenEdge OPENEDGE_BUILD=/path/to/build OPENEDGE_SOLPS_DIR=/path/to/solps-iter
cd examples/workflows/particulates/st40_lithium_powder_dropper/coupling
./run_coupling.sh --dry          # plan only
./run_coupling.sh                # stage (first time) + loop
```

Standalone check of the hand-over without SOLPS, from a finished OpenEdge run:

```bash
python3 $OPENEDGE_ROOT/tools/coupling/source_spse_to_solps.py config.json --case=st40 --check
```
