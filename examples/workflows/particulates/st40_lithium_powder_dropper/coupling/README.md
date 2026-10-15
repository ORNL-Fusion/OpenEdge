# ST40 <-> SOLPS-ITER coupling

Each iteration: OpenEdge runs `openedge/in.openedge` against
`openedge/input/plasma_13589.h5`; `tools/coupling/source_spse_to_solps.py`
averages the last source windows and writes `source2d.00001` (sna0, Li+),
`she2d.00001` (she0) and `b2.sources.profile` into the SOLPS case;
`run_solps.csh` restarts SOLPS for `solps_ntim` steps; the plasma file is
rebuilt from the new `b2fstate`. `tools/coupling/loop_driver.py` repeats this
`loop.max_iters` times and archives each iteration under `loop_st40/`.

Files:

- `config.json`: paths and knobs; `${OPENEDGE_ROOT}` and `${OPENEDGE_BUILD}` come from the environment
- `run_coupling.sh`: stages `solps/st40` from the SOLPS case, checks it has Li, runs the loop
- `run_solps.csh`: SOLPS wrapper, needs `OPENEDGE_SOLPS_DIR`

Requirements: a SOLPS case with Li species (Li, Li+, Li2+, Li3+) converged
with zero Li source (the 13589_2 run is D+C only); `OPENEDGE_ROOT`,
`OPENEDGE_BUILD` with `src/spa_mpi`, `OPENEDGE_SOLPS_DIR`; python with numpy,
scipy, pandas, h5py, freeqdsk.

```bash
export OPENEDGE_ROOT=... OPENEDGE_BUILD=... OPENEDGE_SOLPS_DIR=...
cd coupling
SOLPS_SRC=/path/to/li_case ./run_coupling.sh --dry   # plan only
SOLPS_SRC=/path/to/li_case ./run_coupling.sh         # stage + loop
```

Check the hand-over from a finished OpenEdge run without SOLPS:
`python3 $OPENEDGE_ROOT/tools/coupling/source_spse_to_solps.py config.json --case=st40 --check`
