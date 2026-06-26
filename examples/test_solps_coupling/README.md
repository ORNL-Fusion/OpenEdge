# test_solps_coupling — OpenEdge ↔ SOLPS-ITER coupling for Li sources

Iterative two-way coupling: OpenEdge runs Li droplet transport (Antoine
evaporation + Epstein drag + OML charging + Boris/sheath pusher) against a
SOLPS-ITER plasma background, feeds the resulting Li volumetric source back
into SOLPS, and SOLPS updates the plasma for the next OpenEdge pass. This
repeats for a fixed number of iterations.

## How one iteration works

Each iteration is one OpenEdge → SOLPS leg (the "one-shot" path), driven by
`tools/coupling/oneshot_driver.py`:

1. **OpenEdge** runs against the current `plasma_<case>.h5` until the
   volume-integrated Li ionization source `int Sp` plateaus (or hits
   `max_steps`). Its source dump is `output/source_SpSe.*`.
2. **`source_spse_to_solps.py`** maps that Li source onto the SOLPS B2 grid
   and writes the SOLPS source files (`b2.sources.profile`, `source2d.*`) +
   `b2mn.dat` flags.
3. **Restart handoff**: `b2fstate → b2fstati` is copied so SOLPS continues
   from the *converged* state, not a flat one (printed as a `== RESTART: …`
   banner). iter 0 seeds from the staged standalone state; iter N≥1 from the
   previous iteration's `b2fstate`.
4. **SOLPS** runs (`run_solps.csh` → `b2run b2mn`) to update the plasma.
5. **Rebuild for the next pass**: `convert_solps_plasma.py --heatflux`
   regenerates `plasma_<case>.h5` from the new `b2fstate`, `run_b2plot_wlld.sh`
   refreshes target heat flux, and the Li strip surface state is rebuilt.

The loop (`loop_driver.py`) repeats this `loop.max_iters` times, archiving
each iteration under `ensemble_runs/<case>/loop_<case>/iter_NN/` and recording
`int Sp` / `int Qe` to `convergence.csv`.

## The four cases

`attached__mist`, `attached__no_mist`, `detached__mist`, `detached__no_mist`
(attached vs detached divertor × with/without droplet "mist"). Each has its
own OpenEdge deck and SOLPS run dir; all share the read-only `solps/baserun`.

## Run

**Prerequisites**
- A SOLPS-ITER tree, an OpenEdge build, and the OpenEdge repo.
- A staged SOLPS baserun (`solps/baserun/` with `b2fgmtry`, `dg.equ`) and a
  per-case SOLPS dir (`solps/<case>/` with `b2fstate`/`b2fstati`).
- `coupled_test/` is gitignored (large SOLPS data) — bring your own under
  `solps/`. Paths and knobs live in `coupled_test/config.json`.

**Setup for a new machine** — all paths default to the original layout but are
env-overridable (no file edits needed; just export what differs):

| Env var | Default | Points at |
|---|---|---|
| `OPENEDGE_ROOT` | `/home/cloud/OpenEdge` | the OpenEdge repo |
| `OPENEDGE_BUILD` | `/home/cloud/buildOpenEdge` | the CMake build dir (`$OPENEDGE_BUILD/src/spa_mpi`) |
| `OPENEDGE_SOLPS_DIR` | `/home/cloud/local/solps/solps-iter-3.0.8-devel` | the SOLPS-ITER tree (used by `run_solps.csh`) |

```bash
export OPENEDGE_ROOT=/path/to/OpenEdge
export OPENEDGE_BUILD=/path/to/buildOpenEdge
export OPENEDGE_SOLPS_DIR=/path/to/solps-iter
```

`config.json` references these as `${OPENEDGE_ROOT}` / `${OPENEDGE_BUILD}`; the
drivers expand them at load, and the run scripts export the defaults.

All commands run from `coupled_test/`:

```bash
cd coupled_test

# 1. Single case: stage an ensemble run (if needed) and run the loop
./run_coupling.sh attached__mist
./run_coupling.sh attached__mist --fresh    # rebuild the run from clean inputs
./run_coupling.sh attached__mist --dry      # print the plan, launch nothing

# 2. All four cases, fresh, sequentially (long run -> background it)
nohup bash run_ensemble.sh > ensemble.log 2>&1 &

# 3. One-shot smoke test (a single OE->SOLPS leg, OpenEdge-only)
python3 "$OPENEDGE_ROOT/tools/coupling/oneshot_driver.py" \
        config.json --case=attached__mist --no-solps
```

`run_coupling.sh` stages a **self-contained run** under `ensemble_runs/<case>/`
(a private copy of the SOLPS case + OpenEdge deck/plasma, with the baserun
symlinked read-only) so the loop never mutates your validated standalone
inputs. `ensemble_runs/` is regenerated each time you test a case.

## Key knobs in `config.json`

| Key | Meaning |
|---|---|
| `loop.max_iters` | number of OpenEdge↔SOLPS iterations (the loop runs to this; convergence-stop is disabled) |
| `oe_convergence` | OpenEdge plateau test: `rel_tol`, `window`, `poll_seconds`, `max_steps` |
| `mpi_np_openedge` / `mpi_np_solps` | MPI ranks for each code |
| `cases.<case>.droplet_emit` | per-case droplet emission rate (→ deck `n ${ndrop}`) |
| `plasma_build` | `b2fgmtry`, `equ_file`, grid for `convert_solps_plasma` |
| `solps_run_script` | SOLPS wrapper (`run_solps.csh`) |
| `openedge_binary` | path to `spa_mpi` |

## Physics knobs in the OpenEdge deck (`openedge/full_runs/<case>/in.oedge_wall`)

- `heatflux_scale` — multiplier on the parallel heat flux fed into Antoine
  evaporation. Note the converter now writes the **parallel** flux
  (`fht/sxprl`, ≈20× the old poloidal `fht/sx`); recheck this scale if you
  carried over an old calibration.
- `rocket_eta` — recoil efficiency from asymmetric evaporation along −∇Te
  (0 disables).
- `fix droplet/drag model epstein` — Epstein subsonic drag (`coulomb` also
  available).

## Layout

```
coupled_test/                  active coupling workflow (gitignored data)
  config.json                  all knobs + per-case paths
  run_coupling.sh              single case: stage ensemble run + run loop
  run_ensemble.sh              all 4 cases, fresh, sequential
  run_solps.csh                SOLPS wrapper invoked by the driver
  run_b2plot_wlld.sh           SOLPS WLLD target-heat-flux post-processor
  solps/
    baserun/                   shared read-only SOLPS base (b2fgmtry, dg.equ)
    <case>/                    per-case SOLPS run dir (b2fstate/b2fstati)
  openedge/full_runs/<case>/   per-case OpenEdge deck + output
  ensemble_runs/<case>/        staged self-contained run (created on run)
    loop_<case>/iter_NN/       per-iteration archive + convergence.csv
analysis/                      trajectory / mass-loss / convergence plotters
input/                         shared deck inputs
```

## Driver scripts (`tools/coupling/`)

| Script | Role |
|---|---|
| `setup_coupling.py` | stage a campaign tree from `config.json` |
| `loop_driver.py` | iterate the OE↔SOLPS legs to `max_iters`, archive each |
| `oneshot_driver.py` | one OE→SOLPS leg (OpenEdge → source → SOLPS) |
| `source_spse_to_solps.py` | map OpenEdge Li source onto the SOLPS B2 grid |
| `solps_interface.py` | read SOLPS `b2fgmtry`/`b2fstate`, write `plasma.h5` |
| `build_tsurf_surf.py` | build the Li-strip surface temperature state |
| `convert_solps_plasma.py` | SOLPS `b2fstate` → OpenEdge `plasma.h5` (`tools/converters/`) |
