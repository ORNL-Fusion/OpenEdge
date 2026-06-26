# `solps/` — SOLPS-ITER run data (NOT committed)

This folder holds the SOLPS-ITER plasma background data the coupling loop runs
against. It is **heavy** (each `b2fstate` ≈ 90 MB, `balance.nc`/`b2fplasmf`
≈ 100s of MB) and is **deliberately git-ignored** — only this `README.md` is
tracked. Bring your own SOLPS runs and drop them in with the layout below.

## Required layout

```
solps/
  baserun/          shared, read-only SOLPS base (one per device/equilibrium)
  attached__mist/        per-case converged SOLPS run
  attached__no_mist/
  detached__mist/
  detached__no_mist/
```

The four case names must match `cases.*` in `../config.json`. `setup_coupling.py`
symlinks `baserun/` into each staged ensemble run read-only and copies the
chosen case dir.

## What each dir must contain

**`baserun/`** — the shared base (geometry + equilibrium + rate data):

| File | Purpose |
|---|---|
| `b2fgmtry` | B2 grid geometry (also parsed by `solps_interface` if no `.mat`) |
| `b2fstati` | restart/initial plasma state |
| `dg.equ` | magnetic equilibrium (used by `convert_solps_plasma --equ-file`) |
| `b2fpardf`, `b2frates`, `fort.30` | parameter/rate tables b2run needs |
| `b2.*.parameters*`, `b2ai.dat`, `input.dat` | B2/EIRENE input decks |

**`<case>/`** — a converged SOLPS run for that scenario:

| File | Purpose |
|---|---|
| `b2fstate` | **converged** plasma state — the loop seeds `b2fstati` from this |
| `b2fstati` | restart state (overwritten from `b2fstate` each SOLPS launch) |
| `b2fgmtry`, `b2fpardf`, `b2frates` | geometry + tables |
| `b2mn.dat` | run control |
| `b2mn.dat.openedge.bak` | pre-coupling `b2mn.dat`; `setup_coupling` resets from it |
| `b2.*.parameters`, `input.dat` | B2/EIRENE input decks |

## Notes

- `b2fstate` must be the **converged** standalone solution (not a flat init) —
  the loop continues SOLPS from it (see the `== RESTART:` banner).
- EIRENE atomic-data files (`HYDHEL`, `METHANE`, `SPUTER`, `H2VIBR`, `AMMONX`,
  `fort.*`) are pulled from `baserun/` by `run_solps.csh` at run time.
- The SOLPS-ITER install itself is separate; point at it via
  `OPENEDGE_SOLPS_DIR` (default `/home/cloud/local/solps/solps-iter-3.0.8-devel`).

## Quick check

After dropping in your data:

```bash
for d in baserun attached__mist attached__no_mist detached__mist detached__no_mist; do
  [ -e "solps/$d/b2fgmtry" ] && echo "OK   $d" || echo "MISS $d (no b2fgmtry)"
done
```
