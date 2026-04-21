# External coupling (library API)

C-callable entry points for driving OpenEdge from an outer loop (Python,
Gkeyll, SOLPS). Declared in `src/library.h`, implemented in
`src/library.cpp`.

Standard SPARTA calls (`sparta_open`, `sparta_file`, `sparta_command`,
`sparta_extract_compute`, …) are preserved. OpenEdge extensions:

| Call | Purpose |
|------|---------|
| `openedge_extract_fix(ptr, id, 2, type)` | Return `double*` (`type=0`, vector_grid) or `double**` (`type≥1`, array_grid) for a per-grid fix. Used to pull source tallies from `fix chem/adas`. |
| `openedge_get_ngrid(ptr)` | Number of owned grid cells on this rank. |
| `openedge_reload_plasma(ptr, cid, path)` | Reload plasma HDF5 on a `compute plasma/fields`. `path=NULL` re-reads the existing file; otherwise swap to `path`. Used between Gkeyll/SOLPS iterations. |
| `openedge_reset_fix_tally(ptr, id)` | Zero the 20-col source tally on a `fix chem/adas`. Called between coupling iterations so each handoff sees a fresh accumulation. Leaves per-type counters and exhaust state intact. Silent no-op if the ID is wrong or not a `FixChemAdas`. |

## Coupling loop skeleton

```python
for k in range(n_outer):
    openedge_command(ptr, f"run {n_steps}")
    Sn_Smom_Se = ctypes_cast(openedge_extract_fix(ptr, b"fchem", 2, 1),
                             ngrid, 20)
    plasma = outer_solver.step(Sn_Smom_Se)
    plasma.write_hdf5("plasma_k+1.h5")
    openedge_reload_plasma(ptr, b"cplasma", b"plasma_k+1.h5")
    openedge_reset_fix_tally(ptr, b"fchem")
```

## Existing driver

- `tools/coupling/openedge_solps_driver.py` (subprocess model)
- `solps_interface.py` handles SOLPS file IO
