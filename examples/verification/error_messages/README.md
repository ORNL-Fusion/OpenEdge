# Input-file error messages

Every reader of an input file stops with one message: what failed, the exact file or
dataset, and what to do. `scripts/check.py` builds broken inputs in `output/` and runs
each on 1 and 2 ranks; a case passes if the run stops without hanging and prints the
expected text.

    python3 scripts/check.py        # SPA=<binary, with the Kokkos arguments if any>; needs h5py

Covered: the plasma file of `fix background` (missing, launched from another directory,
unreadable, not HDF5, wrong dimensions, wrong rank, zones3d file read as legacy and the
reverse), the equilibrium file (missing, incomplete field, missing grid size),
`database/processes.h5` (missing, unreadable, not HDF5, element without reactions), the
sputter and reflection tables of `surf_react surface/pwi` (invalid database, wrong
dimensions, missing axis, missing table: none may fall back to the analytic model), a
valid HDF5 file with a user block, and
the summary printed after a successful load.
