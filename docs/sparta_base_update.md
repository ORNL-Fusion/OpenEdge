# Updating the vendored SPARTA base

OpenEdge vendors SPARTA in `src/` (tag or commit SHA recorded in `SPARTA_BASE`) with the
OPENEDGE package and a set of locally modified base files on top.

To update to a new upstream release:

    git clone https://github.com/sparta/sparta.git /tmp/sparta
    git checkout -b sparta-<TAG>-update
    python3 tools/update_sparta_base.py --sparta /tmp/sparta \
        --old $(cat SPARTA_BASE) --new <TAG-or-SHA>     # report
    python3 tools/update_sparta_base.py ... --apply     # safe copies + 3-way merge
    # resolve any listed CONFLICT files by hand (conflict markers in place;
    # update.cpp is ours-heavy: keep the OpenEdge mover, fold upstream in)
    grep -rl '^<<<<<<<' src/ && echo resolve these
    cmake --build <builddir> --target spa_mac_mpi -j
    ./regression/run_regression.sh
    echo <TAG-or-SHA> > SPARTA_BASE && git commit

Notes:
- files deleted locally on purpose stay deleted (the tool preserves them)
- KOKKOS: CPU builds do not compile src/KOKKOS — verify a GPU/kokkos build
  separately after an update
- keep src/OPENEDGE/update.cpp identical to src/update.cpp
