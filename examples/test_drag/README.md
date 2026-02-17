# test_drag

Simple drag-model comparison case using validated geometry/data from `examples/test_evaporation`.

## Run

```bash
cd examples/test_drag
mpirun -np 4 ../../src/spa_mpi -in in.drag_compare
python3 plot_drag_compare.py
```

## Modes

- `mode 1`: `model epstein` (drag only)
- `mode 2`: `model epstein` + evaporation
- `mode 3`: `model coulomb` + evaporation

This isolates Coulomb drag effects relative to baseline Epstein with identical initial conditions.
