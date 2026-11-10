# Toroidal periodic caps on a wedge

A closed 10 deg wedge (`scripts/make_wedge.py` -> `input/wedge.surf`, R 1-2 m, Z +-0.5 m)
with specular walls and `surf_collide toroidal` caps. ~5500 W atoms stream toroidally at
1e4 m/s and cross the caps about 50 times each in 1000 steps; `balance_grid block 1 * 1`
puts every cap crossing on another rank. `scripts/check_caps.py` requires a constant
particle count, zero boundary exits and no "lost at periodic caps" line.

    mpirun -np 4 spa_mac_mpi -in in.wedge [-var gridcut 0.0] -log output/log
    python3 scripts/check_caps.py output/log

Two failure modes it covers:
- `gridcut 0.0` (not gated, documented): the teleported particle's cell is not a ghost of the
  sending rank, it is discarded and counted ("Particles lost at periodic caps"), all 5437 here.
- `gridcut -1` before 2026-11-10: the particle handed to the owning rank carried the PEXIT
  flag, so the receiver skipped the surface checks of its cell and the particle left the
  wedge through the cap (243 of 5437 as silent "Boundary exits"). Fixed (PENTRY).

Registered in `regression/run_regression.sh` as `wedge_periodic_caps`.
