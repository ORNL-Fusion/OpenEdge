# test_gravity — `fix gravity` ballistic validation

Single Ar particle at the origin, released with zero velocity under
`g_z = −9.81 m/s²`. Integrated for 200 steps at `dt = 1 ms`. Analytic:

```
v_z(t) = g_z * t            → v_z(0.2 s) = −1.962 m/s
z(t)   = 0.5 * g_z * t^2    → z(0.2 s)   = −0.1962 m
```

## Files

| File | Purpose |
|---|---|
| `in.gravity3d` | Input deck (3D Cart, `fix gravity all 0 0 -9.81`). |
| `ar.species` | Ar species table. |
| `source.1` | Launch state: 1 Ar particle at (0,0,0) with zero velocity. |

## Run

```bash
source /opt/intel/oneapi/setvars.sh --force
mpirun -np 4 /home/cloud/buildOpenEdge/src/spa_mpi -in in.gravity3d
```

Produces `dump.gravity3d`. Inspect the final frame — `vz` and `z` should
match the analytic values above to machine precision (the leap-frog
integrator is exact for constant acceleration).

Last-run reference on mora (4 ranks): `max |vz − vz_ref| = 4.4e-16`,
`max |z − z_ref| = 5.0e-7`.
