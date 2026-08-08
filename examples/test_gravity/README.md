# test_gravity — `fix force/gravity` ballistic validation

Single Ar particle at the origin, released with zero velocity under
`g_z = −9.81 m/s²`. Integrated for 1000 steps at `dt = 1 ms`. Analytic:

```
v_z(t) = g_z * t            → v_z(1 s) = −9.81 m/s
z(t)   = 0.5 * g_z * t^2    → z(1 s)   = −4.905 m
```

## Files

| File | Purpose |
|---|---|
| `in.gravity3d` | Input deck (3D Cart, `fix grav force/gravity all 0 0 -9.81`). |
| `ar.species` | Ar species table. |
| `source.1` | Launch state: 1 Ar particle at (0,0,0) with zero velocity. |
| `check_gravity.py` | Compares `output/dump.gravity3d` to analytics; PASS/FAIL exit code. |

## Run

```bash
mpirun -np 4 ~/build_oe/src/spa_mac_mpi -in in.gravity3d
python3 check_gravity.py
```

## Pass criteria (enforced by the script, exit 0/1)

- `max |vz − g t| < 1e-9` (leap-frog is exact for constant acceleration;
  measured ~2e-15)
- `max |z − g t²/2| < 1e-5` (measured ~4.5e-6 on 4 ranks)
