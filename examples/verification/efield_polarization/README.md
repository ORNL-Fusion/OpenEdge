# Polarization drift

One proton in uniform B_z = 1 T with E_x(t) = E_0 sin(omega t),
E_0 = 50 V/m, omega = 0.05 omega_c. The gyro-averaged v_y and the
guiding-center displacement y(t) are compared with polarization-drift
theory.

## Files

| File | Purpose |
|---|---|
| `in.input` | Deck; dt = 0.02/omega_c, 62832 steps (about 10 field periods), dump every 50 steps to `state` |
| `plasma.species` | Single H species |
| `source` | One particle at the origin |
| `scripts/plot_polarization.py` | Parses `state`, writes figures to `figs_polarization/` |

## Run

```bash
mpirun -np 1 /path/to/spa_mpi -in in.input
python3 scripts/plot_polarization.py
```

## Pass criterion

Gyro-averaged v_y(t) and y(t) track theory within a few percent. Figures:
raw v_y, gyro-averaged v_y, and y displacement, each against theory.
