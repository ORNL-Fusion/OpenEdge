# test_polarization

Single-particle polarization-drift validation for the OpenEdge Boris
pusher in SI units. Compares the simulation output against the
guiding-center theory for

- `B  = B0 * zhat`  (uniform)
- `Ex(t) = E0 * sin(omega * t)`

## Files

| File | Purpose |
|---|---|
| `in.input` | Input deck (3D Cart, uniform Bz, time-dependent Ex). |
| `plasma.species` | Single H species table (proton mass, charge +1). |
| `source` | Launch state: 1 particle at the origin with small initial vy/vz. |
| `plot_polarization.py` | Post-process: parses `state`, writes comparison figures to `figs_polarization/`. |

## Run

```bash
source /opt/intel/oneapi/setvars.sh --force
mpirun -np 8 /home/cloud/buildOpenEdge/src/spa_mpi -in in.input
python3 plot_polarization.py
```

The deck runs for ~10 periods of the slow Ex(t) oscillation
(`omega = 0.05 * omega_c`, `dt = 0.02 / omega_c`, `numStep = 62832`).
Particle state is dumped every 50 steps to `state`.

## Field wiring

```sparta
fix    1 efield/particle Ex Ey Ez
fix    2 bfield/particle Bx By Bz
global efield particle 1 0
global bfield particle 2 0
```

## Setup parameters

- `B0 = 1.0 T`, `E0 = 50.0 V/m`
- `omega_c = q*B0/m`, `omega = 0.05 * omega_c`
- `dt = 0.02 / omega_c` keeps `|q/m|*|B|*dt ~= 0.02`

## Generated figures

Written to `figs_polarization/`:

- `a_vy_raw_vs_theory.{png,pdf}` — raw `v_y(t)` overlaid on theory
- `b_vy_gyroavg_vs_theory.{png,pdf}` — gyro-averaged `v_y(t)`
- `c_y_displacement_vs_theory.{png,pdf}` — `y(t)` guiding-center integral

## Pass criterion

The gyro-averaged `v_y(t)` and `y(t)` should track the theory within a
few percent. The raw-orbit panel shows fast gyration about the slow
drift; the gyro-averaged + displacement panels are the physics check.
