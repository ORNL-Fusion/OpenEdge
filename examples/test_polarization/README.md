# test_polarization

Single-particle polarization-drift validation case for OpenEdge Boris pushing in SI units.

This case compares simulation output against guiding-center theory for:

- `B = B0 * zhat`
- `Ex(t) = E0 * sin(omega * t)`

The script `test_polarization.py` runs the simulation and generates comparison plots in `figs_polarization/`.

## Run

From this directory:

```bash
python3 test_polarization.py
```

Or run the simulation manually:

```bash
mpirun -np 8 ~/buildOpenEdge/src/spa_mpi -in in.input
```

## Input Syntax

```sparta
fix    1 efield/particle Ex Ey Ez
fix    2 bfield/particle Bx By Bz
global efield particle 1 0
global bfield particle 2 0
```

Both `efield` and `bfield` particle-style commands take: `global <field> particle <fixID> <freq>`.

## SI Setup

- `B0 = 1.0` T, `E0 = 50.0` V/m
- `omega = 0.05 * omega_c`
- `dt = 0.02 / omega_c` (keeps `|q/m|*|B|*dt` near 0.02)
- Species: proton mass in kg, charge = +1

## Generated Figures

- `figs_polarization/a_vy_raw_vs_theory.{png,pdf}`
- `figs_polarization/b_vy_gyroavg_vs_theory.{png,pdf}`
- `figs_polarization/c_y_displacement_vs_theory.{png,pdf}`
