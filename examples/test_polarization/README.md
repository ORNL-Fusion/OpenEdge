# test_polarization

Single-particle polarization-drift validation case for OpenEdge Boris pushing in SI units.

This case compares simulation output against guiding-center theory for:

- `B = B0 * zhat`
- `Ex(t) = E0 * sin(omega * t)`

The script `test_polarization.py` reads the particle dump (`state`) and generates comparison plots in `figs_polarization/`.

## Run Simulation

From this directory:

```bash
../../src/spa_mpi -in in.input
```

Expected output:

- `state` (particle dump)
- `log.openedge` (or your selected log file)

## Analyze Results

Python dependencies:

```bash
python3 -m pip install numpy matplotlib
```

Run analysis:

```bash
python3 test_polarization.py
```

Generated figures:

- `figs_polarization/a_vy_raw_vs_theory.{png,pdf}`
- `figs_polarization/b_vy_gyroavg_vs_theory.{png,pdf}`
- `figs_polarization/c_y_displacement_vs_theory.{png,pdf}`

## Notes on Current Input Syntax

For this repository version, the field registration in `in.input` is:

```sparta
fix    1 efield/particle Ex Ey Ez
fix    2 bfield/particle Bx By Bz
global efield particle 1
global bfield particle 2 1
```

The `efield` form above does not take an update frequency argument, while `bfield` currently does in this input style.

## SI Setup Used in This Case

Species file uses:

- mass in kg (`m_i = 1.67262192369e-27`)
- charge in units of elementary charge (`z = +1`)

Input uses SI field values and derived frequencies:

- `B0 = 1.0` T
- `E0 = 50.0` V/m
- `omega_c = q_e * B0 / m_i`
- `omega = 0.05 * omega_c`

Timestep is tied to cyclotron frequency:

```sparta
variable dt equal 0.02/v_omegac
```

This keeps `|q/m|*|B|*dt` near `0.02` (before any Boris subcycling), which is a stable SI-scale setting for this test.
