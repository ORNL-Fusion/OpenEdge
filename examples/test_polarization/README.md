# test_polarization

Single-particle polarization-drift validation case for OpenEdge Boris pushing in SI units.

This case compares simulation output against guiding-center theory for:

- `B = B0 * zhat`
- `Ex(t) = E0 * sin(omega * t)`

The script `test_polarization.py` reads the particle dump and generates comparison plots in `figs_polarization/`.

## Run Simulation

Two input variants are provided that produce equivalent results:

- `in.input.part` — particle-style E/B fields (variables evaluated per particle)
- `in.input.grid` — grid-style E/B fields (variables evaluated per grid cell)

From this directory:

```bash
# particle-style fields
mpirun -np 1 ../../src/spa_mpi -in in.input.part

# grid-style fields
mpirun -np 1 ../../src/spa_mpi -in in.input.grid
```

Expected output:

- `state.part` or `state.grid` (particle dump)
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

## Input Syntax

All `global efield` and `global bfield` commands use the same 4-argument format:

```sparta
global efield <particle|grid> <fixID> <Nfreq>
global bfield <particle|grid> <fixID> <Nfreq>
```

Particle-style example (`in.input.part`):

```sparta
variable Ex particle "v_Ex_global"
fix    1 efield/particle Ex Ey Ez
fix    2 bfield/particle Bx By Bz
global efield particle 1 1
global bfield particle 2 0
```

Grid-style example (`in.input.grid`):

```sparta
variable Ex grid "v_Ex_global"
fix    1 efield/grid Ex Ey Ez
fix    2 bfield/grid Bx By Bz
global efield grid 1 1     # Nfreq=1 for time-dependent Ex
global bfield grid 2 0     # Nfreq=0 for static B (computed once at start)
```

Each field (efield, bfield) has its own independent Nfreq. An Nfreq of 0 computes the field once at run start. An Nfreq of 1 recomputes every timestep (required for time-dependent fields).

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
