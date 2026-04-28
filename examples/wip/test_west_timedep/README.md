# WEST Time-Dependent B-Field: Strike Point Sweeping

> **Note (2026-04-20):** Currently runs in legacy 2D Cartesian layout
> (per-radian wedge convention). Migration to SPARTA-native axisymmetric
> is queued — see `CLAUDE.md` § "Migration cookbook".

## Overview

This test case demonstrates **time-dependent magnetic geometry** in OpenEdge
for modeling strike point sweeping on the WEST lower divertor. Strike point
sweeping is a standard heat flux mitigation technique in tokamaks: the
poloidal field coil currents are modulated to oscillate the divertor strike
point position, spreading the heat and particle load over a wider area of
the target tiles.

The approach pre-computes a sequence of B-field snapshots from shifted
equilibria and loads them into the new `compute bfield/timedep` class, which
linearly interpolates between bracketing snapshots as the simulation
advances.  Plasma fields (ne, Te, Ti) remain static in flux-surface
coordinates — only the magnetic geometry evolves in time.

## Physics Basis

The rigid-shift model applies a radial displacement Delta_R to the poloidal
flux map psi(R,Z), translating the entire magnetic equilibrium horizontally.
This shifts the inner and outer strike points along the divertor target by
approximately Delta_R / sin(theta_inc), where theta_inc is the field-line
incidence angle.  For WEST with typical incidence angles of 2-3 degrees,
a 2 cm radial shift corresponds to roughly 40-60 cm of strike point
displacement along the target surface.

B-field components are recomputed from the shifted psi:

    B_R = -(1/R) d(psi)/dZ
    B_Z =  (1/R) d(psi)/dR
    B_t =  B_tf * R_tf / R     (vacuum toroidal field)

This rigid-shift approximation is valid for small displacements (|Delta_R| <
5 cm) where the plasma equilibrium shape changes minimally.  For WEST, this
matches the operational strike point sweep amplitude of +/- 2-5 cm at
frequencies of 0.2-1 Hz.

## How to Run

### Step 1: Generate B-field Snapshots

```bash
python ../../tools/converters/gen_bfield_sweep.py \
    --equ input/equilibrium.equ \
    --delta-r 0.0 0.02 0.04 0.02 0.0 -0.02 -0.04 -0.02 \
    --times   0.0 0.5  1.0  1.5  2.0  2.5   3.0   3.5 \
    --outdir input/bfield_sweep/
```

This produces 8 B-field snapshots covering one full sweep cycle (4 seconds).
The `--delta-r` values follow a triangle wave pattern, typical for WEST
sweeping operations.

### Step 2: Run OpenEdge

```bash
mpirun -np 4 spa_openedge < in.west_timedep
```

### Step 3: Verify

- Check dump files for evolving B-field values (`c_cBt[1..3]`)
- Plot strike line positions on the wall surface at different times
- Verify heat flux redistribution across divertor tiles

## Input Files Required

| File | Description |
|------|-------------|
| `input/equilibrium.equ` | SOLPS-format equilibrium file with psi(R,Z) grid |
| `input/plasma.h5` | Static background plasma (ne, Te, Ti, flows) |
| `input/species.dat` | Species definition file |
| `input/bfield_sweep/bfield_times.txt` | Generated manifest (auto-created) |
| `input/bfield_sweep/bfield_t*.h5` | Generated B-field snapshots (auto-created) |

## compute bfield/timedep Syntax

```
compute ID bfield/timedep group file_list <manifest.txt> \
    [nevery N] [interp linear|step]
```

- **file_list**: path to manifest file containing `time filename` pairs
- **nevery N**: update B-field every N timesteps (default: 1)
- **interp linear**: linearly interpolate between bracketing snapshots (default)
- **interp step**: snap to nearest snapshot (no interpolation)

The compute outputs 3 per-grid columns: `c_ID[1]` = Br, `c_ID[2]` = Bt, `c_ID[3]` = Bz.
These feed directly into `fix bfield/grid`:

```
fix bB bfield/grid c_cBt[1] c_cBt[2] c_cBt[3]
global bfield grid bB 0
```

## Verification Tests

1. **Static reproduction**: single snapshot -> results must match static
   `compute plasma/fields` B-field output
2. **Two identical snapshots**: verify B-field does not change in time
3. **Small shift**: Delta_R = 2 cm -> strike point moves ~40 cm on target
4. **Full sweep cycle**: 8 snapshots with sinusoidal sweep -> smooth
   oscillation of heat flux footprint

## Literature References

The following papers provide context, experimental data, and modeling
approaches relevant to this test case:

### WEST Divertor and Strike Point Control

1. M. Missirlian, J. Bucalossi, Y. Corre, et al.,
   "The WEST/ITER-like actively cooled tungsten divertor: operational limits
   and behaviour under power handling,"
   *Nuclear Materials and Energy*, vol. 12, pp. 1165-1170, 2017.
   DOI: 10.1016/j.nme.2016.12.030
   — Describes the WEST tungsten divertor design and strike point sweeping
   as a key power handling strategy for ITER-like plasma-facing components.

2. Y. Corre, M. Firdaouss, J.-L. Gardarein, et al.,
   "Heat flux calculation and problem of flaking on the WEST divertor,"
   *Nuclear Fusion*, vol. 62, 086005, 2022.
   DOI: 10.1088/1741-4326/ac6e6d
   — Presents IR thermography heat flux measurements on the WEST divertor,
   quantifying strike point position control effectiveness. Provides
   experimental heat flux profiles suitable for model validation.

### Divertor Heat Flux Mitigation (General)

3. A. Loarte, B. Lipschultz, A.S. Kukushkin, et al.,
   "Chapter 4: Power and particle control,"
   *Nuclear Fusion*, vol. 47, no. 6, pp. S203-S263, 2007.
   DOI: 10.1088/0029-5515/47/6/S04
   — ITER Physics Basis chapter establishing strike point sweeping alongside
   detachment and impurity radiation as primary heat flux mitigation methods.

4. R.A. Pitts, S. Carpentier, F. Escourbiac, et al.,
   "Physics basis for the first ITER tungsten divertor,"
   *Nuclear Materials and Energy*, vol. 20, 100696, 2019.
   DOI: 10.1016/j.nme.2019.100696
   — ITER divertor design basis: sweeping frequency ~1 Hz, amplitude ~few cm,
   peak heat flux limit ~10 MW/m^2 steady-state.

5. T. Eich, B. Sieglin, A. Scarabosio, et al.,
   "Inter-ELM power decay length in JET and ASDEX Upgrade,"
   *Journal of Nuclear Materials*, vol. 438, pp. S72-S77, 2013.
   DOI: 10.1016/j.jnucmat.2013.01.011
   — Empirical scaling of SOL power width (lambda_q), which determines
   required sweeping amplitude to spread the heat footprint.

### Time-Dependent Equilibrium Modeling

6. J.-F. Artaud, F. Imbeaux, J. Garcia, et al.,
   "METIS: a fast integrated modelling tool for scenario design,"
   *Nuclear Fusion*, vol. 58, 105001, 2018.
   DOI: 10.1088/1741-4326/aad5b1
   — METIS code used at CEA/WEST for equilibrium evolution, including
   time-dependent strike point sweeping scenarios.

7. F. Koechl, A.R. Polevoi, et al.,
   "ITER plasma scenarios with the JINTRAC integrated modelling suite,"
   *Nuclear Fusion*, vol. 57, 086023, 2017.
   DOI: 10.1088/1741-4326/aa7399
   — Demonstrates free-boundary equilibrium evolution with strike point
   sweeping via PF coil current modulation.

### Codeposition and Erosion Under Sweeping

8. J. Roth, E. Tsitrone, A. Loarte, et al.,
   "Recent analysis of key plasma wall interactions issues for ITER,"
   *Journal of Nuclear Materials*, vol. 390-391, pp. 1-9, 2009.
   DOI: 10.1016/j.jnucmat.2009.01.037
   — Discusses strike point sweeping as a strategy to redistribute erosion
   and codeposition patterns, reducing localized tungsten codeposition and
   tritium retention.

## Typical WEST Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Major radius R0 | 2.50 m | |
| Minor radius a | 0.50 m | |
| Toroidal field Bt | 3.7 T | at R0 |
| Sweep amplitude | +/- 2-5 cm | radial shift |
| Sweep frequency | 0.2-1 Hz | |
| SOL power width | 3-8 mm | mapped to midplane |
| Peak heat flux | 5-10 MW/m^2 | without sweeping |
| Divertor target | Tungsten | ITER-like monoblocks |
