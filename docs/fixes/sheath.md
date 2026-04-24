# Sheath models — `global sheath ...`

Per-particle sheath physics applied at and near walls. Two execution
modes (kick vs. spatial), two spatial models (Borodkina, Coulette-
Manfredi), driven by `compute nearest_surf/grid` for cell-level wall
geometry and a plasma provider for upstream Te/ne/Ti.

Implementation: `src/OPENEDGE/sheath_models.{h,cpp}` for the physics,
`update.cpp` for the per-particle and per-subcycle integration.

## Syntax

```
global sheath geom_compute <ID> {plasma_compute|plasma_fix} <ID> \
       [model borodkina|coulette_manfredi] \
       [kick yes|no] \
       [dmax <m>] \
       [pot_mult <Z>] \
       [mD_amu <amu>]
```

- **`geom_compute <ID>`** — `compute nearest_surf/grid` providing per-cell
  distance to the nearest wall surface, surface index, and outward normal.
- **`plasma_compute|plasma_fix <ID>`** — source of upstream Te/ne/Ti at
  the particle position. Either a `compute plasma/fields` or a
  `fix plasma/data`.
- **`model`** (default `borodkina`) — spatial profile choice; ignored
  when `kick yes`.
- **`kick yes|no`** (default `no`) — kick mode applies the sheath as a
  velocity boost at wall collision; spatial mode integrates the sheath
  E-field per Boris subcycle as the particle approaches the wall.
- **`dmax`** (default 0 — auto) — extra ceiling on the sheath
  engagement distance. The runtime always computes its own physics-
  derived `d_max` (see below); `dmax` clamps it from above.
- **`pot_mult`** — total wall drop in units of Te. Borodkina default
  2.5 (single floating wall, Z=1); Coulette-Manfredi default 0
  (uses internal kinetic-fit value).
- **`mD_amu`** — background ion mass for sheath scales (default
  D = 2.014).

## Modes

### `kick yes` — velocity boost at wall collision

At the moment of wall impact, the ion's normal velocity is augmented by
`Δv = sqrt(2·Z·e·φ/m)` where `φ = pot_mult · Te`. No subcycle E-field,
no `model` selection, no `dmax`. Recommended for IEAD / impact-energy
diagnostics — the deposited energy is correct without resolving the
sheath thickness on the Boris timestep, and grazing-angle ions don't
get gyro-orbit blowups from sub-Debye E-fields.

Uses `particle->species[isp].mass` and `.charge` (NOT
`particles[i].mass`, which is zero for gas-phase particles).

### `kick no` — spatial sheath E-field

The Boris pusher evaluates an E-field at every subcycle when the
particle sits inside `d_max` of the nearest wall. The field is sourced
from one of:

- **`borodkina`** — Debye + magnetic-presheath two-exponential profile.
  `pot_mult` sets the total drop in Te units (default 2.5).
- **`coulette_manfredi`** — Coulette-Manfredi kinetic PIC fit, two
  exponentials in `s = d/λ_D`, with a Borodkina-style Chodura tail
  blended in for `s ≥ 60` so the magnetic-presheath physics survives
  past the kinetic-Debye region. The slow component scales by
  `ρ_i/λ_D`. Better for CS+DS transition; default `pot_mult = 0`
  uses the kinetic fit value internally.

## α-angle (Chodura metrics)

The angle between **B** and the outward wall normal is the input that
sets the magnetic-presheath thickness `L_MPS = ρ_i · tan(α_n)`. It is
computed at every sheath evaluation by `chodura_metrics()` from
`(B, n)` at the particle position, not assumed perpendicular. This is
why the kick mode and the spatial mode both work at grazing angles —
the wall drop and the MPS scale degrade gracefully as α → 0.

`α_n` is clamped to `[0°, 90°]` in `tan()` to keep `L_MPS` finite at
exactly grazing incidence.

## `auto_dmax` — engagement cutoff

When `dmax` is unset (or 0), the runtime activates the sheath only for
particles within

```
d_max = max(5 · L_MPS, 10 · λ_D)
```

evaluated per particle from local `Te, Ti, ne, |B|, α`. The factor of
5 captures the slowly-decaying CM Chodura tail; the floor at `10·λ_D`
keeps the cutoff sensible when α is near 90° and `L_MPS → 0`. A
user-supplied `dmax` is applied as an additional upper bound on top of
this — useful for capping CPU when very-large `L_MPS` would otherwise
engage the sheath several mm into the cell.

## `prepare` / `evaluate` split

The Boris subcycle loop hits the sheath E-field every step, but
upstream `Te, ne, |B|, α` and the derived scales (`λ_D, ρ_i, L_MPS,
fd, φ_0`, the model fit coefficients) are constant over a single Boris
call. The split:

- **`sheath_prepare_borodkina` / `sheath_prepare_coulette_manfredi`** —
  run the expensive transcendentals once per Boris call, return a
  `SheathEmagCoeffs` struct.
- **`sheath_emag_at_distance(coeffs, d)` / `sheath_phi_at_distance(...)`** —
  per-subcycle: 2-4 `exp()` and a handful of multiplies.

This is the dominant per-step cost saving for spatial sheath runs at
moderate-to-high subcycle counts.

## Boltzmann electron-density correction

When sheath is active, `Update::cache_plasma_particles()` overwrites
the per-particle `ne_vec` with

```
ne_local = ne_upstream · exp(-φ(d_particle) / Te)
```

inside `d_max`, where `φ(d_particle)` is evaluated by the same
prepared-sheath coefficients used by the pusher. Outside `d_max` the
upstream value passes through unchanged.

**Consequence for `fix volume/chem/adas`:** the fast-path consumer
reads `ne_vec[ip]` directly and therefore sees the depleted electron
density inside the sheath, *without* any plumbing in the chem fix
itself. Ionization sources fall off correctly into the magnetic
presheath as the bulk density depletes — this is what makes
near-wall neutral transport quantitative without a separate sheath
model in the chem path.

## Required setup

```
compute       cgeom    nearest_surf/grid all <wall-group> dist nx ny nz surfid
compute       cplasma  plasma/fields all file plasma.h5 ...
global        sheath geom_compute cgeom plasma_compute cplasma kick yes
```

Or with a `fix plasma/data` provider:

```
fix           pd       plasma/data file plasma.h5 ...
compute       cgeom    nearest_surf/grid all <wall-group> dist nx ny nz surfid
global        sheath geom_compute cgeom plasma_fix pd kick no model coulette_manfredi
```

## Related

- `compute nearest_surf/grid` — per-cell wall geometry (distance,
  surface index, outward normal). Required input.
- `fix volume/chem/adas` — consumes the Boltzmann-corrected `ne` from
  the per-particle plasma cache; near-wall ionization rates fall off
  automatically. See [`volume_chem_adas.md`](volume_chem_adas.md).
- `compute thermal_sheath/grid` — per-cell sheath drop diagnostic.
