# `global pusher`

Configure the charged-particle pusher: full-orbit Boris, guiding-center
(GCA), or the hybrid mode that switches between them per particle. An
optional sheath model accelerates ions toward absorbing surfaces.

## Syntax

```text
global pusher mode boris|hybrid|gca \
    [plasma ID] [skip MIXTURE_ID] [subcycles N] \
    [gca_switch F] [gca_integrator rk4|rk2|simple] \
    [boris_near M [rhoL]] [gc_wall a0|flux] [switch_log FILE] \
    [bad_dt_check yes|no] [bad_dt_limit MAX] \
    [dump yes|no] [dump_every N] \
    [sheath off|kick|spatial|boundary [geom ID] [mD_amu M] \
            [dmax D] [waveform ATTR FREQ_HZ]]
```

## Modes

- `boris` integrates the full gyro-orbit with the Boris rotation. Use it
  when the gyroradius is resolved or gyro-phase physics matters.
- `gca` integrates the guiding center (drifts, mirror force, parallel
  dynamics) and is orders of magnitude cheaper when the gyroradius is far
  below the resolved scales.
- `hybrid` selects per particle: GCA where the local gyroradius is small
  against the cell size and Boris elsewhere. `gca_switch F` sets the
  switching threshold (gyroradius relative to the local cell scale).

Particles in the `skip` mixture bypass the pusher entirely — use this for
finite-size particulates, whose forces come from the particulate fixes.

## Keywords

- `plasma ID` names the plasma/field provider (a `fix background`).
- `subcycles N` sub-steps the orbit integration each move (default 1).
- `gca_integrator` selects the guiding-center integrator: `rk4` (default,
  4-stage), `rk2` (midpoint), or `simple` (1-stage, no curvature/curl-b
  terms — cheapest, for scoping only).
- `boris_near M` forces full-orbit Boris within distance `M` (meters) of
  the sheath-geometry surfaces; append `rhoL` to interpret `M` as a
  multiple of the local gyroradius instead. `0` disables.
- `gc_wall a0|flux` chooses how a guiding center is tested against walls:
  `a0` uses the guiding-center point; `flux` accounts for the gyro-averaged
  flux surface of the orbit (default used by the supported cases).
- `switch_log FILE` records hybrid mode switches for diagnostics.
- `bad_dt_check yes` guards against integration steps that exceed
  `bad_dt_limit` (fraction of a gyro/drift period) and subdivides them.
- `dump yes` + `dump_every N` write pusher diagnostic trajectories.

## Sheath model

- `off` — no sheath.
- `kick` — impulse model: the ion receives the sheath energy at the wall.
- `spatial` — the sheath potential is applied as a spatially resolved
  impulse over the sheath width.
- `boundary` — the sheath acts as a boundary layer at absorbing surfaces
  (the standard choice for the supported cases).

`geom ID` must name a `compute nearest_surf/grid` that provides per-cell
wall distance and normals; it is required for every mode except `off`.
`mD_amu` sets the background-ion mass (default 2.0). The sheath extent
and potential are set automatically — `dmax = max(5 L_MPS, 10 lambda_D)`
with a Bohm floating-wall potential, using a combined Coulette–Manfredi
near-wall profile with a Borodkina tail. `dmax D` overrides the extent.
`waveform ATTR FREQ_HZ` drives the wall potential from a per-surface
attribute at the given frequency (RF-biased surfaces).

## Example

```text
compute cgeom nearest_surf/grid all wall dist surfid nx ny nz
global pusher mode hybrid plasma pd skip particulates \
       gca_switch 2.5 gca_integrator rk4 \
       bad_dt_check yes bad_dt_limit 0.5 \
       gc_wall flux sheath boundary geom cgeom mD_amu 2.0
```
