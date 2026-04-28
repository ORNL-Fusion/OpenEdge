# `fix particle/weight` — per-particle weight infrastructure

Adds a `pweight` per-particle custom attribute, set at launch by every
weighted-emit fix (`fix surface/emit/source`,
`fix surface/emit/recycle`, `fix surface/emit/puff`, …) and consumed
by `compute grid/weighted` for properly normalised density / flux /
moment tallies.

Required whenever spatially-varying emission rates are in play —
without it, the weighted-launch fixes have no place to store the
per-particle weight and the corresponding diagnostics are wrong.

## Syntax

```
fix ID particle/weight
```

No keyword arguments. Single-instance per simulation.

## What it provides

- A custom particle attribute `pweight` (DOUBLE), default 0 on launch.
  Each weighted-emit fix overwrites this at the time the particle is
  added to the simulation, recording how many real particles per second
  this Monte-Carlo particle represents.
- Survives migration / sort / split / coalesce — SPARTA's custom-attr
  machinery follows the particle.

## Example

```
fix pw  particle/weight
fix femit_evp surface/emit/source LiSource divertor cevap \
    perspecies no normal yes nlaunch_total 200 model thermal_tsurf Tsurf_lm
compute n_grid grid/weighted all LiAll nrho_w u_w temp_w
```

## Files

- `src/OPENEDGE/fix_particle_weight.{h,cpp}`
- See [`compute_grid_weighted`](compute_grid_weighted.md) for the
  primary consumer.
