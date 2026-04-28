# `fix bfield/particle` — per-particle B-field source

Per-particle variant of [`fix bfield/grid`](bfield_grid.md). Pulls
three B-field components from per-particle sources (typically
particle-style variables) and exposes them to the pusher via SPARTA's
`field_particle` slot. Useful for benchmarks where each test particle
carries its own analytic B (e.g. an idealised tokamak field set by a
variable rather than a converter).

## Syntax

```
fix ID bfield/particle <BxSrc> <BySrc> <BzSrc>
```

Each `*Src` is `v_<varname>` (particle-style variable) or `NULL` to
skip that axis.

## Example

```
variable Bx0 particle 0.0
variable By0 particle 0.0
variable Bz0 particle 1.0
fix fbp bfield/particle v_Bx0 v_By0 v_Bz0
```

For production runs with a real plasma background, prefer
`fix bfield/grid` (cell-resolution) or
`global bfield_compute <plasma_fields_id>` (point-query, equilibrium-
derived).

## Files

- `src/OPENEDGE/fix_bfield_particle.{h,cpp}`
