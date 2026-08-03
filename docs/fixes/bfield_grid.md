# `fix bfield/grid` — apply per-cell B-field array to the pusher

Reads three per-cell B-field columns (typically from `compute
plasma/fields`) and exposes them to the Boris / GCA pusher via
SPARTA's `field_grid` slot. Optional: with `every` set to a non-zero
interval, the columns are re-fetched periodically (for time-dependent
plasma backgrounds).

## Syntax

```
fix ID bfield/grid <BxSrc> <BySrc> <BzSrc> [every <N>]
```

| arg | meaning |
|---|---|
| `BxSrc`, `BySrc`, `BzSrc` | per-cell sources, typically `c_<compute>[col]`. Use `NULL` for any axis you don't want set (the pusher then sees zero on that axis). |
| `every N` | refresh the per-cell array every `N` steps. Defaults to once at init (static). |

The sources are interpreted in **SPARTA slot order** — for axisymmetric
runs with `x = Z, y = R`, you should not be passing physical
`(B_R, B_Z, B_φ)` into the `Bx, By, Bz` slots directly. Use
`compute plasma/fields` to project the physical components into the
right SPARTA columns automatically.

## Example

```
fix pd background file plasma.h5
compute cpf plasma/fields all background pd br bz bt
fix bfg bfield/grid c_cpf[1] c_cpf[2] c_cpf[3] every 0
global bfield_compute cpf      # GCA prefers the point query
```

For the Boris pusher, `fix bfield/grid` provides the cell-averaged
B-field. For per-particle field accuracy (GCA, near-X-point work), use
`global bfield_compute <plasma_fields_id>` instead, which routes
through `pd->query_bfield_at_point()`.

## Files

- `src/OPENEDGE/fix_bfield_grid.{h,cpp}`
- Companion: [`fix bfield/particle`](bfield_particle.md) for
  per-particle B sources.
