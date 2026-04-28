# `fix efield/particle` — per-particle E-field source

Per-particle variant of [`fix efield/grid`](efield_grid.md). Pulls
three E-field components from particle-style variables and exposes
them to the pusher.

## Syntax

```
fix ID efield/particle <ExSrc> <EySrc> <EzSrc>
```

Each `*Src` is `v_<varname>` or `NULL`.

## Example

```
variable Ez0 particle 1e3      # 1 kV/m, axial
fix fep efield/particle NULL NULL v_Ez0
```

For production decks the per-cell `fix efield/grid` driven by
`compute plasma/fields` is the standard path.

## Files

- `src/OPENEDGE/fix_efield_particle.{h,cpp}`
