# `fix efield/grid` — apply per-cell E-field array to the pusher

Per-cell electric-field counterpart to
[`fix bfield/grid`](bfield_grid.md). Reads three per-cell E-field
columns (typically from `compute plasma/fields ex ey ez`) and exposes
them to the Boris pusher.

## Syntax

```
fix ID efield/grid <ExSrc> <EySrc> <EzSrc> [every <N>]
```

`ExSrc`, `EySrc`, `EzSrc` are per-cell `c_<compute>[col]` sources, in
SPARTA slot order. Use `NULL` for any axis you don't want set.

## Example

Plasma-native E from the converter:

```
fix pd background file plasma.h5
compute cpf plasma/fields all background pd ex ey ez
fix fE efield/grid c_cpf[ex_col] c_cpf[ey_col] c_cpf[ez_col]
```

After this, the Boris pusher applies `qE/m` per particle inside each
cell. See [`compute plasma/fields`](efield_plasma.md) for the
per-converter E-field provenance.

## Files

- `src/OPENEDGE/fix_efield_grid.{h,cpp}`
