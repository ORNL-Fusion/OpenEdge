# `fix thermal_force` — Braginskii thermal forces on impurity ions

Applied as leapfrog half-kicks (`START_OF_STEP` + `END_OF_STEP`).

## Syntax

```
fix ID thermal_force Nevery \
    bfield BxSRC BySRC BzSRC \
    [ion_thermal gradTiR_SRC gradTiZ_SRC [coeff VAL]] \
    [elec_thermal gradTeR_SRC gradTeZ_SRC [coeff VAL]]
```

## Physics

- **Ion thermal force**: `F = β_i · Z² · e · grad_par(Ti)` (default
  `β_i = 2.6`, Neu 1974 heavy-impurity limit).
- **Electron thermal force**: `F = α_e · Z² · e · grad_par(Te)` (default
  `α_e = 0.71`, Braginskii `Z_eff = 1` limit).

Both push impurities toward higher temperature (toward the core).

## Source conventions

- **B-field sources** (`bx`, `by`, `bz`) must be in SPARTA coordinate order,
  matching the velocity slot mapping. `compute plasma/fields` does the
  projection automatically — pipe its columns in directly.
- **Temperature gradient sources** are always cylindrical (`grad_ti_r`,
  `grad_ti_z`).

## Mesh-only plasma.h5 caveat (2026-04-20+)

`compute plasma/fields` currently returns zero for `grad_te_{r,z}`,
`grad_ti_{r,z}` when the plasma is mesh-only. Consumers should migrate to
per-particle finite-difference queries against `fix plasma/data`
(`pd->query_plasma_at_point(R ± dR, Z)` then subtract). Known gap in the
mesh-only transition.
