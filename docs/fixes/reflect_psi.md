# `fix reflect/psi` — inner-core ψ-boundary action

Particles that cross a normalised flux-surface threshold $\psi_n =
(\psi - \psi_\mathrm{axis})/(\psi_\mathrm{LCFS} - \psi_\mathrm{axis})$
are either reflected back to the SOL or absorbed (deleted). This is
how OpenEdge implements a soft inner-core boundary for impurity
transport runs without having to mesh the closed-flux-surface region.

## Syntax

```
fix ID reflect/psi geqdsk <file.geqdsk> background <plasma_fix> \
    psi_norm <value> [action reflect|absorb]
```

| keyword | meaning |
|---|---|
| `geqdsk <file>` | EFIT-style equilibrium file (alternative path for ψ when `fix background` was loaded without an `equilibrium` group) |
| `background <fix_id>` | ID of an existing `fix background`; ψ is taken from its loaded equilibrium when present |
| `psi_norm <value>` | threshold ψ_n. Particles with ψ_n < value are intercepted. Typical: 0.95–0.98. |
| `action reflect` (default) | mirror the radial velocity component about the local flux surface |
| `action absorb` | delete the particle on crossing |

For inner-core boundaries you usually want `action absorb` — sputtered
impurities entering the closed-flux region cannot escape on a fluid
timescale.

## Example

```
fix pd background file plasma.h5 equilibrium efit.geqdsk static yes
fix fcore reflect/psi geqdsk efit.geqdsk background pd \
    psi_norm 0.97 action absorb
```

## Note on inner-boundary surfaces

`fix reflect/psi` is an **alternative** to building an explicit
`core.surf` segment from `convert_solps_plasma.py --core-out`. The
explicit-surface path is preferred when it works (cleaner SPARTA
geometry); use `fix reflect/psi` when the ψ contour clips into the
divertor wall and SPARTA's flood-fill won't accept a combined
`wall.surf` + `core.surf` (the well-known
`Cell type mis-match when marking on self` error).

## Files

- `src/OPENEDGE/fix_reflect_psi.{h,cpp}`
- See [`converters/plasma_h5_schema`](../converters/plasma_h5_schema.md)
  for the GCA / equilibrium provenance.
