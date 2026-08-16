# `fix bfield/particle`

Set the magnetic field seen by the pusher from per-particle variables.
This is the idealized-field harness used by the verification cases
(analytic fields with a known solution); production runs take their field
from `fix background`.

## Syntax

```text
fix ID bfield/particle Bx By Bz
```

`Bx`, `By`, `Bz` name particle-style variables (referenced without the
`v_` prefix) evaluated per particle each step, in Tesla and internal
(x, y, z) slot order. Use `NULL` to leave a component unset (zero).

## Example

```text
variable Bx particle 0.0
variable By particle 0.0
variable Bz particle 1.0
fix fb bfield/particle Bx By Bz
```

See `examples/verification/efield_polarization` for a complete case with
pass/fail gates.
