# `fix efield/particle`

Set the electric field seen by the pusher from per-particle variables.
Companion to [`fix bfield/particle`](bfield_particle.md); used by the
verification cases with analytic fields. Production decks obtain E from
`fix background`.

## Syntax

```text
fix ID efield/particle Ex Ey Ez
```

`Ex`, `Ey`, `Ez` name particle-style variables (no `v_` prefix) in V/m,
internal (x, y, z) slot order; `NULL` leaves a component unset.

## Example

```text
variable Ex particle 1.0e3
fix fe efield/particle Ex NULL NULL
fix fb bfield/particle NULL NULL Bz0
```

Together these drive the E×B polarization-drift verification
(`examples/verification/efield_polarization`, gates in
`plot_polarization.py`).
