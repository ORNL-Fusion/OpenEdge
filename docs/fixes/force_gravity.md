# `fix force/gravity` — gravity force on droplets

Adds a uniform gravitational acceleration to all particles (or
restricted to a particle group via the `mixture` machinery on the
upstream side). Used by the droplet pipeline for trajectory
calculations where the droplet weight competes with drag and lift
forces.

## Syntax

```
fix ID force/gravity <group-ID> <gx> <gy> <gz>
```

| arg | meaning |
|---|---|
| `<group-ID>` | particle group / mixture ID. Use `all` for every particle. |
| `<gx>, <gy>, <gz>` | acceleration vector $[\mathrm{m}/\mathrm{s}^2]$ in **physical (R, Z, φ)** order, not internal (x, y, z) slot order. The fix routes through `RZphi_force_to_sparta()` so the same input deck works for axi / 2D-Cart / 3D layouts. |

## Example

Vertical gravity on droplets:

```
fix fg force/gravity dropletgroup 0 -9.81 0
```

For 2D axisymmetric `(x = Z, y = R, z = φ)` this is correctly mapped
to a `−Z`-direction acceleration; for 3D Cartesian it is applied along
the Cartesian Y axis.

## Files

- `src/OPENEDGE/fix_force_gravity.{h,cpp}`
