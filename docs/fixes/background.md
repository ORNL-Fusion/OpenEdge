# `fix background`

## Description

This fix defines the plasma background used by plasma-aware OpenEdge
fixes and computes. It can load a converter-generated `plasma.h5` file
or define a constant analytic background for tests and benchmarks.

The fix provides electron and ion density, temperature, flow, electric
field, magnetic field, and optional equilibrium `psi` data. Other
OpenEdge commands consume this data either through per-cell arrays or
through point queries at particle locations.

## Syntax

```
fix ID background \
    file <plasma.h5> [equilibrium <file.equ>] [static yes|no]
```

The most common usage. Less common forms accept explicit per-component
sources for testbench cases:

```
fix ID background constant \
    [r_bounds Rlo Rhi] [z_bounds Zlo Zhi] \
    dens_e VAL temp_e VAL dens_i VAL temp_i VAL \
    [upar VAL | upar_r VAL upar_t VAL upar_z VAL] \
    [grad_te_r VAL grad_te_t VAL grad_te_z VAL] \
    [grad_ti_r VAL grad_ti_t VAL grad_ti_z VAL] \
    [epar VAL] [br VAL bz VAL]
```

| keyword | meaning |
|---|---|
| `file <plasma.h5>` | HDF5 file from a converter (`/mesh/*`, `/equilibrium/*`, `/ion_species/*`) |
| `equilibrium <file.equ>` | optional standalone equilibrium file overriding `/equilibrium/` group |
| `static yes\|no` | if `yes`, plasma data is loaded once at init and never reread (right choice for SOLPS-coupled outer-loop runs). Default `no`. |
| `column_axis <x0> <y0>` | (3D Cartesian only) sets the (x, y) position of the axisymmetric plasma column axis. Default `(0, 0)`. |
| `constant` | declare the plasma analytically — useful for unit tests and benchmarks |

`column_axis` applies only to 3D Cartesian runs. It controls the
SPARTA-to-cylindrical mapping `R = sqrt((x - x0)^2 + (y - y0)^2)` used
for every plasma / B-field / sputter query. Default `(0, 0)` preserves
SOLPS / SOLEDGE3X behavior where the simulation box is centered on the
column. Use it for linear-device cases (MPEX, proto-lite) whose box is
not centered on the column. The keyword is ignored in 2D and
axisymmetric modes.

## Output provided

- **Per-cell arrays** populated at init / `reload_plasma()`:
  `mesh_ne[icell]`, `mesh_te[icell]`, `mesh_ti[icell]`,
  `mesh_upar[icell]`, `mesh_grad_te_{r,z}[icell]`, …,
  `mesh_b{r,z,t}[icell]`, `mesh_e{r,z,t}[icell]`.
- **Per-particle point queries** routed through
  `pd->query_plasma_at_point(x, &ne, &te, &ti, ...)` and
  `pd->query_bfield_at_point(x, &Br, &Bz, &Bt)`. Used by every Boris /
  GCA / sheath / sputter / chemistry consumer.
- **Equilibrium ψ** (when present): `B_R = (1/R)∂ψ/∂Z`,
  `B_Z = -(1/R)∂ψ/∂R`, `B_φ = btf·rtf/R`. Required by the GCA pusher
  and by `fix reflect/psi`.
- **Ion species metadata** at `/ion_species/{names, elements,
  charge_state_z, mass_amu}` — read by
  `compute surface/physical/sputter` for per-element yield routing.

## Examples

```
fix pd background file plasma.h5 static yes
compute cplasma plasma/fields all background pd ne te ti er ez et br bz bt
fix fcoul coulomb/background 1 background pd
```

Constant background for a unit test:

```
fix pd background constant temp_e 20 dens_e 1e19 temp_i 20 dens_i 1e19 \
    br 0.0 bz 5.0
```

## Notes

- Use `file plasma.h5` for production tokamak runs driven by converted
  edge-plasma data.
- Use `constant` for reduced test problems where a full converter output
  is unnecessary.
- When multiple downstream commands need plasma access, this fix is
  usually the single source of truth for the deck.

## Files

- `src/OPENEDGE/fix_background.{h,cpp}` — implementation
- `src/OPENEDGE/plasma_h5_loader.{h,cpp}` — schema-aware HDF5 reader
- See [`plasma_h5_schema`](../converters/plasma_h5_schema.md)
  for the on-disk layout.
