# `fix force/thermal` — Braginskii thermal forces on impurity ions

Applies the ion and electron thermal-force terms to every impurity
particle. The leapfrog half-kick pattern (`START_OF_STEP` +
`END_OF_STEP`) keeps the Boris push symplectic.

## Syntax

Two modes depending on how B-field and temperature gradients are
sourced.

### Mode A — `background` (recommended, mesh-native)

```
fix ID thermal_force Nevery background <plasma_fix_ID> \
    [ion_thermal yes|no] \
    [elec_thermal yes|no]
```

Reads B and `grad_Te / grad_Ti` from a `fix background` instance. B
comes from the mesh-native per-triangle field
(`/mesh/vtx_b{r,z,t}`); gradients come from the per-cell
`/mesh/grad_t{e,i}_{r,z}` datasets written by the post-2026-04-22
converters. No per-particle variables need to be pre-populated.

`ion_thermal` / `elec_thermal` accept `yes`/`no` to enable/disable each
channel independently. Default: both on if the keyword is present, off
if the keyword is absent.

### Mode B — legacy explicit sources

```
fix ID thermal_force Nevery \
    bfield BxSRC BySRC BzSRC \
    [ion_thermal gradTiR_SRC gradTiZ_SRC] \
    [elec_thermal gradTeR_SRC gradTeZ_SRC]
```

Each `*_SRC` is a variable or compute reference (e.g.
`c_pfields[1]`). B sources are in **SPARTA slot order** — use
`compute plasma/fields` to auto-project into the right columns for
the current coordinate layout (axi vs Cart). Gradient sources are
always cylindrical (`grad_ti_r`, `grad_ti_z`).

## Physics

- **Ion thermal force** (Neu 1974, heavy-impurity limit):
  `F = β_i · Z² · e · grad_par(Ti)`, `β_i = 2.6`.
- **Electron thermal force** (Braginskii, `Z_eff = 1`):
  `F = α_e · Z² · e · grad_par(Te)`, `α_e = 0.71`.

Both point toward higher temperature → toward the core. `grad_par` is
the component of the gradient along the local magnetic field.

Both coefficients are hardcoded (`β_i = 2.6`, `α_e = 0.71`) — not yet
exposed as keywords. If you need different coefficients, edit
`src/OPENEDGE/fix_thermal_force.cpp`.

## When to use `nevery > 1`

The force evaluation cost is dominated by the per-particle
`pd->bfield_at()` / `pd->interp2D(grad_*)` calls. Running `nevery 100`
(re-evaluate every 100 steps) amortises that for ~100× cost savings,
but if the gradient is large the accumulated impulse per evaluation
can become comparable to a cell size — risking wall-overshoot where
the subcycle ends with a particle past the wall mesh. Typical safe
settings:

- Low grad (bulk SOL): `nevery 100`.
- Strong grad (near-separatrix, X-point proximity): drop to
  `nevery 20` or add more Boris subcycles
  (`global boris_subcycles 10`).

If the particle dump shows W ions outside the wall polygon (see
`docs/migration/axi_cookbook.md` for the check), thermal-force
overshoot is the usual culprit. The fix does not yet apply a force
cap — that's a follow-up.

## Gradient source compatibility

| plasma.h5 layout | Mode A (`background`) | Mode B (explicit srcs) |
|---|---|---|
| mesh-only (new converters, 2026-04-22+) | ✅ reads `mesh/grad_te_{r,z}` etc. via `pd->mesh_grad_*` | ❌ `compute plasma/fields` returns 0 for gradients on mesh-only |
| legacy regular-grid | ✅ fallback to `pd->interp2D` on `grad_te` | ✅ `compute plasma/fields` populates columns |

**Bottom line:** prefer Mode A everywhere. Mode B is kept only for
old decks that haven't migrated to `fix background`.

## Related

- `fix cross_field_diffusion` — anomalous perpendicular diffusion
  (commonly paired to represent turbulent mixing).
- `fix coulomb/binary` — Coulomb pitch-angle scatter; the Chapman-Enskog
  closure that produces the β_i, α_e coefficients assumes a
  collisional background.
- `compute plasma/fields` — legacy B-field projector (Mode B). For
  new decks, `global bfield_compute <pd>` + `fix background` is
  sufficient.
