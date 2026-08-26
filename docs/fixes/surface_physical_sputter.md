# `compute surface/physical/sputter` — wall incident flux, sputter yield, and erosion

Per-surface diagnostic computing incident plasma flux (Bohm), impact
energy / angle, sputter yield, and gross erosion flux for one or more
projectile elements against a common wall target material. Drives
`fix surface/emit/source` (sputtered-impurity source) and stands in for
EIRENE's SURFMOD recycling + sputter block.

## Element-aware syntax (post-2026-04-23)

```
compute <ID> surface/physical/sputter <surf_group> background <plasma_fix_ID> \
    target <wall_elem> \
    projectiles <elem_list|all> \
    [mass_amu <val>] \
    [incidence poloidal|full3d|directed3d] \
    [visibility <custom_surf_vector>] \
    [static yes|no] \
    [impurity <mass_amu> <frac> <Zmax> <f1> ... <fZmax>] \
    [iead auto|<file.h5>|none] \
    erosion_flux | erosion_rate | \
    nflux_species <slot|all> | incident_angle_species <slot|all> | \
    incident_energy_species <slot|all> | \
    sputter_yield_species <slot|all> | sputter_flux_species <slot|all>
```

- **`target <wall_elem>`** — wall material element symbol (`W`, `C`,
  `Be`, `Fe`, …). Combined with each entry in `projectiles` to build the
  Eckstein coefficient lookup name `<proj>_on_<target>` (e.g. `O_on_W`,
  `D_on_C`). Entries come from `src/eckstein_sputter_data.h`.
- **`projectiles <elem_list|all>`** — comma-separated list of projectile
  element symbols (`D,O,B`) or the keyword `all` to use every element
  present in `fix background`'s `/ion_species/elements`. Each element
  gets its own Eckstein parameter set at init; plasma ion slots are
  routed to the matching parameter set via an internal
  `slot_to_table[s]`, so a WEST-style multi-species plasma
  (D+, O¹⁺…O⁸⁺) picks up the right per-slot yield with a single compute
  instance.
- **`mass_amu <val>`** — background main-ion mass (amu). Used only for
  the sheath sound speed + Chankin impact energy. Default: 2.0 (D⁺).
- **`incidence poloidal|full3d|directed3d`** — magnetic projection used
  for the incident flux and geometric impact angle. `poloidal` is the
  backward-compatible default and evaluates `|Br nr + Bz nz|/|B|`.
  `full3d` evaluates `|B·n|/|B|`, including toroidal surface tilt.
  `directed3d` is one-sided: it uses each ion slot's signed parallel flow
  and evaluates `max(0, -sign(u∥) B̂·n)` for an outward wall normal.
  Thus a face tilted away from the arriving field-aligned ions receives
  zero direct source. Use `directed3d` for non-axisymmetric castellated
  targets when the background contains signed `ions/parr_flow` data.
- **`visibility <custom_surf_vector>`** — optional static field-line
  visibility fraction in `[0,1]` for each surface element. The compute
  multiplies every direct incident-flux channel by this value (`0` = fully
  shadowed, `1` = visible). The attribute must be a floating-point custom
  per-surface vector defined before the compute is initialized. This option
  is intended for a periodic, flow-directed offline ray trace; it is off by
  default.
- **`iead auto|<path>|none`** — fold the per-cell ion energy/angle
  distribution into the yield instead of evaluating Eckstein at the
  mean impact `Y(<E>, <θ>)`. `auto` resolves
  `database/iead/iead_database.h5` via `database_paths::resolve_iead_file`;
  for projectile element `W`, `database/iead/iead_database_W.h5` is
  used when present. `none` (default) keeps the legacy mean-impact
  path. See [Distribution-weighted yield (IEAD)](#distribution-weighted-yield-iead).
- **`erosion_flux`** — gross erosion flux Γ_ero [atoms·m⁻²·s⁻¹], summed
  over all projectile slots (+ optional virtual impurity).
- **`erosion_rate`** — Γ_ero × axisymmetric ring area [atoms·s⁻¹]. Use
  with `compute reduce sum` to get a single global erosion rate.

## Examples

### Single projectile on carbon wall

```
compute cpmi surface/physical/sputter wall background pd \
    target C projectiles D erosion_flux
```

Resolves Eckstein entry `D_on_C` and routes the D⁺ slot to it.

### WEST multi-projectile on tungsten

```
compute cpmi surface/physical/sputter wall background pd \
    target W projectiles D,O erosion_flux
```

Loads Eckstein entries `D_on_W` and `O_on_W`. Plasma ion slots with
element `D` are routed to `D_on_W`; slots with element `O` (all charge
states) are routed to `O_on_W`.

### All plasma species automatically

```
compute cpmi surface/physical/sputter wall background pd \
    target W projectiles all erosion_flux
```

Pulls unique element set from `plasma.h5 /ion_species/elements`
(preserving order), errors at init if any `<proj>_on_W` Eckstein entry
is missing — listed explicitly in the error message.

### Per-species diagnostic output

```
compute cpmiO_diag surface/physical/sputter wall background pd \
    target W projectiles O \
    nflux_species all \
    incident_angle_species all \
    incident_energy_species all \
    sputter_yield_species all \
    sputter_flux_species all \
    erosion_flux
```

Returns one column per plasma ion slot for each species-level quantity,
plus the total. Use with `dump surf` to get a full PMI breakdown.

### Virtual impurity (not in plasma.h5)

```
compute cpmi surface/physical/sputter wall background pd \
    target W projectiles D \
    impurity 10.81 0.03 5 0.05 0.15 0.30 0.30 0.20 \
    erosion_flux
```

Adds a synthesised 3 % B at the wall (5 charge states, given fractions)
on top of the D projectile from plasma.h5. The impurity uses the same
yield-lookup path as the projectiles (currently single analytic
Eckstein set; the impurity element tag for per-pair resolution is
future work — see Known limits).

## Physics

For each (plasma slot `s`, wall surface `m`):

- Incident density from `fix background` at the wall midpoint, routed
  through the axi-aware `OpenEdge::sparta_to_RZ` helper.
- Bohm flux `Γ_s = n_s · c_s · sin(α_B)`, with
  `c_s = √((Te+Ti)/m_bg)` (`m_bg` = `mass_amu`) and `sin(α_B)` the
  selected B-field projection onto the wall normal. In `directed3d`
  mode, signed per-species `u∥` selects only the incoming field direction.
- Impact energy from Chankin 2014 eq. (4) (sheath drop + 2·Ti); impact
  angle from the B-field geometry.
- Sputter yield via the per-element Eckstein coefficient set indexed by
  `slot_to_table[s]`. Slots with `slot_to_table[s] < 0` (main ion that
  isn't in the projectile list, etc.) contribute zero yield but are
  still returned in `nflux_species`/`incident_*` diagnostic columns.
- Erosion contribution `Y_s · Γ_s`, summed across slots and the virtual
  impurity channel into `erosion_flux`.

### Offline visibility-mask interface

A ray tracer can hand a static mask to OpenEdge through the standard
custom-surface file format. The first non-comment line is `<N> 1`, followed
by exactly `N` rows of `<global_surf_id> <visibility_fraction>`:

```text
# field_visibility.dat
9358 1
1 1.0
2 0.0
...
```

Load it after all surfaces and groups have been read, but before the compute
is first initialized:

```text
custom surf create field_visible float 0 &
            file input/field_visibility.dat 1 field_visible
compute cpmi surface/physical/sputter wactive background pd \
            target W projectiles D,O incidence directed3d \
            visibility field_visible static yes erosion_flux
```

For reproducibility, the ray-trace sidecar should record the geometry and
plasma SHA-256 hashes, periodic pitch/transform, start-point epsilon, maximum
trace distance/wrap count, and flow/B convention. Its diagnostic CSV should
include at least `surf_id`, centroid, outward normal, local B, `u_parallel`,
`Bhat_dot_n`, visibility, first occluder ID, distance, and periodic wraps.

## Internals

- `resolve_projectile_tables()` in `src/compute_surface_physical_sputter.cpp` runs
  once from `load_plasma_from_fix`. It expands `projectiles all`,
  looks up `Eckstein::lookup_sputter("<proj>_on_<target>")` for each
  element, populates the flat `per_proj_{Z1,M1,Z2,M2,Es,Eth,Q,ETF}`
  arrays, and fills `slot_to_table[0..pd->nion-1]`.
- Runtime per-slot dispatch (same source file, main compute loop):
  build an `Eckstein::SputterParams` from the per-slot index and call
  `Eckstein::sputter_yield`.
- Header members live in `src/compute_surface_physical_sputter.h`; both `src/` and
  `src/OPENEDGE/` copies are kept in sync (three-copy rule).

## Distribution-weighted yield (IEAD)

When `iead auto` (or an explicit table path) is supplied, the per-slot
yield is the convolution

```
<Y>(τ, ψ, Z, Te) = Σ_{Ẽ, θ} Y_eckstein(Ẽ·Z·Te, θ) f(Ẽ, θ; τ, ψ, Z)
```

where `f` is the per-cell PDF on `Ẽ = E/(Z·Te)` × `θ` bins shipped
in `database/iead/iead_database.h5`. See `database/iead/README.md`
for the schema and the SHEATH repo `docs/iead_normalization.tex` for
the normalization derivation.

### Why it matters at grazing incidence

The legacy mean-impact path uses `θ_impact = ψ_geom = 90° − α_B`,
which assumes the projectile arrives along the magnetic-field
direction with no Larmor rotation in the magnetised presheath. This
is only correct for cold, light projectiles. In real divertor
conditions ψ_geom is typically 87–89° (very grazing) but the actual
impact-angle distribution sits at θ ≈ 60–80° (the Larmor radius
decouples the projectile from the field line inside the MPS, see
Mellet 2017 fig 7). Yamamura's angular factor `Y(θ)/Y(0)` is
exponentially nonlinear:

| θ_impact     | Yamamura factor |
|--------------|-----------------|
| 60°          | ~3              |
| 75°          | ~1              |
| 85°          | ~0.6            |
| 88°          | ~5×10⁻³         |
| 89°          | ~7×10⁻¹⁰        |

so feeding `θ = ψ_geom ≈ 88°` instead of `<θ>_impact ≈ 70°` under-
estimates the per-particle yield by 2–4 orders of magnitude. With
`iead auto`, the convolution sees the full distribution and recovers
the correct value. **Expect a step-change up in `erosion_flux` /
`erosion_rate` when switching legacy decks to `iead auto`** — this is
physics, not a regression.

### Performance

The per-surf inner loop adds one (`n_E × n_θ ≈ 3600`) Eckstein-yield
evaluation per `(surf, projectile-slot)` per `compute_per_surf` call.
For a typical case (~500 surfs × ~5 slots) that is ~9×10⁶ yield
evaluations per call. With `static yes` the result is computed on
first call and reused, so the cost is one-shot.

### Projectile mass classes

Two tables are shipped:

- **light** (`iead_database.h5`) — `m_proj = 2·Z amu` (D-scaling),
  `Z = 1..10`. Bias < 15 % for D, He, Li, B, C, N, O, F, Ne. Used by
  default for any projectile element with `Z_atomic ≤ 18`.
- **W** (`iead_database_W.h5`, optional) — `m_proj = 184 amu`,
  `Z = 1..20`. Used when a projectile element is `W`. If the file is
  absent the code falls back to the light table for W with a warning
  (Larmor angle distribution will be off; energies still ~OK because
  `<E>/Z·Te` ≈ φ̂_tot ≈ 2.86 is mass-independent).

## Known limits

- **Direct incidence is not line-of-sight visibility.** `directed3d`
  rejects an outward-facing local field direction, but it does not trace
  the upstream field line through neighboring triangles. A separate
  periodic visibility mask, supplied with `visibility`, is required to
  represent geometric occlusion by an upstream monoblock.

- **Eckstein-only per-projectile dispatch.** Yields use analytic
  Eckstein from `src/OPENEDGE/eckstein_sputter_data.h`. Per-pair TRIM
  HDF5 tables in `processes.h5:/surface/sputter/<pair>` carry axis
  grids but no `spyld` dataset yet; once ingested a `trim` mode will
  switch the default.
- **Single wall material per compute.** For heterogeneous walls
  (W divertor + C main chamber) spawn one `compute surface/physical/sputter` per
  surf group. A future `targets divertor=W main=C` syntax is deferred.
- **Missing Eckstein entry is fatal at init** (by design — silent
  zero-yield invited wrong answers). The error message lists every
  missing `<proj>_on_<target>` pair. Drop the offending element from
  `projectiles` or add the coefficients to `eckstein_sputter_data.h`.

## Related

- `fix surface/emit/source` — consumes `erosion_flux` as the source
  rate for sputtered impurity neutrals.
- `database/processes.h5` — consolidated surface (reflection, sputter)
  and volume (ADAS) data. Sputter tables populated in a later step.
- `src/eckstein_sputter_data.h` — analytic Eckstein coefficient table
  for named `<proj>_on_<target>` entries.
