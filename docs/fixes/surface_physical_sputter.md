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
    [static yes|no] \
    [impurity <mass_amu> <frac> <Zmax> <f1> ... <fZmax>] \
    [bfield <file.h5>] [equilibrium <file.equ>] \
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
- **`erosion_flux`** — gross erosion flux Γ_ero [atoms·m⁻²·s⁻¹], summed
  over all projectile slots (+ optional virtual impurity).
- **`erosion_rate`** — Γ_ero × axisymmetric ring area [atoms·s⁻¹]. Use
  with `compute reduce sum` to get a single global erosion rate.

> **Legacy paths removed 2026-04-23.** The following pre-element-aware
> keywords and positional args are no longer accepted:
>
> - `file <plasma.h5>` mode — replaced by `background <fix_id>` only.
> - `eckstein <entry_name>` single-pair keyword.
> - positional `<surface.h5>` yield-table path.
> - `projectile_slots <lo> <hi>` — auto-derived from
>   `ion_species/elements` now.
> - `sputter_flux_total` / `sputter_rate_total` keywords — use
>   `erosion_flux` / `erosion_rate`.
>
> All decks that used those forms must migrate to
> `target <elem> projectiles <elem_list>` against a `fix background`.

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
  B-field projection onto the wall inward normal.
- Impact energy from Chankin 2014 eq. (4) (sheath drop + 2·Ti); impact
  angle from the B-field geometry.
- Sputter yield via the per-element Eckstein coefficient set indexed by
  `slot_to_table[s]`. Slots with `slot_to_table[s] < 0` (main ion that
  isn't in the projectile list, etc.) contribute zero yield but are
  still returned in `nflux_species`/`incident_*` diagnostic columns.
- Erosion contribution `Y_s · Γ_s`, summed across slots and the virtual
  impurity channel into `erosion_flux`.

## Internals

- `resolve_projectile_tables()` in `src/compute_surface_physical_sputter.cpp` runs
  once from `load_plasma_from_fix`. It expands `projectiles all`,
  looks up `Eckstein::lookup_sputter("<proj>_on_<target>")` for each
  element, populates the flat `per_proj_{Z1,M1,Z2,M2,Es,Eth,Q,ETF}`
  arrays, and fills `slot_to_table[0..pd->nion-1]`.
- Runtime per-slot dispatch (same source file, main compute loop): if
  `api_new && slot_to_table[s] >= 0`, build an `Eckstein::SputterParams`
  from the per-slot index and call `Eckstein::sputter_yield`. Otherwise
  fall back to the single-set legacy path (`eck_Z1…eck_ETF`) or the
  HDF5 table interpolator.
- Header members live in `src/compute_surface_physical_sputter.h`; both `src/` and
  `src/OPENEDGE/` copies are kept in sync (three-copy rule).

## Known limits

- **Eckstein-only per-projectile dispatch.** The element-aware API
  currently uses analytic Eckstein (`eckstein_sputter_data.h`) for each
  pair. Per-pair HDF5 tables (`database/surface/<proj>_on_<target>.h5`)
  are still only available via the legacy single-table arg at position
  5. Per-element HDF5 dispatch is the next extension.
- **TRIM sputter not wired yet.** `processes.h5:/surface/sputter/<pair>`
  holds axis grids but no `spyld` dataset in the current shipped
  database. Once ingested, adding a `trim` mode alongside `eckstein`
  will switch the default.
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
