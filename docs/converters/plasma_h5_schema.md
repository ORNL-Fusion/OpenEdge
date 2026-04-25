# `plasma.h5` schema (mesh-only, post-2026-04-20)

`convert_solps_plasma.py`, `convert_s3x_plasma.py`, and
`convert_oedge_plasma.py` emit a single mesh-only HDF5 with three top-level
groups and nothing else.

| group | purpose | typical shape |
|---|---|---|
| `/equilibrium/{r, z, psi, btf, rtf, psib}` | ψ map + toroidal axis params | ψ: `(km, jm)` ≈ `(257, 257)` |
| `/ion_species/{names, elements, spec_index, main_ion_spec_index, mass_amu, charge_state_z}` | per-species metadata | 1D |
| `/mesh/{vtx_r, vtx_z, triangles, cell_index, dens_e, temp_e, dens_i, temp_i, parr_flow, ions/{dens, temp, parr_flow}, wall_face_area, wall_surf_cell, wall_surf_area}` | EIRENE triangulation + per-B2-cell plasma + wall geometry | `vtx*, tri*`: ~5–10k; `dens_e` etc.: ~3k (ncell) |

**Removed:** all top-level regular-grid datasets (`r`, `z`, `dens_e`,
`temp_e`, `br`, `bt`, `bz`, `grad_*`, `ions/*`, `n_e/*`, `n_i/*`). File
size for the DIII-D `run_lore2023` case dropped from ~21 MB (with the
regular grid) to ~1.1 MB (mesh-only).

## `/ion_species/elements` (added 2026-04-21)

Per-species element symbol with the charge-state suffix stripped:

| `names` | `elements` |
|---|---|
| `D+` | `D` |
| `O+` … `O8+` | `O` |
| `W5+` | `W` |

Used by `compute surface/physical/sputter target <elem> projectiles <elem_list>` to
aggregate plasma ion slots by element and resolve the corresponding
`<proj>_on_<target>.h5` surface file. Falls back to parsing `names` (strip
trailing digits + `±` characters) when `elements` is absent — so legacy
plasma.h5 files produced by pre-2026-04-21 converters still work.

## Query pattern

All plasma queries route through `fix background`:

```
fix pd background file input/plasma.h5
# Per-particle ne/Te/Ti/B via pd->query_plasma_at_point(x)
# Per-cell via pd->mesh_ne[cell], mesh_te[cell], ...
```

The `compute plasma/fields file <plasma.h5>` mode has been removed and
hard-errors on use. Declare `fix background` then reference it from
`compute plasma/fields all background <fix_id> …`.

## GCA requires equilibrium

The GCA pusher uses analytic B-field derivatives from ψ
(`B_R = −(1/R) ∂ψ/∂Z` etc.), not numerical finite differences on a grid.
If `plasma.h5` does not carry `/equilibrium/{r,z,psi,btf,rtf,psib}` and
no `equilibrium <file>` keyword is provided on the compute line,
`global gca` init aborts with a clear error.

SOLEDGE3X cases must supply the `.equ` separately since `mesh.h5` does not
expose `btf`/`rtf` (the converter writes `psi_axis = psicore` so
`fix reflect/psi` still works without `btf/rtf`, but GCA itself needs the
full equilibrium).

## Gradient fields: mesh-only caveat

`compute plasma/fields` currently returns zero for `grad_te_{r,z}`,
`grad_ti_{r,z}`, `grad_ne_{r,z}` on mesh-only plasma.h5. Consumers that
need gradients (`fix thermal_force`, `fix cross_diffusion`) should migrate
to per-particle FD queries against `fix background`. Known gap in the
mesh-only transition.
