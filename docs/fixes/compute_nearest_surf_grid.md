# `compute nearest_surf/grid` — distance + index of nearest wall surf per cell

Per-grid-cell record of the nearest surface element (line in 2D, tri
in 3D), the Euclidean distance to it, and the surface normal at the
nearest point. Used as a building block for sheath thickness, heat-
flux footprint, and impurity source-localisation diagnostics.

## Syntax

```
compute ID nearest_surf/grid <surf_group> <grid_group> [keyword ...]
```

| keyword | meaning |
|---|---|
| `dist` | Euclidean distance from cell centroid to the nearest surf |
| `surfid` | global surf ID of the nearest segment |
| `surfidx` | local-surf-array index (for fast subsequent lookups) |
| `nx`, `ny`, `nz` | components of the surface normal at the nearest point |

Pick the keywords you need; the compute returns one column per
keyword, in the listed order.

## Example

```
compute cnear nearest_surf/grid wall all dist surfid nx ny nz
dump dgrid grid all 1000 nearest.txt id c_cnear[1] c_cnear[2] \
     c_cnear[3] c_cnear[4] c_cnear[5]
```

This dumps the wall-distance map and the surface normal pointing into
the plasma. Combined with `fix volume/chem/adas` rate suppression near
the wall (Boltzmann factor through the sheath), it gives a clean
near-wall ionisation profile.

## Files

- `src/OPENEDGE/compute_nearest_surf_grid.{h,cpp}`
- See `docs/fixes/sheath.md` for the consumer side that uses
  `nearest_surf` to locate the sheath edge per cell.
