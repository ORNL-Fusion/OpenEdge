# Wall geometry from SOLPS: `convert_solps_plasma.py --wall-source`

Three ways to build `wall.surf` and the per-segment → B2-cell mapping that
`fix emit/surf/recycle` consumes.

| `--wall-source` | source | segment→cell mapping | status |
|---|---|---|---|
| `mesh-extra` (default) | SOLPS `mesh.extra` user-defined wall, parsed into a watertight polygon | every B2 boundary face chooses nearest mesh.extra segment; per-segment area is the aggregate of those faces | **production**, SOLPS-native, no EIRENE files required |
| `b2` | B2 outer-boundary cell faces only | 1:1 segment↔cell | not yet implemented; intended for codes without `mesh.extra` (OEDGE, SOLEDGE3X) |
| `eirene` | `fort.33/34/35` triangulation boundary edges | nearest B2 face | only useful for direct standalone EIRENE cross-validation; requires EIRENE output files |

## Datasets written to plasma.h5

- `mesh/wall_face_area[ncell]` — per-B2-cell wall face area
- `mesh/wall_surf_cell[nseg]` — dominant B2 cell index per wall segment
- `mesh/wall_surf_area[nseg]` — per-segment aggregate area (SOLPS flux
  budget claimed by the segment)

The fix uses these to emit the correct Bohm flux at each wall segment
without any runtime geometric search.

## Triangulation extension to the wall

The EIRENE triangulation from `fort.33/34/35` stops at its "neighbour
polygon" that is typically a few mm to a few cm shy of the `mesh.extra`
wall. The converter re-triangulates the annulus between the EIRENE outer
boundary and the wall polygon using `scipy.spatial.Delaunay` on combined
vertices, keeping only triangles whose centroid sits inside the wall
**and** outside the EIRENE mesh. Each new triangle is projected to the
nearest B2 sheath cell (same cKDTree path used for vacuum/PFR tris).

Result: the mesh covers the full wall polygon; `mesh_cell_at()` never has
to fall back to the 5 cm `max_dist` nearest-triangle search.
