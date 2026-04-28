# Plasma-native electric field

`compute plasma/fields` reads the E-field directly from the plasma code's
native potential at converter time (SOLPS `po`, SOLEDGE3X `zone*/PHI`,
OEDGE `osmns_efpara`). The converter computes `E = −∇ϕ` on the
B2 / SOLEDGE3X triangulation mesh and writes `/mesh/e_r, /mesh/e_z, /mesh/e_t`.

`compute plasma/fields` then reads those per-cell values at the SPARTA cell
centroid (via `findNearestMappedTriangle`) and emits `er`, `et`, `ez`, `ex`,
`ey` output columns.

**`epar` output** = `E · b̂` (dot product of the mesh-stored E vector with
`b̂` from equilibrium ψ).

## Converter status

| code | status |
|---|---|
| SOLPS | fully implemented (reads `balance.nc:po`, Jacobian FD on the B2 `(ix, iy)` grid) |
| SOLEDGE3X | writes `mesh/e_{r,z,t} = 0` placeholders. Zone-based `/zone*/PHI` resampling onto EIRENE triangle centroids is TODO. Prints WARNING at converter time. |
| OEDGE | mesh-output path pending; current builds emit zero E. Proper support comes with a future OEDGE → mesh migration. |

## Feeding into the Boris pusher

```
fix pd background file plasma.h5
compute cplasma plasma/fields all background pd ex ey ez
fix fE efield/grid c_cplasma[ex_col] c_cplasma[ey_col] c_cplasma[ez_col]
global efield grid fE 0
```
