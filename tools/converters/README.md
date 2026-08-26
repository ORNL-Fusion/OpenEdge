# OpenEdge Plasma/B-field Converters

Scripts to convert edge-plasma simulation output into the `plasma.h5`
HDF5 format consumed by `fix background`. B-fields travel inside
plasma.h5 (`mesh/vtx_b*` on the unstructured mesh plus the embedded
`/equilibrium` group) — the legacy standalone `bfield.h5` raster is gone.

## Available converters

| Script | Source code | Description |
|--------|-----------|-------------|
| `convert_solps_plasma.py` | SOLPS-ITER (`b2fgmtry`, `b2fstate`) | Reads SOLPS binary-text files directly (no quixote). Outputs plasma + B-field + multi-ion species + Eirene triangular mesh. |
| `convert_oedge_plasma.py` | OEDGE/DIVIMP (`.nc`) | Reads OEDGE NetCDF background file. Requires a companion `.equ` equilibrium file for B-field reconstruction. |
| `convert_s3x_plasma.py` | SOLEDGE3X (HDF5) | Reads SOLEDGE3X zone/triangle data and native staggered wall fluxes. |
| `convert_solps_heatflux.py` | SOLPS-ITER | Extracts wall heat flux from SOLPS face-centred fluxes. |
| `gen_plasma_sweep.py` | single plasma.h5 | Generates plasma parameter sweeps. |
| `create_surf_from_solps.py` | SOLPS mesh | Extracts wall geometry from SOLPS mesh for SPARTA surface files. |

## Output format

### `plasma.h5`

| Dataset | Shape | Description |
|---------|-------|-------------|
| `r` | `(nr,)` | R coordinates [m] |
| `z` | `(nz,)` | Z coordinates [m] |
| `dens_e` | `(nz, nr)` | Electron density [m⁻³] |
| `temp_e` | `(nz, nr)` | Electron temperature [eV] |
| `dens_i` | `(nz, nr)` | Ion density [m⁻³] |
| `temp_i` | `(nz, nr)` | Ion temperature [eV] |
| `parr_flow` | `(nz, nr)` | Parallel flow speed [m/s] |
| `parr_flow_r` | `(nz, nr)` | Parallel flow, R component [m/s] |
| `parr_flow_t` | `(nz, nr)` | Parallel flow, toroidal component [m/s] |
| `parr_flow_z` | `(nz, nr)` | Parallel flow, Z component [m/s] |
| `grad_te_r/t/z` | `(nz, nr)` | Electron temperature gradient components [eV/m] |
| `grad_ti_r/t/z` | `(nz, nr)` | Ion temperature gradient components [eV/m] |
| `ion_species/*` | varies | Multi-ion metadata (names, mass, charge) |
| `ions/dens` | `(nspec, nz, nr)` | Per-species ion density |
| `ions/temp` | `(nspec, nz, nr)` | Per-species ion temperature |
| `ions/parr_flow*` | `(nspec, nz, nr)` | Per-species parallel flow + components |
| `wall_flux/gamma_i` | `(nspec, nwall)` | Native per-species particle flux into the source wall [m⁻² s⁻¹] |
| `wall_flux/r,z` | `(nwall,)` | Wall-flux sample positions [m] |

## Quick-start examples

### SOLPS-ITER

```bash
python convert_solps_plasma.py /path/to/solps_run \
    --equ-file equilibrium.equ \
    --plasma-out plasma.h5 \
    --nr 300 --nz 300 --plot
```

### OEDGE/DIVIMP

```bash
python convert_oedge_plasma.py d3d-204953-bkg-v25.nc \
    --equ-file 204953_3000.x16.equ \
    --plasma-out plasma.h5 \
    --nr 300 --nz 300 --plot
```

### SOLEDGE3X

Pass the SOLEDGE3X run directory to the CLI. A fluid-neutral run can be
converted directly from `mesh.h5`, `metric_raptorX.h5`, `plasmaFinal.h5`,
and `refParam_raptorX.h5`; no SOLEDGE3X post-treatment package is required.

```bash
python convert_s3x_plasma.py /path/to/s3x_run \
    --plasma-snapshot plasmaFinal.h5 \
    --plasma-out plasma.h5 \
    --wall-out wall.surf \
    --wall-flux required
```

With `--wall-flux auto` (the default), the converter extracts
`zone*/spec*/fluxn/{psi,theta}` on plasma/material interfaces, removes the
SOLEDGE3X face metric using `metric_raptorX.h5`, and writes the common
`/wall_flux` schema. `required` makes missing face-flux inputs an error;
`off` disables extraction. Face midpoints are projected onto the exact
source wall before OpenEdge maps them onto its own surface geometry.

## Dependencies

- Python 3.7+
- `numpy`, `scipy`, `h5py`
- `netCDF4` (OEDGE converter only)
- `matplotlib` (optional, for `--plot`)
- `freeqdsk` (GEQDSK reader, optional — only if using `--gfile` in SOLPS converter)

## Notes

- All converters reconstruct the poloidal B-field (Br, Bz) from an equilibrium
  file (`.equ` or G-EQDSK) using `Br = -(1/R) ∂ψ/∂Z`, `Bz = (1/R) ∂ψ/∂R`,
  `Bt = F/R`. The OEDGE `.nc` file only stores the toroidal field `BTS`, so a
  companion equilibrium file is always required.
- The OEDGE converter projects OEDGE's parallel gradients (`TEGS`, `TIGS`) and
  parallel flow (`KVHS`) into cylindrical (R, φ, Z) components using the B-field
  unit vector from the equilibrium.
- Scattered cell-centre data is interpolated onto the regular grid using
  `scipy.interpolate.griddata` (linear + nearest-neighbour fill).
