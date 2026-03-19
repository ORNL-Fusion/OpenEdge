# OpenEdge Plasma/B-field Converters

Scripts to convert edge-plasma simulation output into the HDF5 format
(`plasma.h5` + `bfield.h5`) expected by OpenEdge's
`compute plasma/fields file` command.

## Available converters

| Script | Source code | Description |
|--------|-----------|-------------|
| `convert_solps_plasma.py` | SOLPS-ITER (`b2fgmtry`, `b2fstate`) | Reads SOLPS binary-text files directly (no quixote). Outputs plasma + B-field + multi-ion species + Eirene triangular mesh. |
| `convert_oedge_plasma.py` | OEDGE/DIVIMP (`.nc`) | Reads OEDGE NetCDF background file. Requires a companion `.equ` equilibrium file for B-field reconstruction. |
| `convert_s3x_plasma.py` | SOLEDGE3X (HDF5) | Reads SOLEDGE3X triangle-based mesh/data files. |
| `convert_solps_heatflux.py` | SOLPS-ITER | Extracts wall heat flux from SOLPS face-centred fluxes. |
| `geqdsk2bfield_h5.py` | G-EQDSK equilibrium | Converts a G-EQDSK file to `bfield.h5`. |
| `gen_bfield_sweep.py` | single equilibrium | Generates time-dependent B-field sequence via rigid radial shifts of ψ. |
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

### `bfield.h5`

| Dataset | Shape | Description |
|---------|-------|-------------|
| `r` | `(nr,)` | R coordinates [m] |
| `z` | `(nz,)` | Z coordinates [m] |
| `br` | `(nz, nr)` | Radial B-field [T] |
| `bt` | `(nz, nr)` | Toroidal B-field [T] |
| `bz` | `(nz, nr)` | Vertical B-field [T] |

## Quick-start examples

### SOLPS-ITER

```bash
python convert_solps_plasma.py /path/to/solps_run \
    --equ-file equilibrium.equ \
    --plasma-out plasma.h5 --bfield-out bfield.h5 \
    --nr 300 --nz 300 --plot
```

### OEDGE/DIVIMP

```bash
python convert_oedge_plasma.py d3d-204953-bkg-v25.nc \
    --equ-file 204953_3000.x16.equ \
    --plasma-out plasma.h5 --bfield-out bfield.h5 \
    --nr 300 --nz 300 --plot
```

### SOLEDGE3X

The SOLEDGE3X converter is called as a library function rather than a
CLI tool.  It requires four HDF5 input files from a SOLEDGE3X run:

```python
from convert_s3x_plasma import interpolate_and_save_plasma_field

interpolate_and_save_plasma_field(
    ref_file="refParam_raptorX.h5",       # reference parameters
    mesh_file="meshEIRENE.h5",            # Eirene triangular mesh
    bfield_file="mesh_raptorX.h5",        # B-field on mesh
    data_file="plasmaFinal.h5",           # plasma solution
    wall_file=None,                       # optional external wall
    plasma_out="plasma.h5",
    bfield_out="bfield.h5",
    nR=200, nZ=200,
    main_ion_spec=1,
    use_mesh_wall=True,
    wall_sparta_file="wall.txt",          # optional SPARTA wall output
    debug_plot_file="soledge_fields.png",
)
```

### G-EQDSK only (B-field)

```bash
python geqdsk2bfield_h5.py g123456.01000 \
    --out bfield.h5 --nr 300 --nz 300
```

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
