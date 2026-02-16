# OpenEdge Droplet Evaporation Model and HDF5 I/O

This document describes the **currently implemented** model in `fix_evaporation.cpp` and the HDF5 layouts expected by OpenEdge readers.

## 1) Evaporation model currently solved (`fix evaporation`)

Source: `fix_evaporation.cpp` (`FixEvap::droplet_evaporation_model`, `FixEvap::evap_half`).

### State variables per droplet
- Radius: `r_d` (m)
- Temperature: `T_d` (K)
- Mass: `m_d` (kg)

### Heat input
- `Q_s` (W/m^2), from either:
  - `heatflux/constant <value>`
  - `heatflux/file <heatflux.h5>` (interpolated at droplet `(R,Z)`)

### Evaporation flux and ODEs
The implementation uses:
- Antoine-like vapor pressure fit:
  - `vpres = 760 * 10^(a1 + b1/T)` with `a1=5.055`, `b1=-8023`
- Atom evaporation flux:
  - `G_evap = 1e4 * 3.513e22 * vpres / sqrt(xm1*T)` (atoms/(m^2 s)), `xm1=6.939`
- Radius evolution:
  - `dr_d/dt = -(AM/Rho) * G_evap`
- Energy balance used in code:
  - `HF = Q_s - G_evap*(DHm/AN)`
  - `dT_d/dt = (3/(Rho*Cp*r_d)) * HF`

where constants in the current implementation are:
- `AM = 1.53e-26` kg/atom
- `Rho = 534` kg/m^3
- `Cp = 4200` J/(kg K)
- `DHm = 3.158e3` J/mol
- `AN = 6.022e23` 1/mol

### Time integration actually applied
At timesteps where `ntimestep % nevery == 0`:
- One half update at `START_OF_STEP`: `dt/2`
- One half update at `END_OF_STEP`: `dt/2`

So total evaporation advance per active step is `dt`.

### Removal criterion (90% mass loss)
After update, droplets are removed when:
- `m <= 0.1 * m0_ref`
- `m0_ref = set_mass` if provided, otherwise `species.mass`

Removal is done by marking particle for deletion and calling `compress_rebalance()`.

### Current assumptions/limitations
- Evaporation model is implemented for **2D geometry only**.
- If interpolated heat flux is non-finite or negative, it is clamped to zero.
- If `Q_s <= 0`, droplet state is left unchanged for that substep.

## 2) `plasma.h5` expected layout (`compute plasma/fields ... file plasma.h5 bfield.h5 ...`)

Source: `compute_plasma_fields.cpp` (`ComputePlasmaFields::readPlasmaFileData`).

### Required 1D coordinates
- `r` : shape `(nr,)`
- `z` : shape `(nz,)`

### Required 2D fields (all required)
Each must have shape `(nz, nr)`:
- `dens_e`
- `temp_e`
- `dens_i`
- `temp_i`
- `parr_flow`
- `parr_flow_r`
- `parr_flow_t`
- `parr_flow_z`
- `grad_te_r`
- `grad_te_t`
- `grad_te_z`
- `grad_ti_r`
- `grad_ti_t`
- `grad_ti_z`

If any required dataset is missing or has wrong shape, reader throws.

### Optional multi-ion extension
Optional metadata:
- `ion_species/spec_index` : `(ns,)` int
- `ion_species/charge_state_z` : `(ns,)` int
- `ion_species/mass_amu` : `(ns,)` double
- `ion_species/names` : `(ns,)` string

Optional 3D ion fields (shape `(*, nz, nr)`):
- `ions/dens`
- `ions/temp`
- `ions/parr_flow`
- `ions/parr_flow_r`
- `ions/parr_flow_t`
- `ions/parr_flow_z`

For multi-ion fields, `nz` and `nr` must match `r,z` and all provided 3D ion arrays must share the same species dimension.

## 3) `heatflux.h5` expected layout (`fix evaporation ... heatflux/file`)

Source: `fix_evaporation.cpp` (`FixEvap::readHeatFlux`).

### Supported coordinate layouts
One of:
1. Legacy 1D coords
- `grid/Rc` : `(nr,)`
- `grid/Zc` : `(nz,)`

2. 2D coords under `grid/`
- `grid/R` : `(nz, nr)`
- `grid/Z` : `(nz, nr)`

3. 2D coords at root
- `R` : `(nz, nr)`
- `Z` : `(nz, nr)`

For 2D coordinate layouts, axes are reconstructed as:
- `r[j] = R[0,j]`
- `z[i] = Z[i,0]`

### Heat flux field dataset
One of:
- `fields/q_mag` : `(nz, nr)`
- `q_mag` : `(nz, nr)`

### Validation rules
- `nr >= 2`, `nz >= 2`
- `q_mag` shape must exactly match `(nz, nr)`
- Reconstructed `r` and `z` must be monotonic increasing
- Non-finite or negative `q_mag` values are clamped to `0`

## 4) Minimal command examples

### Evaporation with file heat flux
```sparta
fix fE evaporation 1 droplet_mix mass 1.0e-12 temp 773.15 radius 3.495e-5 heatflux/file ../heatflux.h5
```

### Evaporation with constant heat flux
```sparta
fix fE evaporation 1 droplet_mix mass 1.0e-12 temp 773.15 radius 3.495e-5 heatflux/constant 5.0e6
```

## 5) Practical note
If you regenerate inputs with `solps2openedge.py`, keep `plasma.h5` and `heatflux.h5` in the above shapes to avoid runtime HDF5 errors and silent zero-flux behavior.
