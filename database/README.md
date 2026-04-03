# OpenEdge Database

Central repository for atomic, surface, and material data used by OpenEdge simulations.

## Structure

```
database/
  adas/           # ADAS ionization/recombination rate tables (HDF5)
  surface/        # BCA sputtering/reflection yield tables (HDF5)
```

## ADAS Data (`adas/`)

Pre-computed rate coefficients from the [ADAS](https://www.adas.ac.uk/)
database, stored as HDF5 files. Each file contains ionization (SCD),
recombination (ACD), and charge exchange (CCD) rate tables as functions
of electron temperature and density.

| File | Element | Z | Rates |
|------|---------|---|-------|
| `ADAS_Rates_1.h5` | Hydrogen/Deuterium | 1 | ionization, recombination, CX |
| `ADAS_Rates_6.h5` | Carbon | 6 | ionization, recombination, CX |
| `ADAS_Rates_8.h5` | Oxygen | 8 | ionization, recombination, CX |
| `ADAS_Rates_73.h5` | Tantalum | 73 | ionization, recombination |
| `ADAS_Rates_74.h5` | Tungsten | 74 | ionization, recombination |

**HDF5 datasets:**
- `IonizationRateCoeff[nQ, nT, nD]` — ionization rates in log10(cm³/s)
- `RecombinationRateCoeff[nQ, nT, nD]` — recombination rates
- `ChargeExchangeRateCoeff[nQ, nT, nD]` — CX rates (optional, backward compatible)
- `gridTemperature_*[nT]`, `gridDensity_*[nD]` — log10 grids
- `gridChargeState_*[2, nQ]` — charge state pairs

**Molecular dissociation** (D₂ → 2D) uses Janev polynomial fits from
HYDHEL (not ADAS tables). Coefficients are specified directly in the
reactions file with style `J`:
```
D2 --> D + D
D J -2.787e+01 1.052e+01 -4.973e+00 1.451e+00 -3.063e-01 4.433e-02 -4.096e-03 2.160e-04 -4.929e-06
```

**Usage in input scripts:**
```
# Impurity ionization/recombination/CX (Z=6 carbon)
fix chem chem/adas 1 6 plasma.reactions adas_dir ../../database/adas plasma Te Ne

# Neutral transport (Z=1 hydrogen with D2 dissociation)
fix chem chem/adas 1 1 neutral.reactions adas_dir ../../database/adas plasma Te Ne
```

**Regenerating from raw ADF11 data:**
```bash
# Symlink ADAS ADF11 source data
ln -s /path/to/solps/modules/adas/adf11/acd89 database/adas/adf11/acd89
ln -s /path/to/solps/modules/adas/adf11/scd89 database/adas/adf11/scd89
ln -s /path/to/solps/modules/adas/adf11/ccd89 database/adas/adf11/ccd89

# Generate HDF5 files
cd database/adas
# Edit adas.py to set element/Z, then:
python adas.py        # generates ADAS_Rates_{Z}.h5 with ion/rec/CX
```

## Surface Data (`surface/`)

BCA (Binary Collision Approximation) sputtering and reflection yield tables,
pre-computed as functions of incident energy and angle.

| File | System | Description |
|------|--------|-------------|
| `74_on_74.h5` | W on W | Self-sputtering yields |
| `O_on_W.h5` | O on W | Oxygen sputtering of tungsten |
| `6_on_6_pmi.h5` | C on C | Carbon self-sputtering (for surf_react pmi) |

HDF5 structure for `surf_react pmi`: `E` (energies), `A` (angles), `RN` (reflection probability), `RE` (reflected energy fraction), `spyld` (sputter yield), `E_bind` (binding energy).

Legacy format: `E`, `A`, `rfyld`, `spyld`.

**Generating C-on-C tables with RustBCA:**
```bash
cd database/surface
python generate_c_on_c.py --output 6_on_6_pmi.h5 --nsamples 1000
```
