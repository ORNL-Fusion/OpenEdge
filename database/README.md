# OpenEdge Database

Central repository for atomic, surface, and material data used by OpenEdge simulations.

## Structure

```
database/
  adas/           # ADAS ionization/recombination rate tables (HDF5)
  surface/        # BCA sputtering/reflection yield tables (HDF5)
```

## ADAS Data (`adas/`)

Pre-computed ionization and recombination rate coefficients from the
[ADAS](https://www.adas.ac.uk/) database, stored as HDF5 files.

| File | Element | Z |
|------|---------|---|
| `ADAS_Rates_8.h5` | Oxygen | 8 |
| `ADAS_Rates_73.h5` | Tantalum | 73 |
| `ADAS_Rates_74.h5` | Tungsten | 74 |

**Usage in input scripts:**
```
fix chem chem/adas 1 74 plasma.adas adas_dir ../../database/adas
```

**Regenerating from raw ADF11 data:**
```bash
export ADAS_ADF11_DIR=/path/to/eirene-db/Database/AMdata/Adas_Eirene_2010/adf11
cd database/adas
python adas.py        # generates ADAS_Rates_{Z}.h5
python adas_ta.py     # generates tantalum (Z=73) from tungsten truncation
```

## Surface Data (`surface/`)

BCA (Binary Collision Approximation) sputtering and reflection yield tables,
pre-computed as functions of incident energy and angle.

| File | System | Description |
|------|--------|-------------|
| `74_on_74.h5` | W on W | Self-sputtering yields |
| `O_on_W.h5` | O on W | Oxygen sputtering of tungsten |

HDF5 structure: `A` (angles), `E` (energies), `rfyld` (reflection yield), `spyld` (sputter yield).
