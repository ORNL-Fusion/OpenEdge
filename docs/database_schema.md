# OpenEdge consolidated process-data schema

All volume-process (atomic, molecular, radiation) and surface-process
(sputter, reflection, recycling, evaporation, desorption) data consumed
by OpenEdge fixes and computes lives in a single HDF5 file,
`database/processes.h5`.  The filename reflects *what the data
describes* — elementary physical processes — not the consumer code;
the data itself is curated from open-ADAS, TRIM, literature, and other
external sources.  Two top-level groups match the physical partition
of the problem:

```
processes.h5
├── /volume/       ← homogeneous processes in the plasma bulk
└── /surface/      ← heterogeneous processes at a wall / solid interface
```

## Why two buckets

Every process OpenEdge models happens either in the gas phase (ionization,
charge exchange, dissociation, line radiation) or at an interface
(reflection, sputtering, recycling, evaporation, thermal desorption).
These two regimes have distinct governing physics, different source
conventions (rate coefficients vs. yield functions), and different input
data origins (open-ADAS vs. TRIM / SRIM / surface-chemistry codes), so
the top-level split mirrors a real physical dichotomy rather than an
implementation artefact.  It is the same split used by Cantera
(gas-phase vs. surface mechanisms), EIRENE (AM data vs. surface models),
and OpenFOAM combustion (homogeneous vs. heterogeneous reactions).

## Top-level groups

### `/volume/`

| subgroup                | contents                                    | source        |
|-------------------------|---------------------------------------------|---------------|
| `/volume/rates/`        | σv reaction-rate coefficients               | open-ADAS adf11: scd / acd / ccd |
| `/volume/radiation/`    | power coefficients (electron cooling)       | open-ADAS adf11: plt / prb |
| `/volume/pec/`          | per-line photon-emission coefficients       | open-ADAS adf15 (EIRENE distribution) |
| `/volume/thresholds/`   | scalar constants: E\_ion[q], E\_diss, bond energies | NIST, ADAS ionization potentials, literature |
| `/volume/reactions/`    | per-element reaction catalogs (YAML or JSON serialized strings) | authored in-repo |

Each dataset is 3-D `(n_Q, n_T, n_D)` stored as log10 values on a
log-Te × log-ne × 0-based charge-state grid.  Grid axes come as sibling
1-D datasets (`gridTemperature_*`, `gridDensity_*`, `gridChargeState_*`).

### `/surface/`

| subgroup                 | contents                                        | source                 |
|--------------------------|-------------------------------------------------|------------------------|
| `/surface/sputter/`      | sputter yields `Y(E, θ)` per projectile×target  | TRIM / SRIM runs       |
| `/surface/reflection/`   | reflection coefficients, energy/angle dists     | TRIM                   |
| `/surface/recycling/`    | recycling model parameters per wall material    | empirical + JET/DIII-D |
| `/surface/evaporation/`  | Antoine coefficients, Hertz-Knudsen models      | literature             |
| `/surface/desorption/`   | thermal desorption rates, outgassing spectra    | literature             |

Sub-group names reflect the physical process, which matches the fix
naming convention (`fix surface/emit/source` reads from
`/surface/sputter/`, `fix surface/emit/recycle` reads from
`/surface/recycling/`, etc.) — consumers can derive the HDF5 path
directly from the fix style string.

Datasets are binned on `(E, θ)` grids with arbitrary moments of the
outgoing distribution (`cos_polar_q`, `Eout_q`, etc.) as used by the
existing `surf_react surface/pwi` loader.

## Naming conventions

- **Lowercase, snake\_case** for all groups and dataset names.
- **Element symbols** are canonical lowercase one- or two-letter (`h`,
  `he`, `be`, `c`, `w`, …) — not periodic-table-capitalized, not `D`
  for hydrogen-isotopes.  Isotope handling is an attribute
  (`species = "D"`, `species = "T"`) on the consumer side.
- **Projectile × target pairs** separate with `_on_` to match the existing
  TRIM filename convention (e.g. `d_on_w`, `c_on_c`).
- **Units** always SI unless explicitly marked otherwise.  Every dataset
  carries a `units` attribute (see §Attributes below).

## Required dataset attributes

Every non-metadata dataset MUST have these three attributes so the
data is self-describing and machine-verifiable:

```python
ds.attrs["units"]  = "m^3/s"                                         # SI, spelled out
ds.attrs["source"] = "open-ADAS adf11 scd89 (retrieved 2026-04-22)"  # provenance
ds.attrs["method"] = "bilinear in log10(Te) × log10(ne)"             # how to interpolate
```

Optional but recommended:

```python
ds.attrs["citation"] = "Summers 2006, ADAS User Manual"
ds.attrs["date"]     = "2026-04-22"
```

## File-level attributes

`/` (root) and `/meta/`:

```python
f.attrs["schema_version"]  = "1.0"
f.attrs["generated"]       = "2026-04-22T18:30:00Z"
f.attrs["openedge_commit"] = "<git sha of the ingest code>"
f.attrs["sources"]         = [list of upstream databases, versions]
```

## Schema version and compatibility

Every `processes.h5` carries `/@schema_version` as `MAJOR.MINOR`:

- **MINOR** bumps are additive only: new datasets, new attributes, new
  subgroups.  Existing consumers read old MINOR files transparently.
- **MAJOR** bumps are reserved for breaking changes (dataset layout
  changes, unit changes, renames).  C++ loaders MUST check the major
  version and fail loudly on mismatch.

## Build and update

The single source of truth is `database/ingest/build_openedge_h5.py`,
which consolidates all raw inputs (adf11 text files, TRIM output, hand-
curated reaction catalogs) into `database/processes.h5`.  Run it after
any upstream update:

```bash
cd database/ingest
python3 build_processes_h5.py
```

Raw inputs live under `database/ingest/{adf11,reactions,surface_generators,
ionization_potentials}/`. The consolidated `processes.h5` is committed.

The previous per-element files (`ADAS_Rates_<Z>.h5`,
`database/surface/trim/*_on_*.h5`, `database/surface/<proj>_on_<targ>.h5`)
have been removed; all C++ consumers now read `processes.h5` exclusively.
