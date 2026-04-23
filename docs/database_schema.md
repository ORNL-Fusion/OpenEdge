# OpenEdge consolidated atomic-data schema

All atomic, molecular, and surface data consumed by OpenEdge fixes and
computes lives in a single HDF5 file, `database/openedge.h5`, organized
into two top-level groups that match the physical partition of the
problem:

```
openedge.h5
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
| `/volume/thresholds/`   | scalar constants: E\_ion[q], E\_diss, bond energies | NIST, ADAS ionization potentials, literature |
| `/volume/reactions/`    | per-element reaction catalogs (YAML or JSON serialized strings) | authored in-repo |

Each dataset is 3-D `(n_Q, n_T, n_D)` stored as log10 values on a
log-Te × log-ne × 0-based charge-state grid.  Grid axes come as sibling
1-D datasets (`gridTemperature_*`, `gridDensity_*`, `gridChargeState_*`).

### `/surface/`

| subgroup                 | contents                                        | source                 |
|--------------------------|-------------------------------------------------|------------------------|
| `/surface/yields/`       | sputtering yields `Y(E, θ)` per projectile×target | TRIM / SRIM runs     |
| `/surface/reflection/`   | reflection coefficients, energy/angle distributions | TRIM                |
| `/surface/recycling/`    | recycling model parameters per wall material    | empirical + JET/DIII-D |
| `/surface/thermal/`      | desorption rates, outgassing spectra            | literature            |
| `/surface/evaporation/`  | Antoine coefficients, Hertz-Knudsen models      | literature            |

Datasets are binned on `(E, θ)` grids with arbitrary moments of the
outgoing distribution (`cos_polar_q`, `Eout_q`, etc.) as used by the
existing `surf_react wall_pwi` loader.

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

Every `openedge.h5` carries `/@schema_version` as `MAJOR.MINOR`:

- **MINOR** bumps are additive only: new datasets, new attributes, new
  subgroups.  Existing consumers read old MINOR files transparently.
- **MAJOR** bumps are reserved for breaking changes (dataset layout
  changes, unit changes, renames).  C++ loaders MUST check the major
  version and fail loudly on mismatch.

## Build and update

The single source of truth is `database/ingest/build_openedge_h5.py`,
which consolidates all raw inputs (adf11 text files, TRIM output, hand-
curated reaction catalogs) into `database/openedge.h5`.  Run it after
any upstream update:

```bash
cd database/ingest
python3 build_openedge_h5.py
```

Raw inputs live under `database/raw/` (gitignored; regenerable from
public sources).  The consolidated `openedge.h5` is committed.

## Legacy per-element files

For a transition period, OpenEdge also ships the older per-element
files (`ADAS_Rates_<Z>.h5`, `database/surface/trim/*_on_*.h5`).
Consumers prefer `openedge.h5` if present and fall back otherwise.
These legacy files will be removed when all C++ consumers have been
migrated.
