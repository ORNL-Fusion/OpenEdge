# OpenEdge Database

Runtime atomic-physics, PMI, and surface data consumed by OpenEdge at
simulation time, plus the ingest pipeline that builds the consolidated
`processes.h5` from raw ADAS / TRIM sources.

## Layout (2026-04-23)

```
database/
  README.md                 # this file.
  processes.h5              # the one and only runtime data file.
                            # Carries ADAS rates + thresholds + PEC +
                            # TRIM reflection + reaction catalogs,
                            # all under /volume/* and /surface/*.
  ingest/                   # build-time pipeline, LOCAL-ONLY
                            # (gitignored). Developer tooling to
                            # regenerate processes.h5 from open-ADAS
                            # and TRIM sources; NOT shipped in the
                            # repo or release tarballs.
    build_processes_h5.py   # driver: adf11 + IP + TRIM + reactions
                            # -> processes.h5
    adas.py                 # adf11 parser + element / class tables
    adas_boron.py           # B-specific ingest helper
    check.py                # sanity checks on processes.h5
    reactions/              # reaction-list text files (D.reactions,
                            #   W.reactions, ...) — tracked source of
                            #   /volume/reactions/<elem>/catalog
    adf11/                  # raw open-ADAS text files (scd89, acd89, ...)
                            #   — gitignored; fetch via download_data.sh
    ionization_potentials/  # ADAS_ionization_potentials_<elt> text files
                            #   — gitignored; fetch via download_data.sh
    trim/                   # raw TRIM reflection tables (when shipped)
                            #   — gitignored
    surface_generators/     # legacy generate_*.py for per-pair Eckstein
                            #   yield tables (deprecated: sputter computes
                            #   use analytic Eckstein from header now)
```

Per-pair surface yield tables (`database/surface/*.h5`) and standalone
PEC line files (`database/pec/*.h5`) were **removed** after
`processes.h5` became the consolidated source. The reaction-list text
files at `database/adas/reactions/` also moved to
`database/ingest/reactions/` since the runtime path now reads the
`/volume/reactions/<elem>/catalog` string directly from processes.h5
(legacy text-file fallback still works but runtime no longer touches
the tracked text files in a normal run).

## `processes.h5` — the runtime consolidation

Single HDF5 file carrying every atomic-physics and surface-reaction
coefficient used during a run. Layout:

```
/volume/
  rates/
    scd/<elem>/{coefficient, temperature, density}       # ionization
    acd/<elem>/...                                       # recombination
    ccd/<elem>/...                                       # charge exchange
  radiation/
    plt/<elem>/...                                       # line radiation
    prb/<elem>/...                                       # recomb + bremsstrahlung
  thresholds/<elem>/{coefficient, charge_state}          # ionization potentials
  pec/<elem>/<line>/...                                  # PEC (ingested; compute
                                                        #   photon_emissivity/grid
                                                        #   still uses per-line files)
  reactions/                                             # catalog of reaction types
/surface/
  reflection/<proj>_on_<target>/                         # TRIM reflection moments
  sputter/<proj>_on_<target>/{E, theta}                  # sputter axes (yields TBA)
```

### Consumers (runtime reads — all from `processes.h5`)

- `fix volume/chem/adas` — `/volume/rates/*`, `/volume/thresholds/*`,
  `/volume/reactions/<elem>/catalog` (via
  `ProcessLibrary::load_reactions_catalog`). Legacy text-file fallback
  at `database/ingest/reactions/<elem>.reactions` only fires when the
  catalog is absent.
- `surf_react surface/pwi` — `/surface/reflection/<pair>`.
- `compute surface/physical/sputter` — analytic Eckstein coefficients
  from `src/eckstein_sputter_data.h` (compiled in). Does *not* read
  any HDF5 yield table at runtime in the `target`/`projectiles` API.
- `compute photon_emissivity/grid` — `/volume/pec/<elem>/<id>/<line>/`
  (via `ProcessLibrary::load_pec_line`).

## Reaction-list catalogs (`/volume/reactions/<elem>/catalog`)

Runtime input for `fix volume/chem/adas`. Plain-text lists that tell
the fix which reactions to consider and with which rate class (`A` =
ADAS, `J` = Janev polynomial), ingested into
`processes.h5:/volume/reactions/<elem>/catalog` as opaque strings.

```
D --> D+
I A 1.0 0.0 0.0 0.0 0.0

D2 --> D + D
D J -2.787e+01 1.052e+01 -4.973e+00 1.451e+00 ...
```

Usage:

```
fix fchem volume/chem/adas 1 D auto            # → processes.h5 catalog
fix fchem volume/chem/adas 1 W auto            # → processes.h5 catalog
fix fchem volume/chem/adas 1 D ../input/custom.reactions   # override with file
```

The fix tries `/volume/reactions/<elem>/catalog` first; the text-file
fallback at `database/ingest/reactions/<elem>.reactions` only fires if
the element is absent from processes.h5.

## PEC tables (`/volume/pec/<elem>/<id>/<line>/`)

Per-emission-line photon emissivity coefficients for
`compute photon_emissivity/grid`. Stored in processes.h5 under
`/volume/pec/<elem>/<pec_id>/<line_key>/`, ingested from **open-ADAS
adf15** files by `build_processes_h5.py`.

## Regenerating `processes.h5`

Needs raw open-ADAS ASCII text (adf11 rates + adf15 PEC +
ionization-potential files) under `database/ingest/`. If those aren't
shipped with your checkout, symlink from a SOLPS / open-ADAS mirror:

```
ln -s /path/to/adas/adf11/scd89 database/ingest/adf11/scd89
ln -s /path/to/adas/adf11/acd89 database/ingest/adf11/acd89
ln -s /path/to/adas/adf11/ccd89 database/ingest/adf11/ccd89
ln -s /path/to/adas/adf11/plt89 database/ingest/adf11/plt89
ln -s /path/to/adas/adf11/prb89 database/ingest/adf11/prb89
ln -s /path/to/adas/adf15      database/ingest/adf15

cd database/ingest
python3 build_processes_h5.py
```

The ingest also absorbs TRIM reflection tables from `ingest/trim/`
(currently empty — pending SDTrimSP run batch) and the reaction-list
text files from `ingest/reactions/`.
