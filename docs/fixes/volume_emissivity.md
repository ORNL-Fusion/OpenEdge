# `compute volume/emissivity/grid` — synthetic line emission

Per-grid volumetric line emissivity:

```
ε = ne · nz · PEC(Te, ne)   [photons/m³/s/sr]
```

Uses per-particle `pweight` for weighted density (`nz`), Te/ne from a
`compute plasma/fields`, and a PEC table pulled from `processes.h5`.

> **Renamed 2026-04-24.** Formerly `compute photon_emissivity/grid`.
> Moved into the `volume/*` namespace alongside `fix volume/chem/adas`
> (volumetric atomic processes share a prefix).
>
> **Migrated to `processes.h5` 2026-04-23.** The legacy `pec_file PATH`
> keyword + standalone `database/pec/*.h5` files were removed. PEC data
> now lives under `/volume/pec/<elem>/<pec_id>/<line>/` in
> `database/processes.h5`, ingested by
> `database/ingest/build_processes_h5.py` from **open-ADAS adf15**
> files.

## Syntax

```
compute ID volume/emissivity/grid group mix \
        pec <elem> <pec_id> <line_key> \
        plasma_compute CID \
        [pec_units cm3s|m3s]
```

- **`pec <elem> <pec_id> <line_key>`** — 3-token selector matching the
  `/volume/pec/<elem>/<pec_id>/<line_key>/` group in
  `database/processes.h5`. Example: `pec w 4009 wi_4295_pec`.
- **`plasma_compute CID`** — reference to a `compute plasma/fields`
  that exposes Te/ne at particle positions.
- **`pec_units`** — `cm3s` (ADAS convention; default, multiplies by
  10⁻⁶) or `m3s` (already SI).

## `/volume/pec/` HDF5 layout (in `processes.h5`)

Ingested by `database/ingest/build_processes_h5.py`. Each line has:

```
/volume/pec/<elem>/<pec_id>/<line_key>/
    coefficient  (nT, nNe)   log10(PEC [cm^3/s or m^3/s])
    temperature  (nT,)       log10(Te [eV])
    density      (nNe,)      log10(ne [m^-3])
```

Loaded at compute init via `ProcessLibrary::load_pec_line()` with
rank-0-read + MPI_Bcast (same pattern as ADAS rates and TRIM
reflection tables).

## Output

One column per species group in the mixture. Combine with
`fix ave/grid` + `dump grid` for time-averaged emissivity maps.

## Related

- Raw PEC inputs are open-ADAS **adf15** files (excitation +
  recombination coefficients per emission line). Place them in the
  ingest pipeline alongside the adf11 rate files at
  `database/ingest/adf15/` and rerun `build_processes_h5.py`.
- `fix volume/chem/adas` — the matching ionization/recombination rate
  loader from `processes.h5:/volume/rates/`.
