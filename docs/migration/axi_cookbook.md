# Migration cookbook: 2D-Cart → SPARTA native axisymmetric

Moving a legacy `2D-Cart-mis-named-axi` test to true axi. Pilot was
`test_diii_d_neutrals` (commit history from 2026-04-19 → 2026-04-20).

## Tests by layout (as of 2026-04-20)

- **Axi**: `test_diii_d_neutrals` (pilot)
- **Cart 2D (legacy, awaiting migration)**: `test_west_axi`,
  `test_west_neutrals`, `test_west_timedep`, `test_d3d_walldyn`,
  `test_d3d_mateja`, `test_evaporation`, `test_solps_coupling`,
  `test_gca`, `test_neutral_transport`
- **3D Cart (unaffected)**: `test_west_3d`, etc.
- **True 1D slab (unaffected)**: `test_slab_stangeby2000`

## Steps

### 1. Regenerate plasma + wall with the SOLPS converter

Produces a single mesh-only `plasma.h5` (no `bfield.h5`, no `--equ-file`
at run time — equilibrium is embedded in `/equilibrium/*`):

```bash
python3 tools/converters/convert_solps_plasma.py <SOLPS_RUN> \
    --b2fgmtry <baserun>/b2fgmtry --equ-file <baserun>/dg.equ \
    --mesh-extra <baserun>/mesh.extra \
    --plasma-out input/plasma.h5 \
    --wall-out input/wall.surf \
    --wall-source mesh-extra
```

### 2. Flip the input deck

- `boundary o o p` → `boundary o ao p` (yhi `o` keeps outflow; ylo `a`
  marks the axis).
- `create_box X1 X2 Y1 Y2 Z1 Z2` → `create_box Z1_phys Z2_phys 0 R_max Z1 Z2`.
  `ylo = 0` is mandatory (`create_box.cpp:55`).
- `create_grid Nx Ny Nz` — swap the first two arguments so x stays the
  longer axial dimension and y is the radial dimension. Often double the
  original Nx (now along Z) since the axial range is wider.
- All `region block xlo xhi ylo yhi zlo zhi` — swap to the new layout
  (`xlo,xhi` are now Z range, `ylo,yhi` are now R range). Lower divertor
  region: `Z` is negative; upper divertor: `Z` is positive.
- `compute plasma/fields` — drop any `file plasma.h5 …` syntax (rejected).
  Declare `fix pd background file input/plasma.h5` first, then reference
  it: `compute cp plasma/fields all background pd …`. Drop any
  `equilibrium <file>` keyword and any `bfield.h5` arg — the mesh-only
  `plasma.h5` carries everything.
- All B-field / E-field source columns from `compute plasma/fields` stay
  named `bx by bz` — the compute projects to the right SPARTA slots
  automatically (commit dd6a746).

### 3. Re-tune `fnum`

The wall-segment cone-frustum area in axi mode is `2π·R·L` (full
revolution) instead of just `L` (per-radian). This bumps the per-step
physical emission rate by a factor `2π·R̄ ≈ 10`. Multiply `fnum` by ~5 to
keep similar sim-particle counts. Watch the first 1000 steps; if Np
climbs too fast, kill and bump `fnum` further.

### 4. Update post-processing

Anything that reads dump `xc`/`yc` as `(R, Z)` now reads `(Z, R)` — in axi
mode `xc` is `Z`, `yc` is `R`. The example `compare.py` in
`test_diii_d_neutrals` auto-detects the layout from the `yc` range.

### 5. Re-run, verify

The `[emit/surf/recycle]` diagnostic should print a single sensible Bohm
rate (~SOLPS-EIRENE ionization total / 5–15× amplification factor).
