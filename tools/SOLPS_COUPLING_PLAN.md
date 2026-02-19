# OpenEdge → SOLPS Coupling Plan

## Goal

Produce `source2d.00001` for SOLPS-ITER from OpenEdge droplet evaporation
output.  Only Li0 (neutral lithium) is non-zero; SOLPS ionization physics
populates Li+1..Li+3 internally.

SOLPS case: `fnacore=6.00e22_pheat=90.00MW_cont_dt=1e-6_te_up`
Grid: **nx=170, ny=36, ns=17**  (with guard cells: 172 × 38 per species block)

---

## What Is Wrong With the Existing Source File

Three bugs were identified in `openEdgeSource/create_source.py` and `get_source.py`.

### Bug 1 — Write-loop axes are transposed (critical)

The existing code loops over **radial columns** and writes **poloidal values** per line:
```python
# WRONG
for k in range(nx):          # 172 radial iterations → becomes the line count
    write arr[:, k]          # ny+2 = 38 values per line
```

Jeremy's MATLAB reference writer and the SOLPS Fortran reader both expect
**poloidal rows** as lines, with all **radial values** on each line:
```matlab
% CORRECT (Jeremy's MATLAB)
for i = 1:ny+2              % poloidal rows  → 38 lines per species
  for j = 1:nx+2            % radial values  → 172 values per line
    fprintf(...)
  end
  fprintf('\n')
end
```
```fortran
! SOLPS Fortran reader
Do j = 0, ns-1              ! species
  Do i = -1, ny             ! poloidal rows (38)
    Read(99,*) profile3d(i,j)   ! reads nx+2=172 values from one line
  End Do
End Do
```

Both the wrong and correct format contain the same total number of values
(17 × 38 × 172 = 111,112), so SOLPS reads without error — but the
source is deposited in completely wrong cells.

**Correct file shape:** 646 lines × 172 values
(17 species × 38 poloidal rows = 646; each row = 172 radial values)

### Bug 2 — Wrong source quantity and missing time factor

The existing script reads column 9 of `droplet_vapor_diag.rank0.dat` as a
number density [m⁻³]. SOLPS expects an **atom source rate [atoms/m³/s]**.
The distinction:

| Quantity | Units | Source |
|---|---|---|
| density | atoms/m³ | wrong column, wrong concept |
| source rate | atoms/m³/s | = dm_kg / (dt · AM · vol_3D) |

Missing factor: `1/dt` (converts atoms-per-step to atoms-per-second).

The correct data source is the new `mass_loss.txt` from `fix_evaporation`
(columns `f_fml_I[2]` and `f_fml_O[2]`).

### Bug 3 — Toroidal geometry not accounted for

The 3D cell volume for a cell at major radius R in a tokamak is:

```
vol_3D = 2π × R × dR_OE × dZ_OE    [m³]
```

The existing scripts use the SPARTA 2D Cartesian cell area (dR×dZ×0.1 m)
as the volume, which ignores the R-dependent toroidal factor `2πR`.
This factor ranges from ~15 at R=2.5 m to ~25 at R=4 m — a 1.7×
variation across the domain that matters for the source density.

---

## SOLPS File Format

```
source2d.00001 layout:
  ns=17 species blocks, concatenated
  Each block: ny+2=38 lines, one per poloidal row (iy = -1 .. ny)
  Each line:  nx+2=172 values (ix = -1 .. nx), space-separated
  Value at (iy, ix, is): S[iy,ix] * vol[iy,ix]   [atoms/s]

Total: 17 × 38 = 646 lines, each with 172 values.
```

Species order (0-based index):
```
0:D0  1:D+1  2:Ne0 .. 12:Ne+10  13:Li0  14:Li+1  15:Li+2  16:Li+3
```
→ Only index 13 (Li0) is non-zero in our source file.

---

## Pipeline (new script: tools/openedge_to_solps.py)

```
mass_loss.txt                     b2fgmtry
(OpenEdge grid dump)              (SOLPS geometry)
      │                                 │
      ▼                                 ▼
R_OE, Z_OE, dn_atoms/step       R_SOLPS, Z_SOLPS, vol_SOLPS
      │                            (ny+2, nx+2)
      │ S = dn/(dt × 2πR·dR·dZ)
      │ [atoms/m³/s]
      │
      └──── interpolate (linear + nearest) ──►  S_SOLPS(iy, ix)
                                                       │
                                                       ▼
                                           value = S_SOLPS × vol_SOLPS
                                           write  source2d.00001
```

### Step-by-step

1. **Read `mass_loss.txt`** — SPARTA grid dump with columns:
   `id  xc  yc  dm_inner  dn_inner  dm_outer  dn_outer`

2. **Compute source rate**:
   ```
   dn_total = dn_inner + dn_outer          [atoms/step/SPARTA-cell]
   vol_3D   = 2π × R × dR_OE × dZ_OE     [m³]   (toroidal cell)
   S        = dn_total / (dt × vol_3D)    [atoms/m³/s]
   ```

3. **Parse `b2fgmtry`** (pure Python, no quixote):
   - Read `crx` (26144 floats), reshape to (4, nx+2, ny+2) Fortran order,
     average corners → R_SOLPS (ny+2, nx+2)
   - Same for `cry` → Z_SOLPS
   - Read `vol` (6536 floats), reshape to (nx+2, ny+2) → transpose → (ny+2, nx+2)

4. **Interpolate** OpenEdge (scattered R,Z points) onto SOLPS grid:
   - Linear inside OE domain convex hull
   - Nearest-neighbour outside hull (no NaN extrapolation)

5. **Write `source2d.00001`**:
   ```python
   for ispec in range(ns):           # 0..16
     for iy in range(ny+2):          # 0..37  (outer loop → line count)
       write (S_SOLPS * vol)[iy, :]  # nx+2=172 values on one line
   ```

---

## Running for 110 Steps

`b2.sources.profile` in the SOLPS case already enables the external source:
```fortran
&profile
  read_sna0_2d = .true.
  sna0_2d_filename = "source2d.00001"
  sources_time_switch = 1.d-5
/
```

The source file is **steady-state** (constant per step), so running 110 steps
gives a short coupled test without modifying `b2.sources.profile`.

For a time-varying source (future): write one `source2d.NNNNN` per dump
interval and update `sources_time_switch` appropriately.

---

## Sanity Checks

The script prints a budget check at the end:

```
[sanity] OE total rate  : X.XXe+YY atoms/s
         SOLPS file rate : X.XXe+YY atoms/s
         Ratio (should be ~1): 0.9XX
```

If the ratio deviates significantly from 1.0, the interpolation is losing
mass (e.g. SOLPS domain doesn't cover the full OE domain). Acceptable
range: 0.9–1.1.

Additional checks:
- Plot `S_SOLPS` in R-Z: source should be concentrated near the inner/outer
  divertor (R≈2.5m, Z≈-3.8m) matching the droplet launch sites.
- Compare total atoms/s to known droplet injection rate.

---

## Files

| File | Role |
|------|------|
| `tools/openedge_to_solps.py` | New clean coupling script (this plan) |
| `examples/test_droplet/in.droplet_emission` | SPARTA input that writes `mass_loss.txt` |
| `openEdgeSource/create_source.py` | Old script — do not use (bugs 1-3) |
| `openEdgeSource/get_source.py` | Old script — do not use (same bugs) |

---

## Open Questions (to verify before production run)

1. **fnum factor**: `global fnum 0.0000001` in `in.droplet_emission`. In
   SPARTA, fnum = real particles per simulated particle. For macroscopic
   droplets where each simulated particle IS one real droplet, fnum should
   effectively be 1 for the droplet physics. If OpenEdge applies fnum to
   droplet counts, then `dn_total` in step 2 must be multiplied by `fnum`.
   Check: does `fix_evaporation` scale dm by fnum? (Currently it does not.)

2. **Full torus vs. sector**: The 2D simulation assumes toroidal symmetry
   (`2π × R` factor used above). If the simulation only represents a
   toroidal sector of angle `θ`, replace `2π` with `θ`.

3. **Ionization state source**: Should Li+1 also receive a source term
   (for droplets that partially ionize in flight)? For now, only Li0.
