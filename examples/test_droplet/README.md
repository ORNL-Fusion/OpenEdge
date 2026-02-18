# test_drag

Simple drag-model comparison case using validated geometry/data from `examples/test_evaporation`.

## Run

```bash
cd examples/test_drag
mpirun -np 4 ../../src/spa_mpi -in in.drag_compare
python3 plot_drag_compare.py
```

## Modes

- `mode 1`: `model epstein` (drag only)
- `mode 2`: `model epstein` + evaporation
- `mode 3`: `model coulomb` + evaporation

This isolates Coulomb drag effects relative to baseline Epstein with identical initial conditions.

---

## Developer Notes

### `compute surf` always returns zero hits (surf_tally signature mismatch)

**Symptom:** `tmp.all.surf` (or any surf dump using `fix ave/surf` + `compute surf`) shows
all-zero values even though particles visibly hit surfaces (`SurfColl occurs > 0` in the log).

**Root cause:** OpenEdge extended the `surf_tally` virtual function signature in `update.cpp`
by prepending `double dtremain` as the first argument, and updated the base-class declaration
in `src/compute.h` to match:

```cpp
// compute.h (base class) – OpenEdge version
virtual void surf_tally(double, int, int, int, Particle::OnePart *,
                        Particle::OnePart *, Particle::OnePart *) {}
```

But the concrete subclass overrides were **not updated** and kept the old 6-arg signature:

```cpp
// OLD – does NOT override the virtual above; C++ treats it as a new, unrelated method
virtual void surf_tally(int, int, int, Particle::OnePart *,
                        Particle::OnePart *, Particle::OnePart *);
```

Because the signatures differ, C++ never registers the subclass method in the vtable for the
base virtual. Every call from `update.cpp` dispatches to the empty base `{}` — zero tallying.

**Fix:** Add `double /*dtremain*/` as the first parameter to the declaration **and** definition
of `surf_tally` in every subclass that overrides it:

| Header | Implementation |
|---|---|
| `src/compute_surf.h` | `src/compute_surf.cpp` |
| `src/compute_react_surf.h` | `src/compute_react_surf.cpp` |
| `src/compute_isurf_grid.h` | `src/compute_isurf_grid.cpp` |
| `src/compute_react_isurf_grid.h` | `src/compute_react_isurf_grid.cpp` |

```cpp
// FIXED – now overrides the base virtual correctly
virtual void surf_tally(double /*dtremain*/, int, int, int, Particle::OnePart *,
                        Particle::OnePart *, Particle::OnePart *);
```

**Rule for new compute classes** (e.g. PMI deposition/erosion via `fix surf/react`): any class
that sets `surf_tally_flag = 1` and overrides `surf_tally` must use the 7-argument signature
with `double` as the first parameter, otherwise hits will silently count as zero.

---

## Per-cell Li mass loss output (`mass_loss.txt`)

`in.droplet_emission` writes a grid dump `mass_loss.txt` every `Ndump` steps
with time-averaged per-cell Li mass loss for use as a SOLPS source term.

### Output columns

| Column | Quantity | Units |
|--------|----------|-------|
| `id`        | SPARTA cell index           | — |
| `xc`, `yc`  | Cell centre (R, Z)          | m |
| `f_fml_I[1]` | mean dm_kg/step, inner Li  | kg/cell/step |
| `f_fml_I[2]` | mean dn_atoms/step, inner Li | atoms/cell/step |
| `f_fml_O[1]` | mean dm_kg/step, outer Li  | kg/cell/step |
| `f_fml_O[2]` | mean dn_atoms/step, outer Li | atoms/cell/step |

### Equations (computed in `fix_evaporation`)

For each droplet per half-step:

```
dm_kg    = ρ · (4/3)π · (r_old³ − r_new³)          [kg, ≥ 0]
dn_atoms = dm_kg / A_M                               [atoms]
heat_J   = Q_s · 4π · r_new² · dt_half              [J]
```

Constants: ρ = 534 kg/m³, A_M = 1.53×10⁻²⁶ kg.

Both half-kicks per timestep accumulate into the same per-cell array
(zeroed once at `start_of_step`), so each column represents the **full
timestep dt = 2·dt_half** integrated mass/atom loss.

### Post-processing: Li source rate for SOLPS

```python
# source_rate [atoms/m³/s] = mean_atoms_per_step / (dt * cell_vol)
# mean_atoms_per_step : column f_fml_I[2] (or f_fml_O[2]) from mass_loss.txt
# dt                  : timestep (variable dt in the input script)
# cell_vol            : grid cell volume [m³], computed from xc/yc spacings
#                       (for a 2D grid with toroidal symmetry: V = 2π·R·ΔR·ΔZ)

import numpy as np

data = np.loadtxt("mass_loss.txt", comments="#")
# columns: id, xc, yc, fml_I1, fml_I2, fml_O1, fml_O2
xc, yc = data[:,1], data[:,2]
dn_inner = data[:,4]   # atoms/cell/step, inner droplets
dn_outer = data[:,6]   # atoms/cell/step, outer droplets

dt = 1e-5              # match variable dt in in.droplet_emission
dR = np.median(np.diff(np.unique(xc)))
dZ = np.median(np.diff(np.unique(yc)))
cell_vol = 2 * np.pi * xc * dR * dZ   # axisymmetric cell volume

source_rate_inner = dn_inner / (dt * cell_vol)   # atoms/m³/s
source_rate_outer = dn_outer / (dt * cell_vol)
```

### Physics sanity check

The total atoms lost per dump interval should match the drop in total
droplet mass observed in the particle trajectory dump (`case.drag.mode.1`):

```python
total_atoms_lost = dn_inner.sum() * Ndump   # inner, over one dump interval
expected = delta_mass_kg / A_M              # from particle dump
```
