# WEST axisymmetric W-transport example (OpenEdge)

True-axisymmetric SOLEDGE3X-driven test: tungsten erosion by oxygen ions,
transport through the SOL with Braginskii thermal force + anomalous
cross-field diffusion + Nanbu Coulomb collisions, ionisation/recombination
via ADAS, and TRIM wall reflection.

Plasma + wall are regenerated from a SOLEDGE3X 3MW snapshot using the
mesh-native converter — no separate B-field or equilibrium file is needed.

## Layout

- `dimension 2`, `boundary o ao p` (SPARTA-native axisymmetric).
- Slot map: `x = Z` (axial), `y = R` (radial, axis at y=0), `z = phi` (wedge).
- All per-cell/per-surf diagnostics are full-3D — no `2π·R̄` post-multiply.

## Run

```bash
source /opt/intel/oneapi/setvars.sh --force
cd examples/test_west_axi
mpirun -np 64 ~/buildOpenEdge/src/spa_mpi -in in.west \
    -var nlaunchTotalW 100 -var Nphase 100000 -var Ndump 10000 -var tag A
```

Override `-var nlaunchTotalW` for macroparticle-count scans;
`-var tag {A,B,C}` separates outputs across runs.

## Input files

- `input/plasma.h5` — mesh-only plasma + embedded equilibrium + B-field
  (`/equilibrium/{r,z,psi,psib,psi_axis}`, `/ion_species/*`, `/mesh/*`).
- `input/wall.surf` — SPARTA polyline in axi `(Z,R)` layout, inward normals
  (no `invert` on `read_surf`).
- `input/core.surf` — psi_norm = 0.95 contour around the magnetic axis,
  used as the inner core-absorb boundary (`surf_collide vanish`).
  Regenerated alongside `plasma.h5` by the converter's `--core-out` flag.
- `input/wall.recycle` — TRIM reflection rules for D, O, W on W (resolves
  pair keys from `database/processes.h5:/surface/reflection/`).
- `input/plasma.species` — SPARTA species definitions for W through W20+.

## Physics stack in `in.west`

1. `fix plasma/data` — loads mesh-native plasma.h5, provides per-particle
   queries of `ne`, `Te`, `Ti`, `B`, and mesh-level `grad_Te` / `grad_Ti`.
2. `surf_collide diffuse` + `surf_react surface/pwi` — thermal wall +
   TRIM-reflect PWI via `processes.h5`.
3. `fix volume/chem/adas` — W ionisation / recombination chain from
   `processes.h5:/volume/rates/`.
4. `fix thermal_force` — Braginskii ion + electron thermal forces
   (`grad_Ti` / `grad_Te` from mesh).
5. `fix coulomb/background` — Coulomb pitch-angle scatter against D⁺ background.
6. `fix cross_diffusion` — anomalous `D_perp` perpendicular diffusion.
7. `compute surface/physical/sputter` — analytic Eckstein W erosion by O⁺…O⁸⁺.
   (*TODO: migrate to processes.h5 TRIM sputter table when the
   `target`/`projectiles` API documented in `docs/fixes/pmi_surf_data.md`
   lands in `compute_pmi_surf_data.cpp`.*)
8. `fix surface/emit/source` — emits sputtered neutral W from the PMI
   erosion source.

## Main outputs

- `output/grid.dens.<tag>.west` — per-cell `nrho_w` for W through W20+
  plus cell volume, dumped every `Ndump` steps.
- `output/state.<tag>.west.final` — per-particle snapshot at end of run.

## Plot

```bash
python3 analysis/plot_grid_density_west.py \
    --dump output/grid.dens.A.west \
    --wall input/wall.surf \
    --out output/grid_density.west.png \
    --show --log
```

*Note: the plotter was originally written for the 2D-Cartesian layout.
Axi support (`xc = Z`, `yc = R`) may need a small tweak; check axis labels
before trusting a new plot.*

## Regenerating plasma + wall + core from SOLEDGE3X

```bash
python3 tools/converters/convert_s3x_plasma.py <SOLEDGE_RUN> \
    --plasma-snapshot plasma_00010.h5 \
    --plasma-out input/plasma.h5 \
    --wall-out input/wall.surf \
    --core-out input/core.surf --psi-norm-core 0.90 \
    --main-ion-spec 1
```

`<SOLEDGE_RUN>` must contain `refParam_raptorX.h5`, `meshEIRENE.h5`,
`mesh_raptorX.h5`, `mesh.h5`, and the requested plasma snapshot.
`--core-out` traces the psi_norm contour from the equilibrium embedded
in plasma.h5 and writes it as a SPARTA surface file.
