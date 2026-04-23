# test_diii_d_neutrals

OpenEdge **plasma-neutral interaction** benchmark on a DIII-D
axisymmetric geometry with a fixed SOLPS plasma background.  Drives
wall recycling plus volumetric chemistry (ionisation, CX, dissociation;
recombination is latent since ions are fluid-only in this mode) and
dumps per-cell source moments so the reaction footprint is visible
across the poloidal plane.

Default: 50,000 steps at 5 ns (≈10 τ) — reaches steady state.
~24 s wall time on 8 MPI ranks.

## Layout

```
test_diii_d_neutrals/
  in.diii_d_neutrals         the deck
  plasma.h5                  SOLPS ne, Te, Ti, upar on the EIRENE mesh
                             + embedded /equilibrium/* (psi, r, z, btf, rtf)
  wall.surf                  SPARTA wall segments (axi: x = Z, y = R)
  core.surf                  psi_norm = 0.95 absorb contour
                             (regen via tools/extract_psi_contour.py)
  wall.recycle               TRIM + thermal absorb-and-reemit spec
  neutral.species            D2, D, D+ species table
  scripts/NOTES_fnum.md      fnum-sizing notes
  plots/                     post-processing figures (gitignored)
  diii_d_neutrals.grid       dump output (gitignored)
  log.openedge               SPARTA log (gitignored)
```

## Running

```bash
cd examples/test_diii_d_neutrals
source /opt/intel/oneapi/setvars.sh --force
mpirun -np 8 ~/buildOpenEdge/src/spa_mpi -in in.diii_d_neutrals
```

`NP = 8..32` is fine; 64 ranks trip an upstream SPARTA ghost-cell
clumping warning on this geometry and are not recommended.

## Coordinate convention

True SPARTA axisymmetric: `boundary o ao p`, `x = Z` (axial), `y = R`
(radial, axis at `y = 0`).  All per-volume / per-area diagnostics are
full-3D quantities — no `2π · R̄` post-multiply.  See
[`../../CLAUDE.md`](../../CLAUDE.md) for the slot-mapping details.

## Physics pieces

| fix / read\_surf                 | role                                                  |
|----------------------------------|-------------------------------------------------------|
| `fix pd plasma/data`             | loads `plasma.h5` (mesh + plasma + equilibrium)       |
| `read_surf wall.surf`            | outer vessel; diffuse reflection                      |
| `surf_react wallPWI surface/pwi`    | TRIM + thermal re-emission of incoming D / D2 / D⁺    |
| `read_surf core.surf ... vanish` | psi\_norm = 0.95 inner absorb surface                 |
| `fix frec emit/surf/recycle`     | Bohm-flux wall source, `R = 0.99`                     |
| `fix fchem chem/adas`            | volumetric D ionisation + CX + D₂ dissociation        |

## Dump schema

`diii_d_neutrals.grid` (every 1000 steps) — 15 cols per cell:

| col | field          | units         |
|-----|----------------|---------------|
| 1   | cell id        |               |
| 2   | xc = Z         | m             |
| 3   | yc = R         | m             |
| 4–7 | xlo ylo xhi yhi (cell bounds, for grid overlays) | m |
| 8   | nD             | m⁻³           |
| 9   | nD2            | m⁻³           |
| 10  | Sp             | m⁻³ s⁻¹       |
| 11  | Sm\_x          | kg m⁻² s⁻²    |
| 12  | Sm\_y          | kg m⁻² s⁻²    |
| 13  | Sm\_z          | kg m⁻² s⁻²    |
| 14  | Qe             | W m⁻³         |
| 15  | Qi             | W m⁻³         |

Summary mode (Sp, Sm vector, Qe, Qi) is the plasma-frame source-moment
layout from [`../../docs/neutral_plasma_coupling/main.tex`] §4.  For
per-reaction breakdown (20 cols, unsigned raw tally), swap
`output summary → output detailed` on the `fix fchem` line.

## Post-processing

Three plotters in `tools/` (all optional):

```bash
# field maps: densities + 2x2 source panels + optional radiation
python3 ../../tools/make_neutral_plots.py \
    --dump diii_d_neutrals.grid \
    --wall wall.surf --core core.surf \
    --plasma-h5 plasma.h5 \
    --adas-rates ../../database/processes.h5 \
    --out-dir plots

# domain-integrated convergence (SI units; steady-state sanity check)
python3 ../../tools/plot_convergence.py \
    --dump diii_d_neutrals.grid \
    --out  plots/convergence.png \
    --dt   5e-9

# psi-contour regeneration if the equilibrium in plasma.h5 changes
python3 ../../tools/extract_psi_contour.py \
    --plasma-h5 plasma.h5 \
    --out       core.surf \
    --psi-norm  0.95 \
    --preview   plots/core_contour_preview.png
```

## Choosing `global fnum`

`fnum` is the statistical weight — each sim particle represents `fnum`
real particles.  Pick it so the steady-state sim-particle count lands in
the 10k–100k range:

```
fnum ≈ total_emission_rate [real/s] · mean_lifetime [s] / N_sim_target
```

For DIII-D with SOLPS-matched recycling (~5e27 real ions/s), τ_ioniz ≈
10⁻⁵ s (hot-SOL mix), and `N_sim_target = 5e4`, this gives `fnum ≈ 1e18`.
The current deck uses a smaller `fnum = 5e15` because only part of the
divertor footprint is emitting; scale up once the full SOLPS rate is
active.  Sanity check at runtime: if `Np` in the stats block grows
unbounded `fnum` is too low; if it hovers near zero `fnum` is too high.

## Regenerating `plasma.h5` / `wall.surf` from a SOLPS run (one-time)

```bash
python3 ../../tools/converters/convert_solps_plasma.py \
    <SOLPS case dir> \
    --b2fgmtry   <SOLPS baserun b2fgmtry> \
    --equ-file   <SOLPS baserun dg.equ> \
    --mesh-extra <SOLPS baserun mesh.extra> \
    --plasma-out plasma.h5 \
    --wall-out   wall.surf \
    --wall-source mesh-extra
```

`plasma.h5` is mesh-only (~2 MB for DIII-D).  Plasma, equilibrium, and
the wall-to-B2-cell map all live in it — no separate files at run time.
