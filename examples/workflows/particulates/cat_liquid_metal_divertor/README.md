# CAT liquid-metal divertor sources

This CAT workflow evaluates lithium release from a liquid-metal divertor in a
static, SOLPS-derived plasma background. It combines a spatially varying
surface state with three wall-source models:

- thermal evaporation from the local surface temperature;
- deuterium-driven adatom desorption;
- physical sputtering by D, Ne, and Li projectiles.

The workflow also contains a particulate branch that launches lithium droplets
and follows their charging, drag, and plasma-driven evaporation.

## Model boundary

`in.openedge` loops over the attached and detached backgrounds.
For each case, `fix surface/state/lm` creates and fills the per-surface
`Tsurf_lm` and `Gamma_D_lm` fields in memory. The source computes consume those
fields in the same simulation. The CSV is diagnostic output and is never read
back into a second `.surf` file.

Both cases use one closed, full-wall geometry. Its inner and outer
liquid-metal divertor regions are locally refined and selected by surface ID;
there are no separate slab or case-specific surface files.
The wall is the axisymmetric full-device geometry used by the droplet workflow,
with only the two divertor legs refined to roughly 5 mm segments.

`clear` resets particles, fixes, and surfaces before the detached case while
preserving the `case` input variable.

Atomic `surface/emit/source` commands remain disabled because this deck follows
liquid droplets. An atomic transport case should define Li atom/ion species and
the associated chemistry.

Droplet heating is explicitly set to `heating oml`, using local `ne`, `Te`, and
`Ti`. The particle dump records the applied heat flux and `a/lambda_D` so the
collection regime can be checked along each trajectory. This is a provisional
closure for the millimetre droplets, which are outside the formal OML range;
quantitative temperatures, evaporation rates, and lifetimes are not yet
validated.

Until D/Ne/Li-on-Li angle-energy tables are installed, physical sputtering uses
the analytic fallback and is least reliable at grazing incidence.

This is therefore a device workflow, not the constant-flux emitter check. The
focused cadence and particle-count test lives at
`../../../verification/surface_emission/constant_flux/`.

## Main files

| File | Purpose |
|---|---|
| `in.openedge` | Attached/detached OpenEdge deck |
| `input/plasma_attached.h5`, `input/plasma_detached.h5` | Static SOLPS-derived backgrounds |
| `input/wall.surf` | Full wall with refined inner and outer divertors |
| `input/ld_tg_{i,o}_{attached,detached}.dat` | Inner/outer liquid-metal inputs |
| `scripts/analysis.ipynb` | Analysis-only attached/detached comparison |
| `scripts/evap_adatom.py` | Python reference implementations of the wall-source models |
| `scripts/plotter.py` | Shared analysis plotting helpers |

## Run

From this directory:

```bash
mpirun -np 4 /path/to/spa_mpi -in in.openedge
```

Results are written under `output/attached/` and `output/detached/`, with
separate inner- and outer-divertor surface diagnostics.
