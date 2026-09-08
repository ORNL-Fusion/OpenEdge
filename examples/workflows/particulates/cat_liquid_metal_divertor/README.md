# CAT liquid-metal divertor sources

Lithium release from a liquid-metal divertor in static SOLPS backgrounds
(attached and detached). `fix surface/state/lm` sets per-surface temperature
and D flux; three wall-source models consume them:

- thermal evaporation from the local surface temperature
- deuterium-driven adatom desorption
- physical sputtering by D, Ne and Li

A particulate branch launches Li droplets and follows their charging, drag
and evaporation with `heating oml`. The dump records the applied heat flux
and a/lambda_D along each trajectory.

`in.openedge` loops over both backgrounds with one closed full-wall
geometry; the two divertor legs are refined to about 5 mm and selected by
surface ID. `clear` resets state between cases and keeps the `case`
variable.

## Files

| File | Purpose |
|---|---|
| `in.openedge` | Attached and detached deck |
| `input/plasma_attached.h5`, `input/plasma_detached.h5` | SOLPS backgrounds |
| `input/wall.surf` | Full wall with refined divertors |
| `input/ld_tg_{i,o}_{attached,detached}.dat` | Liquid-metal surface inputs |
| `scripts/analysis.ipynb` | Attached vs detached comparison |
| `scripts/evap_adatom.py` | Python reference for the wall-source models |
| `scripts/plotter.py` | Plot helpers |

## Run

```bash
mpirun -np 4 /path/to/spa_mpi -in in.openedge
```

Results go to `output/attached/` and `output/detached/`, with separate
inner and outer divertor surface diagnostics.

Caveats: mm droplets are outside the formal OML range, so droplet
temperatures and lifetimes are indicative. Li sputtering uses the analytic
yield until angle-energy tables are installed. The constant-flux emitter
check is `verification/surface_emission/constant_flux/`.
