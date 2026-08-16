# Lithium-droplet transport in a SOLPS plasma background

Verification gate for the particulate movers, isolated from surface
physics: three Li droplets of radii 1.5 mm / 2.5 mm / 3.5 mm launched
from the outer divertor, subject to DUSTT collection + Coulomb drag
(`model dustt2005 coulomb/self yes`, Pigarov 2005) + gravity +
evaporation (Antoine + Hertz-Knudsen) + OML charging against a
SOLPS-derived plasma background. A regression in drag, charging, or
thermal response flips the PASS/FAIL script; the integrated
liquid-metal workflow (`workflows/particulates/cat_liquid_metal_divertor`)
exercises the same stack but couples it to everything else.

## Files

| File | Purpose |
|---|---|
| `in.openedge` | Input deck (axisymmetric, x=Z / y=R slots). |
| `input/droplets.species` | Species table for the three droplet sizes. |
| `input/source` | Launch-state particle dump (3 droplets at same R/Z/v). |
| `input/plasma.h5` | Mesh-only plasma: D+ + Ne0–Ne10+ + Li0–Li3+, plus equilibrium. |
| `input/wall.surf` | Axisymmetric wall geometry with line endpoints in `(Z, R)`. |
| `scripts/plot_trajectories.py` | Post-process: reads `output/state.trj`, renders `output/trajs.png`; PASS/FAIL exit code. |

## Run

```bash
mpirun -np 4 ~/build_oe/src/spa_mac_mpi -in in.openedge
python3 scripts/plot_trajectories.py
```

All run products land in `output/` (gitignored). The deck runs up to
`N = time/dt = 1e6` steps but a halt fix ends it when every droplet is
gone: the droplets return ballistically to the wall within tens of ms
(surf reaction deletes them) having lost a negligible fraction of their
radius — mm-scale Li at 773 K spends its first ~0.1 s heating before
strong evaporation, so the flight ends in the heat-up phase.

## Pass criteria (enforced by the script, exit 0/1)

- three droplets present with launch radii 1.5 / 2.5 / 3.5 mm
- each radius monotonically non-increasing
- all trajectory points inside the domain box

## Regenerating `plasma.h5`

The background is the CAT attached case (converged SOLPS solution,
wall_flux verified against the b2pl target files) — the same file as
`workflows/particulates/cat_liquid_metal_divertor/input/plasma_attached.h5`:

```bash
python3 ../../../../tools/converters/convert_solps_plasma.py ~/jjl_solps_runs/attached \
    --b2fgmtry ~/jjl_solps_runs/baserun/b2fgmtry \
    --b2fstate ~/jjl_solps_runs/attached/b2fstate \
    --equ-file ~/jjl_solps_runs/baserun/dg.equ \
    --plasma-out input/plasma.h5 \
    --wall-in input/wall.surf \
    --geometry axi --heatflux jeremy_total
```

## Notes on the physics

- Heating uses the local OML collection closure from `ne`, `Te`, and `Ti`.
  Millimetre droplets are outside its formal validity range, so temperatures,
  evaporation rates, and lifetimes are provisional; the trajectory and
  geometry checks remain useful.
