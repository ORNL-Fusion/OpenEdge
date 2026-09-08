# Lithium droplet transport in a SOLPS background

Three Li droplets (radii 1.5, 2.5, 3.5 mm) launched from the CAT outer
divertor with dustt2005 collection and Coulomb drag, gravity, OML charging
and Antoine/Hertz-Knudsen evaporation in a SOLPS-derived background. The
droplets return ballistically to the wall within tens of ms; a halt fix
ends the run when all are gone.

## Files

| File | Purpose |
|---|---|
| `in.openedge` | Axisymmetric deck (x=Z, y=R) |
| `input/droplets.species` | Three droplet sizes |
| `input/source` | Launch states |
| `input/plasma.h5` | SOLPS plasma with equilibrium, same as the CAT workflow's attached case |
| `input/wall.surf` | Axisymmetric wall in (Z, R) |
| `scripts/plot_trajectories.py` | Reads `output/state.trj`, writes `output/trajs.png`; PASS/FAIL exit code |

## Run

```bash
mpirun -np 4 /path/to/spa_mpi -in in.openedge
python3 scripts/plot_trajectories.py
```

## Pass criteria

- three droplets present with launch radii 1.5, 2.5, 3.5 mm
- each radius non-increasing
- all trajectory points inside the domain

## Regenerating `plasma.h5`

```bash
python3 ../../../../tools/converters/convert_solps_plasma.py <attached_run_dir> \
    --b2fgmtry <baserun>/b2fgmtry --b2fstate <attached_run_dir>/b2fstate \
    --equ-file <baserun>/dg.equ --plasma-out input/plasma.h5 \
    --wall-in input/wall.surf --geometry axi --heatflux jeremy_total
```

OML heating is outside its formal validity for mm droplets, so droplet
temperatures and lifetimes are indicative; the trajectory gates are not
affected.
