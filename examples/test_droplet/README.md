# test_droplet — 3-droplet trajectories in a SOLPS plasma background

OpenEdge axisymmetric droplet transport: three Li droplets of radii
1.5 mm / 2.5 mm / 3.5 mm launched from the outer divertor, subject to
Epstein drag + gravity + evaporation (Antoine + Hertz-Knudsen) + OML
charging against a SOLPS-derived plasma background.

## Files

| File | Purpose |
|---|---|
| `in.droplet_emission` | Input deck (axisymmetric, x=Z / y=R slots). |
| `droplet.species` | Species table for the three droplet sizes. |
| `source` | Launch-state particle dump (3 droplets at same R/Z/v). |
| `plasma.h5` | Mesh-only plasma: D+ + Ne0–Ne10+ + Li0–Li3+, plus equilibrium. Generated from `/home/cloud/li/v0_1` via `convert_solps_plasma.py`. |
| `wall.surf` | OpenEdge axi-oriented wall geometry (line endpoints in `(Z, R)`). |
| `plot_trajectories.py` | Post-process: reads `case.outer`, renders `trajs.png`. |

## Run

```bash
source /opt/intel/oneapi/setvars.sh --force
mpirun -np 4 /home/cloud/buildOpenEdge/src/spa_mpi -in in.droplet_emission
python3 plot_trajectories.py
```

Expected: `case.outer` trajectory dump + `trajs.png` with the (R, Z)
arc plus radius(t) and T(t) panels.

## Regenerating `plasma.h5`

```bash
python3 ../../tools/converters/convert_solps_plasma.py /home/cloud/li/v0_1 \
    --b2fgmtry   /home/cloud/li/v0_1/b2fgmtry \
    --b2fstate   /home/cloud/li/v0_1/b2fstati \
    --equ-file   /home/cloud/li/baserun/g000001.00001_symm.X4.equ \
    --mesh-extra /home/cloud/li/baserun/mesh.extra \
    --plasma-out plasma.h5 \
    --wall-out   wall.surf \
    --wall-source mesh-extra
```

`b2fstati` (initial state, 94 MB ASCII) is read rather than the stub
`b2fstate` because the run didn't write a converged state.

## Notes on the physics

- The deck warns at init that `plasma.h5` has no `q_par`/`q_perp` — the
  SOLPS converter doesn't write these yet, so `fix plasma/data` falls
  back to `q_par = 50 MW/m²`, `q_perp = 0`. `fix evaporation` applies
  `Qs = 0.25·|q|` (sphere geometric factor) to the droplet surface.
- Droplets evaporate out long before gravity or drag can visibly bend
  their trajectory (evap lifetime ~0.2 s, drag timescale ~600 s, ballistic
  peak ~2 s). To see bending, shrink the droplets to ~100 μm or drop
  `heatflux/scale` on the evap fixes.
