# Boris and GCA orbit verification

One H+ ion in the analytical axisymmetric tokamak field of Khan et al.
(2012). The Boris pusher at gyro-resolving `dt` is the reference; the GCA
pusher (rk2 and rk4) must reproduce its guiding-center orbit in 3D, 2D
Cartesian and 2D axisymmetric slots.

## Field

```
B_R = -Z/(2R)    B_phi = R0/R    B_Z = (R-1)/(2R)
psi = Z^2/4 + R^3/6 - R^2/4
```

`scripts/make_khan_plasma_h5.py` writes this psi map to `khan_plasma.h5`;
`fix background` reconstructs B and its derivatives from it.

## Setup

| Parameter | Value |
|---|---|
| Launch | (1.09, 0, 0) m, v = (0, 6.005e4, -1.972e5) m/s |
| dt | 5e-10 s, 600 000 steps (about 6 bounce periods) |
| Boris | `subcycles 1`, `gca_switch 1e12` |
| GCA | `subcycles 1`, `gca_switch 2.5`, `gca_integrator rk2` or `rk4` |

## Files

| File | Purpose |
|---|---|
| `in.boris` | Boris reference, writes `output/traj.boris` |
| `in.gca` | 3D GCA, `-var gcaIntegrator rk2|rk4` |
| `in.gca.2d`, `in.gca.axi` | Same orbit in 2D Cartesian (x=R, y=Z) and axisymmetric (x=Z, y=R) slots |
| `input/source.*`, `input/plasma.species` | Launch states and species |
| `scripts/plot_trajectories.py` | Compares a GCA dump with `traj.boris`; PASS/FAIL exit code |
| `check_mpi.sh` | 1-rank vs 4-rank agreement to dump precision |

## Run

```bash
./run.sh /path/to/spa_mpi        # all decks, all gates, MPI check
```

Or by hand:

```bash
python3 scripts/make_khan_plasma_h5.py
mpirun -np 1 $SPA -in in.boris
mpirun -np 1 $SPA -var gcaIntegrator rk4 -in in.gca
python3 scripts/plot_trajectories.py --gca-dump traj.gca.rk4 --tag rk4
python3 scripts/plot_trajectories.py --gca-dump traj.gcaaxi.rk4 --tag axi.rk4 --mode axi
```

Figures go to `output/`: trajectory, guiding-center, and rho_L/L_B panels
per tag.

## Pass criteria

- banana orbit matches Khan Fig. 3a; doubling v_phi in `source.single` gives the passing orbit of Fig. 3b
- guiding-center (R, Z) RMS difference Boris vs GCA < 1e-2 m
- GC energy invariant H = m v_par^2/2 + mu B: secular |dH/H| < 20 ppm, excursion < 500 ppm
- rho_L/L_B stays in the GCA regime after the switch

Multi-rank runs agree to dump precision for about 150k steps; beyond that
roundoff is amplified by the marginally trapped orbit.
