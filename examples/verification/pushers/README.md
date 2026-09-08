# Pusher verification

| Test | Covers | Reference |
|---|---|---|
| `orbit/` | Boris vs GCA (rk2/rk4) orbit physics in 3D, 2D Cartesian and 2D axisymmetric; MPI rank invariance | Khan et al. (2012) banana orbit |
| `hybrid/` | Boris/GCA near-wall handoff: impact state, sheath barrier, operator coupling, axisymmetric target | Gyro-resolved Boris reference |

Each `run.sh` runs its decks and check scripts and exits 0 on PASS.
Single-particle decks run at `-np 1` for reproducible gate numbers.
