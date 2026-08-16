# WEST boron powder dropper

This axisymmetric workflow injects 150 µm boron grains from the WEST upper
shelf into the shared SOLEDGE-EIRENE background. It follows grain heating,
sublimation, B/B+ transport, and the resulting boron density.

## Layout

| Path | Purpose |
|---|---|
| `in.openedge` | Canonical OpenEdge deck |
| `input/` | Atomic and grain species plus boron reactions |
| `scripts/analysis.ipynb` | Geometry, trajectory, and density analysis |
| `scripts/plot_b_drop.py` | Standalone plotting helper |
| `../../impurity_transport/west_tungsten_transport/input/` | Shared WEST wall and plasma inputs |

## Run

```bash
mpirun -np 4 /path/to/spa_mpi -in in.openedge
python3 scripts/plot_b_drop.py
```
