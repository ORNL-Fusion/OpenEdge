# Oxygen charge-state balance

2000 O8+ particles in one uniform cell ionize and recombine under ADAS
rates until coronal equilibrium. Checks `fix volume/chem/adas` against
the expected charge-state distribution at Te = 12 eV, which peaks at
O5+/O6+ by about 1 ms.

## Files

| File | Purpose |
|---|---|
| `in.ionization_recombination` | Deck; 100 ms at dt = 0.1 us |
| `plasma.species` | O charge states |
| `plasma.reactions` | 8 ionization and 8 recombination channels |
| `scripts/plot_charged_states.py` | Parses the log stats block, writes `oxygen_balance.png` |

## Run

```bash
mpirun -np 4 /path/to/spa_mpi -in in.ionization_recombination   # ~1 min
python3 scripts/plot_charged_states.py
```

Set `time` to 10 ms for a quick smoke, or 500 ms if the highest charge
states are still drifting.
