# test_ionization_recombination

Charge-state balance of atomic oxygen in a uniform plasma.  A single
cell is seeded with 2000 fully-stripped O8+ particles that ionize and
recombine under ADAS scd/acd rates until the local coronal equilibrium
is reached.  Serves as an end-to-end check of `fix volume/chem/adas`
against the expected Te, ne charge-state balance.

## Files

| File | Purpose |
|---|---|
| `in.ionization_recombination` | The deck. |
| `plasma.species` | SPARTA species table (O and W charge states; only O used). |
| `plasma.reactions` | 16 channels: 8 ionization + 8 recombination for O. |
| `plot_charged_states.py` | Parses `log.openedge` stats block, writes `oxygen_balance.png`. |

## Run

```bash
BIN=~/build_oe/openedge_mac_mpi/src/spa_mac_mpi
mpirun -np 8 $BIN -in in.ionization_recombination
python3 plot_charged_states.py
```

Default run: 100 ms at dt = 0.1 us (1 M steps), ~ 1 min wall time on
4 ranks.  Drop `time` to 10 ms for a quick smoke test; bump to 500 ms
if the highest charge states are still drifting.

The current deck uses `global pusher plasma pfields` so
`fix volume/chem/adas` reads Te and ne through the per-particle plasma
cache. The older inline `plasma <Te> <Ne>` arguments are no longer
accepted by the parser.

## Expected result

At Te = 12 eV the coronal equilibrium peaks at O5+ / O6+.  The log-time
plot shows O8+ cascading down through the charge states and settling
into a stable distribution by ~ 1 ms.
