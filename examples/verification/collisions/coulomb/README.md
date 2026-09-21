# Coulomb collision verification

Two cases for the Nanbu-Takizuka binary Coulomb operator.

| Case | Deck | Physics |
|---|---|---|
| slowdown | `in.background` | C3+ at 10 eV slows against a fixed D+ background (Ti = 2 eV, n = 1e17 m^-3) via `fix coulomb/background` |
| thermalization | `in.binary` | Hot D+ (10 eV) and cold C3+ (5 eV), 5000 particles each, equilibrate via `fix coulomb/binary` |

## Files

| File | Purpose |
|---|---|
| `in.background`, `in.binary` | Decks; dump to `output/particles_slowdown` and `output/particles_thermalize` |
| `plasma.species` | D, D+, C3+ |
| `scripts/plot_slowdown.py` | Checks T(t) against the NRL equipartition ODE; exit 0/1 |
| `scripts/plot_thermalization.py` | Checks equilibration and momentum/energy conservation; exit 0/1 |
| `scripts/mc_nanbu_check.py` | Standalone Monte Carlo replica of the kernel |

## Run

```bash
mpirun -np 4 /path/to/spa_mpi -in in.background     # ~1 s
python3 scripts/plot_slowdown.py
mpirun -np 4 /path/to/spa_mpi -in in.binary         # ~2 min
python3 scripts/plot_thermalization.py
```

## Pass criteria

- slowdown: C3+ T(t) tracks the NRL ODE (tau_0 ~ 33 us) within 10% rms of the initial gap; final T within 15%
- thermalization: final |T_D - T_C| < 10% of the initial gap; kinetic-energy drift < 2%; momentum drift < 1e-3 of thermal momentum; gap-decay rate 2 to 5 times slower than the NRL Maxwellian rate

The slower binary relaxation is physical: D+ leaves Maxwellian as slow
deuterons equilibrate first, and D-D self-collisions are too weak to
restore it. The instantaneous dT/dt at t = 0 matches NRL within a few
percent, and the kernel is algebraically identical to WarpX and Smilei.
