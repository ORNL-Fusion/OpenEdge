# test_coulomb — Coulomb collision validation (fix coulomb/background)

Two independent cases validating the Nanbu–Takizuka binary Coulomb
scattering operator, both using `fix coulomb/background` with
`plasma_data <fix_id>`.

| Case | Deck | Physics |
|---|---|---|
| slowdown | `in.nanbu_slowdown` | C3+ at 10 eV slows down against a fixed D+ background (Ti = 2 eV, n_i = 10¹⁷ m⁻³). Binary self-collisions are negligible (fnum = 1). |
| thermalization | `in.nanbu_thermalize` | Hot D+ (10 eV, 5000 part.) and cold C3+ (5 eV, 5000 part.) equilibrate via binary pair collisions toward ~7.5 eV. No background partner. |

Both decks use constant-background `fix plasma/data` for the Coulomb
logarithm input (Te, ne).

## Files

| File | Purpose |
|---|---|
| `in.nanbu_slowdown` | Slowing-down deck (background collisions). |
| `in.nanbu_thermalize` | Thermal-equilibration deck (binary collisions). |
| `plasma.species` | D, D+, C3+ species table. |
| `plot_slowdown.py` | Parses `output/particles*`, checks relaxation vs. Spitzer τ_eq. |
| `plot_thermalization.py` | Parses `output/particles*`, checks equilibration + momentum/energy conservation. |

## Run

```bash
source /opt/intel/oneapi/setvars.sh --force

# slowdown
mpirun -np 4 /home/cloud/buildOpenEdge/src/spa_mpi -in in.nanbu_slowdown
python3 plot_slowdown.py        # -> slowdown.png

# thermalization
mpirun -np 4 /home/cloud/buildOpenEdge/src/spa_mpi -in in.nanbu_thermalize
python3 plot_thermalization.py  # -> thermalization.png
```

Decks create `output/` automatically (via `shell mkdir -p output`) so
the particle dump never fails on a fresh checkout.

## Pass criteria

- **slowdown**: C3+ temperature relaxes from 10 eV toward 2 eV on the
  Spitzer cross-species timescale τ_eq ≈ 9 μs at these parameters.
- **thermalization**: D+ and C3+ populations converge toward a common
  ~7.5 eV temperature on τ_eq ≈ 90–100 μs; total momentum is exactly
  conserved per pair, total KE conserved to stochastic fluctuations.

## Notes

- `fix coulomb/background 1 plasma_data pd` uses `Te, ne` from the constant
  plasma/data for the Coulomb log; with `background A_bg Z_bg` it pulls
  `Ti, n_i, V_par, B` from the same fix for virtual-background partners.
- Without `background`, only binary self/pair collisions run — used by
  the thermalization case.
