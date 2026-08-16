# Constant-flux surface emission and cadence scaling

Emits Li from a 1 m² source surface at a constant flux of 10⁶ m⁻²s⁻¹
(fnum = 1, dt = 10 µs, 1000 steps) into a vacuum box; particles vanish
at the far surfaces. Total emitted must equal

```
flux * area * dt * nsteps = 1e6 * 1 * 1e-5 * 1000 = 10000
```

independent of the emission cadence `nevery` (guards the dt*=nevery
flux-scaling fix).

## Files

| File | Purpose |
|---|---|
| `in.constant_flux` | Deck; `nevery_in` is an index variable set per run. |
| `input/li.species`, `input/source.surf` | Species table and emitting surface. |
| `run.sh` | Does everything: sweeps nevery = 1, 5, 10, 50 → `output/log.n*`, checks totals, plots, PASS/FAIL exit code. |

## Run

```bash
./run.sh              # or: SPA=<binary> NP=<ranks> ./run.sh
```

## Pass criteria (enforced by run.sh, exit 0/1)

- all four nevery runs present
- per run: |total emitted − 10000| / 10000 < 5% (statistical noise ~1%)
