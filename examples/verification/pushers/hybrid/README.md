# Hybrid Boris/GCA near-wall pusher

He+ ions approach an absorbing target in a tilted uniform B field. The
hybrid pusher integrates in guiding-center (GCA) mode far from the wall and
switches to Boris inside a shell of `knear` Larmor radii. Each stage compares
against a gyro-resolved Boris reference run at `dt` and `dt/2`.

| Stage | Script | Deck | Gate |
|---|---|---|---|
| A wall handoff | `run.sh` | `in.wall` | impact energy/angle distributions at 1x, 20x and 400x the Boris dt; shell-skip and leap-through cases switch every particle before impact; hysteresis case re-enters GCA once |
| B1 sheath barrier | `run_b.sh` | `in.sheath` | inbound ions gain `e*phi`; outbound ions below `e*phi` redeposit, above it escape exactly once |
| C operator coupling | `run_c.sh` | `in.coupling` | force/thermal drift, cross-field diffusion and Coulomb heating give the same observable in GCA and Boris |
| D1 axisymmetric target | `run_d.sh` | `in.axiring` | flux-weighted GC wall arrival matches Boris on a ring target; chord impact is rejected |

## Files

| File | Purpose |
|---|---|
| `in.wall`, `in.sheath`, `in.coupling`, `in.axiring` | Stage decks, parameters set with `-var` |
| `input/` | Species, targets and launch states (`scripts/make_inputs.py` regenerates them) |
| `scripts/plot_impact.py`, `plot_sheath.py`, `plot_coupling.py`, `plot_axi.py` | Stage gates; print PASS/FAIL and set the exit code |

## Run

```bash
./run.sh   /path/to/spa_mpi     # stage A, writes output/stageA_summary.png
./run_b.sh /path/to/spa_mpi     # stage B1
./run_c.sh /path/to/spa_mpi     # stage C
./run_d.sh /path/to/spa_mpi     # stage D1, writes output/stageD_summary.png
```

`PYTHON=...` selects the interpreter. Outputs land in `output/`.
