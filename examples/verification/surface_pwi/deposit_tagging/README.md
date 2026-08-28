# surface/pwi deposit tagging and layer seeding

Verifies the `surf_react surface/pwi` keywords `deposit_as`, `adens_init_file`,
`adens_init_group`, the `S`-channel `yscale`, and the
`compute surface/physical/sputter` keywords `target_like` / `yield_scale`.

A hot W beam (`model thermal`, 3e6 K) from a source plane deposits on a W
plate while W self-sputtering (RustBCA `w_on_w`) erodes it.

| case | deck | what it checks |
|---|---|---|
| untagged / tagged | `in.tag` | with `deposit_as W Wd` the net ledger, gross erosion and strata thickness are identical; retained W lands in the `Wd` column; erosion is debited from the exposed material (bulk W only while the reaction zone fills) |
| split | `in.tag` + `w_split.recycle` | `S w_on_w mat Wd yscale 0.5` halves the deposit's self-sputter erosion |
| seed | `in.seed` | `adens_init_file <dump> s_adens_net Wd 1000` seeds a per-surface layer (exact x scale) and `adens_init_group` restricts both file and uniform layers to one surf; strata stack is Wd on bulk |
| bw | `in.bw_smoke` | B-on-W compound deck (WEST wedge) still runs with `deposit_as`; `target Wd target_like W yield_scale 0.5` equals 0.5 x the W compute |

```bash
SPA=~/build_oe/src/spa_mac_mpi PYTHON=python3 ./run.sh      # PYTHON needs numpy
SKIP_BW=1 ./run.sh                                          # skip the 70 MB plasma smoke
```

Exit code 0 = PASS. Runs are deterministic (`comm/sort yes`), <= 2000 steps.
