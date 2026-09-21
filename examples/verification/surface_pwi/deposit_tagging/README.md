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
| seed | `in.seed` | `adens_init_file <dump> s_adens_net Wd 1000` seeds a per-surface layer (exact x scale) and `adens_init_group` restricts both file and uniform layers to one surf; strata stack is Wd on bulk; the initial coating is excluded from the runtime net/dep/ero ledgers across the first synchronization |
| bw | `in.bw_smoke` | B-on-W compound deck (WEST wedge) still runs with `deposit_as`; `target Wd target_like W yield_scale 0.5` equals 0.5 x the W compute |

The `bw` smoke is self-contained. Its WEST surface, species, recycle, and
plasma-background files are in `input/`; it does not depend on an ignored
`examples/wip` directory.

## Interpreting the B-on-W smoke

`cpmi` is only the user-selected ID of the preceding
`compute surface/physical/sputter`; it is not a special OpenEdge keyword.
The binding `adens_erosion cpmi 0 W noconc` tells `surface/pwi` to debit the
W inventory using that compute's per-surface vector (column `0`). The separate
`surface/emit/source` fix launches the corresponding sputtered particles.

`deposit_as W Wd` is optional bookkeeping that separates retained,
redeposited W (`Wd`) from the original W substrate. The pseudo-material has
the W mass but is not a transported species. The `yield_scale 0.5` and
`yscale 0.5` values in this verification deliberately make `Wd` erode at
half the W yield so the feature can be tested. They are not recommended
physics; use `1.0` unless a measured or modeled deposited-layer yield supports
another value.

`surface/pwi` creates `s_adens_*` custom fields during initialization. A deck
must therefore execute `run 0 post no` after defining the surface reaction,
erosion computes, and fixes, but before a `dump surf` references
`s_adens_net`, `s_adens_dep`, `s_adens_ero`, or related fields. The reaction
must also include `adens_surf <name>`.

```bash
SPA=/path/to/spa_mpi PYTHON=python3 ./run.sh      # PYTHON needs numpy
SKIP_BW=1 ./run.sh                                          # skip the 11 MB plasma smoke
```

Exit code 0 = PASS. Runs are deterministic (`comm/sort yes`), <= 2000 steps.
