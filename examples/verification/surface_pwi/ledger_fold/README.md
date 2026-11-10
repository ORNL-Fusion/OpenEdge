# surface/pwi ledger fold

The per-surf `s_adens*` ledgers of `surf_react surface/pwi` are folded across ranks every
`adens_nevery` steps and, since 837c7a4, also before every dump, at the end of a run and
before a restart is written. `in.fold` runs the deposit_tagging W-beam-on-W plate with
`adens_nevery 1` (reference) and `adens_nevery 100`, dumping the ledger every 70 steps up to
step 350 and writing a restart; `scripts/check_fold.py` requires the two ledgers to agree
(rtol 1e-9) at every dump step, and the restart read back to match as well, at 1 and `NP` ranks.

    SPA=<binary> NP=4 ./run.sh        # PASS/FAIL, exit code

Registered in `regression/run_regression.sh` as `pwi_ledger_fold`.
