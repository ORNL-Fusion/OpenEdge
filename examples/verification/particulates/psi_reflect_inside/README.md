# psi_reflect_inside

Regression for `fix reflect/psi ... action reflect` with particles that
start a move already inside the psi contour (born inside by an emitter or
a reaction). The mover must reject the move, reverse v_R and count the
event in the fix vector instead of aborting.

Uses the ST40 equilibrium and grain species from
`examples/workflows/particulates/st40_lithium_powder_dropper/input`.

    ./run.sh /path/to/spa_binary      # prints PASS or FAIL

`OE_PSI_STRICT=1` restores the abort; the second run in `run.sh` checks it.
