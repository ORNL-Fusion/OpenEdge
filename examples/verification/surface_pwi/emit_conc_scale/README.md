# surface/emit/source conc_scale on a depleting layer

A 1e14 atoms/m^2 B layer on a W wedge patch (faces 1800-1900, `rzone 1e15`) erodes under the
plasma (`adens_erosion cpmiB`), and a uniform B wall source redeposits B on bare faces. The B
emitter uses `conc_scale adens_conc 5`, so its source must follow the live B concentration.

`./run.sh` (`SPA=<binary>`, `NP=4`) runs the deck with the sputter compute live (`static no`) and
frozen (`static yes`, no rebalance) and checks:

- emitted B per event (summed marker weights, wall-source markers excluded by weight) against
  sum F_i c_i A_i dt nevery, with c from the surface dump of the previous update (emission
  happens at the start of a step, the surface update at its end), in 5 windows, within 5%;
- the coated faces with F > 1e20 deplete (c 0.19 to below 1e-3), and bare faces switch on;
- `static yes` and `static no` emit the same B over the first 400 steps within 2% (later,
  redeposition on bare faces makes multi-rank runs diverge run to run; with Kokkos the frozen
  run emits on the device and the live one on the host, from different random streams).

Kokkos: `SPA_ARGS="-k on -sf kk"` with a Kokkos binary exercises the device emission path.

`in.two_writers` has two surface/pwi models updating the same concentration array every 100
and every 10 steps; each event must match the live source within 1% (it was 15% high when the
emitter watched only the first model).

Before the fix, `static yes` cached the concentration with the flux: it missed by up to a
factor of 2.9 and bare faces never emitted. Inputs are shared with `../deposit_tagging/input`.
Runtime about 50 s on 4 ranks (the `static no` run dominates).
