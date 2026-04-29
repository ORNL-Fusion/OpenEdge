# Bug: `fix force/thermal` silently under-applies the kick when `Nevery > 1`

## Status
**FIXED** — Option A applied: `kick_half(0.5 * update->dt * nevery)` at both
`start_of_step` (line 285) and `end_of_step` (line 312). Builds cleanly with
the standard CMake target `spa_mpi`. Patched in both
`src/fix_force_thermal.cpp` and `src/OPENEDGE/fix_force_thermal.cpp`.

## Severity
High — silently wrong physics. Users running with `Nevery=100` get **1% of the
correct integrated thermal force**. The fix runs without warning or error.

## Affected file
`src/OPENEDGE/fix_force_thermal.cpp`

## Root cause

The fix gates execution on `(ntimestep % nevery) != 0` (lines 267, 292), but
when it does fire, the kick uses the bare per-step `dt`:

```cpp
// src/OPENEDGE/fix_force_thermal.cpp:285
kick_half(0.5 * update->dt);

// src/OPENEDGE/fix_force_thermal.cpp:312
kick_half(0.5 * update->dt);
```

So with `Nevery = N`, each particle gets one `dt`-magnitude thermal kick
every `N` steps, instead of either:

- (a) one `N·dt`-magnitude integrated kick every `N` steps, or
- (b) one `dt`-magnitude kick every step (`Nevery = 1`).

The result is a `1/N` under-application of the thermal force.

## How `fix cross_field_diffusion` does it correctly

Same `Nevery` gating pattern, but the random-walk step length is scaled:

```cpp
// src/OPENEDGE/fix_cross_field_diffusion.cpp:326
const double dt  = update->dt * nevery;
```

So an `Nevery=100` random walk has the correct integrated variance
`2·D·(N·dt)`. This is the pattern `force/thermal` should follow.

## Reproducer

In `examples/test_west_axi/in.west`:
```
fix ftf force/thermal 100 background pd ion_thermal yes elec_thermal yes
```
For trace W in WEST SOL, the thermal force is the dominant parallel force
in the SOL — under-application by 100× breaks the upstream/downstream
asymmetry that drives oxygen poloidal redistribution (cf. Ciraolo
IAEA-1164 Fig. 4) and changes the W density poloidal map.

## Proposed fix

Two clean options:

### Option A — scale the kick (matches `cross_field_diffusion`)

```cpp
// fix_force_thermal.cpp:285
kick_half(0.5 * update->dt * nevery);

// fix_force_thermal.cpp:312
kick_half(0.5 * update->dt * nevery);
```

Pros: lets users set larger `Nevery` for cost.
Cons: makes the kick impulsive (large Δv every N steps); for large
`Nevery·dt × |F|/m` the integration becomes inaccurate even though the
zeroth-moment integral is correct.

### Option B — error out when `Nevery > 1`

Reject `Nevery > 1` at parse time with a clear error message pointing to
the integration accuracy concern. Users keep the current per-step
behavior with `Nevery = 1` and the fix stays unsurprising.

## Recommendation

Option A *plus* a runtime warning when `Nevery·|F|·dt/m > 0.1·v_thermal`,
i.e. when the impulsive kick is comparable to the particle's thermal speed.
That preserves cost flexibility while flagging accuracy regressions.

## Workaround (for users)

Until fixed: always set `Nevery = 1` for `fix force/thermal`, e.g.
```
fix ftf force/thermal 1 background pd ion_thermal yes elec_thermal yes
```

## Discovered

While reproducing Ciraolo et al., IAEA-1164 / NF 60 (2020) Fig. 5 with
`examples/test_west_axi`. Setting `Nevery=100` (the example default at
the time) gave a markedly weaker O upstream-shift than ERO2.0 reports,
consistent with the thermal force being applied at 1% of its correct
magnitude.
