# Timestep monitor

`fix dt/monitor` against values known in closed form. O atoms leave a wall patch into a
uniform plasma with a uniform magnetic field parallel to the wall and ionize to O+ and O2+:

- gyromotion: |q/m| B dt/sub = Z x 0.009957 / sub for charge Z;
- reactions: rate x interval = 0.009970 x nchem for O and 0.000477 x nchem for O+.

    python3 scripts/check.py        # SPA=<binary, with the Kokkos arguments if any>

Checked: the maxima, also with 2 subcycles and a chemistry interval of 3 steps; the
updated weight against the sum of the inventories present before each update (to 1e-6,
with markers of unequal weight); the Boris exposure against inventory x time (2 %, ions
are removed at the wall inside a step); the weighted fractions with a level between the
charge states; a last window shorter than N and a run shorter than N; "not assessed"
without chemistry; 1, 2 and 4 ranks; and that the fix consumes no random numbers.
