# surf_collide grainbounce

Grain-wall interaction with DUSTT-style restitution.

    surf_collide ID grainbounce pstick pvel pdiff [pmass V] [ptemp V]

- `pstick` — sticking probability per impact (stuck grains are removed).
- `pvel` — velocity restitution |v'|/|v|.
- `pdiff` — diffuse (cosine-law) fraction; remainder is mirror reflection.
- `pmass` — radius retention per bounce, R' = pmass·R (mass ∝ pmass³).
- `ptemp` — temperature retention.

Non-grain particles (radius <= 0) vanish, so the style can be used on walls
that also absorb vapor and ions. DUSTT reference values: pstick ~ 0,
pvel = 0.8, pdiff = 0.5, pmass = 0.95, ptemp = 1.
