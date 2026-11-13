# Newborn reaction exposure

Neutral O is emitted from a plane at 5 eV (cosine distribution) into a uniform
plasma and tallied where it crosses the plane z = d un-ionized. With ionization
rate nu and speed v the transmitted fraction is exactly T = 2 E3(nu d / v).
`in.decay` measures nu from a 0-D decay; `in.newborn` is the slab.

    python3 scripts/scan.py        # SPA=<binary> NP=<ranks>; about 5 minutes on 4 ranks

The scan varies the timestep (nu dt = 0.2 ... 0.01) and the emission cadence
(`nevery` 1, and batches with nu T_emit = 0.4 and 1.0), three seeds each.
PASS requires every row within 1 % of exact, and the depth table (un-ionized
fraction at 0.25, 0.5 and 2 mean free paths) within 2 % at the finer step.

Before the release schedule (2026-11-13) batched emission let new particles fly
U*nevery*dt without chemistry: T was 11-22 % high at nu T_emit = 0.4 and 56-71 %
high at 1.0. With the schedule the batched rows match emission every step.
