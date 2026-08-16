# Particulate verification

Two rungs of the same ladder — keep both:

- `dustt/` — **model unit test**: uniform prescribed plasma
  (`fix pd background constant`, B_z = 1 T), grain charging and
  dustt2005 drag checked against the analytic Pigarov 2005 answers
  (`check_dustt.py`). Catches edits to the formulas.
- `droplet_transport/` — **integration test**: the same movers in the
  real CAT geometry with a converged SOLPS background; three mm-scale
  Li droplets flown to the wall (`scripts/plot_trajectories.py`,
  PASS/FAIL). Catches the plumbing the unit test can't: background
  interpolation, axisymmetric pusher slots, wall interaction, halt.
