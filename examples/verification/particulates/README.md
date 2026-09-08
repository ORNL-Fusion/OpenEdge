# Particulate verification

- `dustt2005/uniform_plasma_benchmark/` — unit test. Grain charging and
  DUSTT-2005 drag in a uniform prescribed plasma are checked against the
  analytic Pigarov (2005) results.
- `dustt2005/cat_solps_droplet_transport/` — integration test. The same
  DUSTT-2005 particulate model is exercised in the CAT geometry with a SOLPS
  background; three Li droplets are flown to the wall.
- `dis2021/uniform_plasma_comparison/` — model comparison. DUSTT-2005 and
  DIS-2021 are run from the same one-grain initial condition, checked against
  their respective equations, and compared for charge, drag, and OML heating.
