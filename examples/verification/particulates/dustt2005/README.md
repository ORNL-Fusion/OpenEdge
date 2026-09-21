# DUSTT-2005 verification

These cases exercise the `dustt2005` implementation at two levels:

- `uniform_plasma_benchmark/` isolates charging, drag, and kinematics in a
  prescribed uniform plasma and compares them with independent analytic and
  numerical reference results.
- `cat_solps_droplet_transport/` integrates the same model with a file-backed
  SOLPS plasma, CAT wall geometry, gravity, evaporation, and surface loss.

Keep the cases separate: the first diagnoses model-level regressions, while the
second catches integration failures involving realistic background and geometry
data. For a controlled explanation of how this model differs from DIS-2021, see
`../dis2021/uniform_plasma_comparison/`.
