# DIS-2021 verification

These cases exercise the `dis2021` particulate model and make its differences
from the legacy `dustt2005` closure explicit.

- `uniform_plasma_comparison/` runs the two models from the same one-grain,
  uniform-plasma initial condition. It checks each result against an
  independent transcription of its published equations and reports the
  resulting differences in charge, ion drag, and OML heat flux.

The larger research audit under `examples/wip/` remains useful for development,
but this directory contains the short, maintained verification intended for
users of the public example tree.
