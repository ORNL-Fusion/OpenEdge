# fix grain/charge (droplet/charge)

Instantaneous OML floating potential per grain; writes the charge number to
the per-particle custom vector `droplet_charge` (consumed by the Boris
pusher, `grain/drag coulomb/self`, and `grain/ablate heating oml`).

    fix ID grain/charge Nevery background PD [keywords]

Keywords:

- `material NAME` — supplies work function and Richardson constant
  (explicit keywords below override).
- `thermionic yes|no` — Richardson–Dushman emission with small-grain
  Schottky correction (default no).
- `work_function_eV V` (default 2.9, Li), `richardson_A V` (default 1.2e6).
- `ion_mass_amu M` (default 2.0), `mass|radius|temp V` seeds, `mixture ID`.

Electrostatic breakup (DUSTT): if the material has `tensile_Pa > 0` and
|phi| exceeds phi* = 0.1·sqrt(F_t[dyne/cm²])·R[um] volts, the grain splits
into two 2^(-1/3)-radius fragments (mass-conserving, charge halved,
grain_nweight copied, small separation kick). tensile_Pa = 0 (e.g. liquid
Li) disables. Built-in B: 1e9 Pa — breakup only for sub-micron grains, as
in DUSTT.
