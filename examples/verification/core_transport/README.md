# core_transport — 1-D core impurity transport solver

The coupled fix uses `CoreTransport1D::advance`, which substeps so that
`dt_sub * max(transport_loss + S + R) <= 0.5`. It checks all resulting
densities and outgoing fluxes and rejects negatives or nonfinite values;
it never repairs them by clipping. Reported fluxes average all substeps.
The original `step` remains available for the Aurora comparison below.

`run.sh` also checks the subcycled solver's positivity and conservation,
including stiff chemistry and a diffusion-only edge pulse. It checks
refined equilibrium volumes against an analytic elliptical torus and
weighted entry reservoir merging against serial sampling. Reservoir slots
are independent weighted samples with replacement: each slot retains the
minimum `-log(U)/pweight`, and slotwise minima merge across ranks. Sampling
a slot uniformly therefore uses physical weights exactly once.

The fix observes every MC timestep, accumulates elapsed `dt`, couples every
requested `nevery` steps, and flushes partial windows at run end. Returned
markers start as `PKEEP` and receive a full ordinary push on the next step.
Output files are initialized per fix instance and always written at run end.

Optional coupled checks, using the SPARC inputs already on disk:

```
python scripts/check_coupling.py --exe /path/to/spa --ranks 1 4
```

These cover both launch modes, MPI creation weights, partial windows,
changed timesteps, recreation of the fix, and the first electric-field
push of returned ions. Results are written under `output/coupling/`.
The smoke deck uses the generated `input/core_profiles.dat` and couples
every 20 steps. Profile extrapolation modes remain in the profile script.
Core volume integration uses an 8x refined cubic equilibrium map;
the old coarse-grid state files consequently do not match the new grid.

For a quantitative pulse comparison, add `export /path/to/core_inputs.json`
to the fix command and use `run 0` to initialize and export without moving
particles. Repeat with the desired `grid` and `depth` settings. The JSON
contains the actual grid, psi mapping, SI plasma profiles, transport and
charge-rate arrays, volumes, deposition weights, and boundary decay length
at 17-digit precision. It is rewritten on each run initialization and does
not contain a restart state or alter the solver inputs.

Convert an export for the existing SPARC pulse driver with:

```
python scripts/make_exported_pulse.py core_inputs.json case_fix.dat --charge 30 --duration 0.005 --window 1e-5
```

The converter copies the rate/grid/deposition arrays and constructs a unit
pulse using the exported per-length volumes. It rejects nonzero convection,
which the existing `core_scan/pulse_driver.cpp` does not read. No geometry
or profile reconstruction is performed. This export does not select a
physical deposition depth; that remains an explicit model input.

Verifies `src/core_transport_1d.cpp` (`CoreTransport1D`), the Linder finite-volume scheme
(Linder et al., NF 60 (2020) 096031) that `fix core/transport` will advance inside the
psi_N interface: cylindrical geometry in the volume radius, implicit, two half steps sweeping
the charge states up then down, one tridiagonal solve per state and half step.

    ./run.sh                      # builds scripts/test_driver.cpp, runs 5 cases, PASS/FAIL, exit code
    AURORA=/path/to/Aurora ./run.sh   # also rebuilds reference/ with Aurora's impden1 (needs gfortran)

Cases (scripts/make_inputs.py, Aurora units cm/s, 61-point grid, 5-state chain, D(r), inward v(r),
bell-shaped S/R, edge neutral source, parallel loss outside 40 cm):

| case | neutrals | boundary | checked against |
|---|---|---|---|
| neutral | fixed density n0 | decay length 1 cm | Aurora impden1 (default mode) |
| evolve | transported, volume source n0 | decay length 1 cm | Aurora impden1, padded fixed state |
| closed | transported | closed (dlen 0) | Aurora impden1, padded fixed state |
| source | transported, per-state volume source on z=2 | decay length | conservation only |
| source_fixed_n0 | fixed n0 + per-state source | decay length | conservation only |

Checks: densities after steps 1/150/300 agree with Aurora to 3e-14 (round-off); the ledger
`N(end) = N(0) + dt sum(source - outflux - parloss)` closes to 1e-14, where `outflux`/`parloss`
are the per-state losses the solver reports from its own boundary and sink terms (what the
coupling fix re-emits into the Monte Carlo domain).

Reference generation: `ref_aurora/aurora_ref.f90` drives Aurora's `impden1` directly (MIT licence,
https://github.com/fsciortino/Aurora, nothing vendored). Aurora's `evolveneut` option is not used
as a reference: it keeps two neutral populations (`nt` from the source and `rn(:,1)` from
recombination) that both ionize the first ion state — the old neutrals are counted twice — and the
final neutral solve overwrites the recombined population every step. The C++ solver transports one
neutral population conservatively; when recombination to neutrals is zero this is exactly Aurora's
default mode with an extra padded state (S = 1, density = our source rate), which is how the
`evolve` and `closed` references are produced with recombination to neutrals switched on.

## Opt-in fix options (defaults unchanged)

    coefficients effective                         # default: dperp/vpinch are the r_vol operator's coefficients
    coefficients physical geometry <file>          # D_eff = g2 D_perp, v_eff = g1 v_perp
    solver substep                                 # default: reference substepped solver
    solver split max_dt <seconds>                  # implicit split, n = ceil(window/max_dt) internal steps

`physical` assumes D_perp constant on a flux surface and a pinch speed directed along the
flux-surface normal; g2 = <|grad r_vol|^2>, g1 = <|grad r_vol|> come from a geometry file
(rho_N r_vol g1 g2, e.g. `examples/wip/sparc/scripts/core_geometry.py`). The file must reach the
interface and agree with the psi map's volume radius there. Its comment
`axis_extrapolated_below_rho_N` marks the inner range where the factors are carried from the
first resolved shell, not measured; that range is exported.

`solver split` is first order in the internal step. `max_dt` is an accuracy control chosen
from a study (`output/test_split study <pulse case> [h_accept]`), not a universal value: for the
SPARC 200-cell 5 mm W30+ pulse, 0.6 us keeps the cumulative return within 0.005 and the returned
charge distribution within 0.01 (L1); 0.2 us has more margin.

The export (format v2) records the convention, the input coefficients, the factors, the effective
coefficients, the solver and its step limit; `make_exported_pulse.py` verifies that
effective = input x factor before writing a case. `python scripts/check_fix_options.py --exe <spa>`
runs the option checks through the fix on the SPARC smoke inputs.
