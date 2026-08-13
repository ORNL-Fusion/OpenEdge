# RFPIE tungsten sputtering in OpenEdge

This is a clean OpenEdge reconstruction of the RFPIE target case. It uses the
original RFPIE CAD and Langmuir-probe measurements as provenance, but none of
the old GITR/FTRIDYN surface tables or GITR transport setup.

The first model is deliberately explicit:

- background: measured He plasma radial profiles, folded/averaged across the
  two probe scans with their uncertainties retained; quasineutral He+ and
  uniform in `z` until better measurements are supplied. The source Excel
  chart labels position in cm; a legacy script interpreted it as mm, which
  caused the apparent kink at 5 mm. The actual scan reaches 50 mm, beyond the
  31 mm case domain. Density is shape-preserving; `Te` uses a smooth
  even-polynomial fit. Slope-matched tails exist beyond the measured edge but
  are unused in this domain;
- ion temperature: uniform `Ti = 0.05 eV`, the explicit hPIC2 assumption used
  for the RF PIE sheath study by Caughman et al., IEEE TPS 52 (2024),
  DOI `10.1109/TPS.2024.3374252`. This is an assumption, not an LP measurement;
- source: He-on-W sputter yield from RustBCA installed in OpenEdge
  `processes.h5`, with Thompson W emission (`Us = 8.68 eV`, `Emax = 80 eV`);
- target voltage: the target remains one watertight surface and each
  plasma-facing triangle carries a custom surface vector. The coarse STL top
  is conformingly retiled into radial rings so centroid wall-flux sampling
  resolves the peaked measured plasma profile;
  `sheath = [Vdc, Vrf_peak, phase_rad]`;
- transport: W, W+, W2+, W3+, and W4+ with ADAS charge-state evolution;
- diagnostics: pweight-aware, interval-averaged density for every W charge
  state, plus particle and surface-source dumps. The notebook plots total W in
  R-z, charge-state profiles along the axis, in-flight column density, gross
  erosion fluence, and equivalent removed W depth. The authoritative
  charge-resolved result is `output/rfpie_w_density.*.dump`; no separate
  spectroscopy or emission handoff is generated;
- production sampling: `64 x 64 x 96` Cartesian cells, 100 launched markers
  per step, per-cell/species roulette control at 200 markers, and two-stage
  RCB balancing (`part` during plume startup, then measured `time` weights);
- sheath: the target-local `boundary` model is a thin, unresolved potential
  sheet for kinetic W ions. It is not a mesh boundary and does not resolve the
  Debye-sheath field profile.

The chamber STL is retained and plotted as provenance, but the first transport
deck uses OpenEdge's open Cartesian box as its outer boundary. Reading both the
old domain shell and the target creates coplanar surfaces at `z=0`, which the
3-D cut-cell algorithm correctly rejects. The material target itself remains
the collision geometry.

## Build and inspect

From this directory, with the OpenEdge Python environment active:

```sh
python scripts/build_geometry.py
python scripts/build_plasma.py
python scripts/build_he_on_w.py
python scripts/install_he_on_w.py /Users/42d/OpenEdge/database/processes.h5
python scripts/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace rfpie_w.ipynb
python scripts/check_case.py
```

Open `rfpie_w.ipynb` before running transport. It checks the raw and reconstructed
plasma, the CAD-to-SI conversion, target tiles/voltage waveform, RustBCA yield,
the implied radial W source, and the legacy factor-of-ten unit error. After
a run, rerunning the notebook adds W density, charge-state, and areal-density
diagnostics from `output/rfpie_w_density.*.dump`.

## Run

After building OpenEdge with the RF sheath patch:

```sh
time mpirun -np 4 /Users/42d/build_oe/src/spa_mac_mpi -in in.rfpie_w
```

Defaults are `Vdc=-500 V`, `Vrf=0`, and 64 phase samples. Edit
`case_config.json`, rebuild `target_face.surf`, and keep `rffreq` in both input
files synchronized when changing frequency. For time-resolved sputtering use
one phase sample and disable the PMI cache; the timestep must resolve the RF
period. The production default phase-averages the nonlinear yield while the
pusher uses the instantaneous tile voltage.

The DC production defaults are `dt=20 ns`, 10,000 total steps (`200 us`), and
100 launched W markers per step. The 20 ns step passed OpenEdge's
`bad_dt_check`; at 80 eV a W atom moves less than 0.15 of the smallest grid
cell per Boris substep (0.29 per full step, with two subcycles). This run spans
about 16 ballistic mean-energy plume transit times. For a nonzero,
time-resolved 13.56 MHz RF voltage, override `dt` back to about `2 ns`; 20 ns
provides only 3.7 steps per RF period and is not adequate for RF phase
resolution.

The DC case is the quantitatively cleaner starting point. With nonzero RF,
the present sputter source averages yield over instantaneous sinusoidal wall
voltages. It does not reproduce the bimodal kinetic IEDF reported by hPIC2;
that will require an IEDF table or a resolved kinetic sheath model.

