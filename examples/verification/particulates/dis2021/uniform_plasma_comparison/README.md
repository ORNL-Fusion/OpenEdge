# DUSTT-2005 versus DIS-2021 in a uniform plasma

This is a short model-comparison test, not a claim that the two models should
produce identical answers. One 1-micrometre boron grain starts from rest in the
same uniform deuterium plasma for both runs (`Te = Ti = 60 eV`, `ne = ni =
10^19 m^-3`, and `u_parallel = 20 km/s`). Secondary-electron and thermionic
emission are disabled so the test isolates the baseline charging, ion-drag, and
OML-heating closures.

The checker independently transcribes the relevant Pigarov (DUSTT-2005) and
Nespoli/Smirnov (DIS-2021) expressions. It verifies each engine result against
its own reference before checking that the expected model separation is
visible. A near-equal floating potential is expected in this negative-grain,
no-emission regime; equal drag and heat flux are not.

For this fixture, the reference equations give floating potentials only 0.106%
apart, while the DIS velocity kick is 2.84% larger and the DUSTT heat flux is
27.2% larger. These are diagnostic values for this controlled state, not
universal correction factors; the separation changes with the plasma and grain
state.

| Quantity | `dustt2005` | `dis2021` |
|---|---|---|
| Charging | Legacy negative-grain OML balance | Signed DIS current balance |
| Ion drag | Pigarov et al. (2005), Eq. 16 | Smirnov et al. (2007) exact OML collection plus scattering |
| OML heating | Fixed 2.5-temperature energy factors | DIS velocity-dependent OML energy moments |
| Emission outside this test | Legacy thermionic treatment and experimental Sternglass SEE | Potential-aware thermionic recollection and named Kollath/Smirnov SEE |
| Evaporation endpoint outside this test | Stops at `R/R0 = 0.1` and records the 0.1% remainder | Continues to the configured numerical mass tolerance |

Run from this directory:

```bash
./run.sh /path/to/spa_mac_mpi
```

`OPENEDGE_BIN`, `PYTHON`, and `NP` can also select the executable, Python
interpreter, and MPI rank count. The run prints deterministic PASS/FAIL gates
and writes a compact `output/comparison.csv` summary.

The case deliberately uses `R/lambda_D < 0.1`, so both OML closures are being
compared inside their small-grain validity range. It does not exercise positive
grain potential, secondary emission, thermionic emission, breakup, or the two
different evaporation endpoints; those require dedicated branch tests.

References: A. Yu. Pigarov et al., *Physics of Plasmas* **12**, 122508
(2005); F. Nespoli et al., *Physics of Plasmas* **28**, 073704 (2021); and
R. D. Smirnov et al., *Plasma Physics and Controlled Fusion* **49**, 347
(2007).
