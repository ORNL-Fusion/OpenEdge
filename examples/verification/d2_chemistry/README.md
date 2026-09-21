# D2 chemistry (0-D)

Uniform-plasma test of the hydrogen volumetric chemistry chain:
D2 electron-impact dissociation (Janev/HYDHEL polynomial, creates the
second D atom with a back-to-back ~3 eV Franck-Condon kick), D
ionization (AMJUEL H.4 2.1.5), D-D+ resonant charge exchange (HYDHEL
H.3 3.1.8 rate + sigma*v-weighted partner sampling).

Expected behavior at Te = Ti = 8 eV, ne = 1e19 m^-3: the 20000 D2
macroparticles fully dissociate (np doubles to 40000), the D population
partially ionizes (~18% D+ at 2000 steps), CX resamples D velocities.

Run (CPU or Kokkos):
    spa_* -in in.d2_chem
    spa_kokkos_* -k on t 1 -sf kk -pk kokkos react/retry yes -in in.d2_chem

Under `-sf kk` the chem fix prints "device path active (Z=1, 5
reactions)"; OE_CHEM_HOST=1 forces the host path for A/B comparison.
