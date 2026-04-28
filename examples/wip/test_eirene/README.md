# test_eirene: OpenEdge vs EIRENE standalone benchmark

## Goal

Compare OpenEdge neutral transport against EIRENE standalone for the
**same 4-reaction D2/D/D+ system** in a **0D box** (periodic, uniform plasma).
This is a volumetric chemistry-only test -- no geometry, no walls, no
sheath, no elastic scattering.  The point is to validate that the
reaction rate implementation matches before adding any complexity.


## Status: READY TO START (2026-04-11)


## Directory layout

```
test_eirene/
  README.md                    <-- you are here
  openedge/
    in.eirene_compare          <-- OpenEdge input (READY)
    plasma.species             <-- D2, D, D+ species
    plasma.reactions           <-- 4 reactions with ADAS/Janev rates
  eirene/
    (EIRENE input goes here -- see Step 2 below)
  output/
    (comparison plots go here)
  compare_inventory.py         <-- post-processing script (READY)
```


## What to do tomorrow (step by step)

### Step 0: Verify rate coefficients match

**DO THIS FIRST** before running anything.  If the rate data disagrees,
the inventory curves won't match regardless of the transport.

At Te = 20 eV, ne = 1e19 m-3, compute <sigma*v> for each channel from
both ADAS (OpenEdge) and HYDHEL/AMJUEL (EIRENE):

| Reaction           | OpenEdge source              | EIRENE source             |
|--------------------|------------------------------|---------------------------|
| D ionization       | ADAS SCD (ADAS_Rates_1.h5)   | AMJUEL H.4 2.1.5 (EI)    |
| D recombination    | ADAS ACD (ADAS_Rates_1.h5)   | AMJUEL H.4 2.1.8 (RC)    |
| D+ CX             | ADAS CCD (ADAS_Rates_1.h5)   | HYDHEL H.1 3.1.8 (CX)    |
| D2 dissociation    | Janev polynomial (H.2 2.2.5) | HYDHEL H.2 2.2.5 (DS)    |

For the Janev polynomial (dissociation), evaluate at Te=20:
  ln(Te) = ln(20) = 2.9957
  ln<sv> = sum of b_n * (ln Te)^n   (coefficients in plasma.reactions)
  <sv> in cm3/s, convert to m3/s by multiplying by 1e-6

For ADAS rates, read from the HDF5 file:
```python
import h5py
f = h5py.File('../../database/adas/ADAS_Rates_1.h5', 'r')
# Look at datasets: IonizRateCoeff, RecombRateCoeff, ChargeExchangeRateCoeff
# Each has Te_grid, ne_grid, and rate_coeff arrays
# Interpolate at Te=20 eV, ne=1e19 m-3
```

For HYDHEL/AMJUEL rates, they're polynomial fits in the .tex files:
```
/home/cloud/local/solps/solps-iter-develop/modules/Eirene/Database/AMdata/hydhel.tex
/home/cloud/local/solps/solps-iter-develop/modules/Eirene/Database/AMdata/amjuel.tex
```
Search for "H.4 2.1.5", "H.4 2.1.8", "H.1 3.1.8", "H.2 2.2.5"

**Expected agreement: within 5-10% for ionization and CX.  Recombination
at 20 eV is very small (order 1e-19 m3/s) so absolute differences there
are fine.  Dissociation should agree well since both use Janev.**


### Step 1: Run the OpenEdge side

```bash
cd /path/to/OpenEdge/examples/test_eirene/openedge/
mpirun -np 1 /path/to/spa_mpi -in in.eirene_compare | tee ../output/openedge.log
```

Output goes to stdout.  The stats lines have columns:
  Step | CPU | np | N_D | N_D2 | N_D+

Save this to a file:
```bash
grep "^[0-9]" ../output/openedge.log > ../output/openedge_inventory.dat
```

What to check immediately:
- D2 count should decrease (dissociation)
- D count should initially increase (from dissociation), then decrease (ionization)
- D+ count should increase (from ionization)
- Total particles should be approximately conserved
  (not exactly, because dissociation creates particles: 1 D2 -> 2 D)
- At Te=20 eV, ne=1e19, ionization is fast (~1e-14 m3/s * 1e19 = 1e5 /s)
  so ionization timescale ~ 10 us.  In 1 ms you should see nearly
  complete ionization.


### Step 2: Set up the EIRENE standalone side

This is the harder part.  EIRENE input format is fixed-format Fortran
(column-sensitive) and very verbose.

**Option A (recommended): Adapt the cylinder test case**

Start from:
```
/home/cloud/local/solps/solps-iter-develop/modules/Eirene/scripts/eirenex_v1.0.4/examples/cylinder/test_1_complete/test_1.in
```

What to strip out:
- Block 4: Reduce 155 reactions to just 4:
    1  HYDHEL H.1 3.1.8    CX   1  1   (D+ + D -> D + D+)
    2  AMJUEL H.4 2.1.5    EI   0  1   (e + D -> D+ + 2e)
   61  AMJUEL H.4 2.1.8    RC   0  1   (D+ + e -> D)
   18  HYDHEL H.2 2.2.5    DS   0  2   (e + D2 -> 2D + e)  ** check this ID **
  NOTE: The cylinder test uses H.2 2.2.10 for dissociation (ionization of D2),
  not H.2 2.2.5 (dissociation into 2 D atoms).  Make sure you pick the
  right reaction -- 2.2.5 is pure dissociation, 2.2.10 is dissociative ionization.

- Block 4a: Keep only D atom (species 1), strip He, N, N*, Ar
- Block 4b: Keep only D2 molecule (species 1)
- Block 2: Simplify geometry to 1D slab
- Block 5b: Set constant plasma profiles using INDPRO=3 (step function)
  with SEP set to a very large radius so the profile is constant everywhere:
    Te: 20 eV  (TE0=20, TE1=20, SEP=1e99)
    Ti: 20 eV
    ni: 1e19 m-3
    flow: 0

**Option B (if you know someone with EIRENE experience): Ask for help**

The EIRENE input format is tricky.  If someone at ORNL or a collaborator
has run EIRENE standalone before, ask them for a minimal D2 test case.
People who have done this: Detlev Reiter (Juelich), or anyone who has
used EIRENE's eirenex tools.

**Option C: Use EIRENE via SOLPS but disable B2**

Some SOLPS users run EIRENE "standalone" by setting up a SOLPS case with
the B2 plasma frozen and only running EIRENE iterations.  This avoids
dealing with the raw EIRENE input format.  The slab standalone example
might be relevant:
```
/home/cloud/local/solps/solps-iter-develop/runs/examples/test_slab_ortho_standalone.tar.gz.md5
```
But this is just an MD5 -- the actual tarball needs to be downloaded.
Check the runs/examples/Makefile for how to get it.


### Step 3: Run the EIRENE side

```bash
cd /path/to/OpenEdge/examples/test_eirene/eirene/
/home/cloud/local/solps/solps-iter-develop/modules/Eirene/builds/standalone.ORNL.gfortran/eirobj < input.dat > eirene.log
```

EIRENE output: Look for particle tallies in the output.  EIRENE reports
volume-averaged tallies (TRCL arrays) for neutral density, ionization
source, etc.

**IMPORTANT: EIRENE time-dependent vs steady-state modes**

EIRENE DOES have a time-dependent mode.  It's controlled by Block 13
of the input file:
  - NTIME > 0 : enables time-dependent mode (number of time cycles)
  - NPRNLI > 0: number of census particles (stored between time steps)
  - DTIMV      : time step size [s]
  - TIME0      : initial time [s]
  - NPTST      : census relaunch mode (-1 = one-by-one)

In time-dependent mode, EIRENE stores a "census" of particles at the
end of each time step and relaunches them at the start of the next.
This is genuine time-dependent Monte Carlo (not just iterating to
steady state).

However, when EIRENE is coupled to B2.5 in SOLPS, the coupling is
always steady-state: B2.5 calls EIRENE, gets back sources, updates
plasma, calls EIRENE again, etc.  The time-dependence in B2.5 is
handled by B2.5's own time stepping, not EIRENE's internal clock.

**For this benchmark, USE EIRENE's time-dependent mode** so you get
a true apples-to-apples comparison of the D2->D->D+ evolution:
  - Set NTIME = 10000 (number of time steps)
  - Set DTIMV = 1e-7 (same dt as OpenEdge)
  - Set NPRNLI = 2000 (same number of census particles as OpenEdge)
  - Compare species inventory at each time step

If time-dependent mode is too hard to set up on day 1, the fallback
is to compare OpenEdge's steady-state (after equilibration at ~1 ms)
against EIRENE's default steady-state tallies.  But the time-dependent
comparison is much more informative.


### Step 4: Post-process and compare

Run the comparison script:
```bash
cd /path/to/OpenEdge/examples/test_eirene/
python3 compare_inventory.py
```

This produces:
  output/species_inventory.png   -- D2, D, D+ counts vs time (OpenEdge)
  output/rate_comparison.txt     -- rate coefficients from both codes

If EIRENE gives steady-state tallies, add them as horizontal lines
on the time-history plot.


## Key caveats and gotchas

### 1. Rate data sources differ
OpenEdge uses ADAS ADF11 (SCD/ACD/CCD) tables.  EIRENE uses AMJUEL/HYDHEL
polynomial fits (or optionally ADAS via the CFILE ADAS card).  At 20 eV
these should agree to a few percent but CHECK FIRST.

### 2. EIRENE CX is not the same as OpenEdge CX
EIRENE CX (HYDHEL H.1 3.1.8): D + D+ -> D+ + D (atom is the test particle)
OpenEdge CX (ADAS CCD):        D+ + D -> D + D+ (ion is the test particle)
The rate coefficient is the same, but the bookkeeping of which particle
gets replaced differs.  In a 0D box with periodic BC this shouldn't matter
for inventory, but it affects individual particle trajectories.

### 3. D2 dissociation products
After D2 -> D + D, both products get velocity sampled from what?
- OpenEdge: shifted Maxwellian at local Ti/vpar if available, else
  Franck-Condon energy (~3 eV per atom, isotropic)
- EIRENE: similar, but check the specific reaction card for energy partitioning
This doesn't affect inventory in a 0D box, but matters for spatial transport.

### 4. Particle statistics
OpenEdge: 2000 simulation particles * fnum = 2e18 real particles in a
1 m^3 box -> effective density = 2e18 m-3.  The prescribed ne = 1e19 m-3
is the ELECTRON (background plasma) density, not the neutral density.
So the initial neutral-to-plasma ratio is D2/ne ~ 0.2.

### 5. Time-dependent vs steady-state
EIRENE has a time-dependent mode (Block 13: NTIME, NPRNLI, DTIMV)
that uses census particles between time steps.  This is what you
want for an apples-to-apples comparison.  The steady-state mode
(NTIME=0) is what B2.5-EIRENE coupling normally uses, but that's
because B2.5 owns the time stepping.  In standalone mode you can
(and should) use EIRENE's own time-dependent capability.
See Step 3 notes above.

### 6. What NOT to do yet
- Don't add wall recycling (test_neutral_transport/wall.recycle) until
  volume chemistry matches
- Don't add elastic scattering (collide vss) unless you also enable it
  in EIRENE with the same cross section
- Don't use a complicated geometry
- Don't include the full EIRENE hydrogenic molecular package
  (vibrationally-resolved H2, H2+, H3+, etc.)


## EIRENE references for input format

- Profile functions (INDPRO): see
  /home/cloud/local/solps/solps-iter-develop/modules/Eirene/src/startup-routines/plasma.f
  /home/cloud/local/solps/solps-iter-develop/modules/Eirene/src/startup-routines/eirmod_profiles.f

- Input format documentation: see
  /home/cloud/local/solps/solps-iter-develop/modules/Eirene/src/startup-routines/read_fixform.F
  (heavily commented, this IS the documentation)

- Reaction format: see lines 78-236 of the cylinder test_1 input

- EIRENE executable:
  /home/cloud/local/solps/solps-iter-develop/modules/Eirene/builds/standalone.ORNL.gfortran/eirobj

- EIRENE databases:
  /home/cloud/local/solps/solps-iter-develop/modules/Eirene/Database/AMdata/hydhel.tex
  /home/cloud/local/solps/solps-iter-develop/modules/Eirene/Database/AMdata/amjuel.tex

- EireneX post-processing manual:
  /home/cloud/local/solps/solps-iter-develop/modules/Eirene/scripts/eirenex_v1.0.4/eirenex_manual.pdf


## How EIRENE vs OpenEdge transport works (reference)

### Neutral transport

Both codes trace neutrals as **straight-line free flights** (no forces,
no B-field interaction).  Neutrals fly until they hit a volumetric
collision event (sampled from the mean free path) or a surface.

- **EIRENE** (`folneut.F`): standard Monte Carlo neutral transport.
  Straight-line flight, collision sampling from cumulative reaction
  probability along the path.  At collision: ionization, CX,
  dissociation, or elastic scattering.
- **OpenEdge** (DSMC/SPARTA): same physics but time-stepped.  Particles
  advance by dt each step, then volumetric reactions are evaluated
  probabilistically per cell per step (`fix chem/adas`).

For the 0D box benchmark these are equivalent -- both just sample
reaction events from the same rate coefficients.

### Charged particle transport

**EIRENE does NOT use a Boris pusher.**  The ion transport model in
EIRENE is fundamentally different from OpenEdge:

**EIRENE (`folion.F`):**
- **Guiding center approximation** -- decomposes velocity into
  v_parallel and v_perp relative to B-field
- Pushes ions **along the B-field line only** (v = v_parallel,
  direction = B-hat).  Full gyromotion is NOT resolved.
- Optional guiding center drifts (ExB, grad-B) via explicit Euler,
  but rarely used in practice
- **Coulomb collisions** via Fokker-Planck energy relaxation:
  - Langer model (analytical slowing-down frequency)
  - Trubnikov (semi-analytical, more accurate)
  - Takizuka binary collision operator
  - Hybrid particle-fluid Fokker-Planck
- Step size: ~0.1 * tau_E (energy relaxation time), so roughly
  10 Coulomb collisions per relaxation time
- Source: `Eirene/src/particle-tracing/folion.F`

**OpenEdge:**
- **Boris pusher** with configurable subcycles, resolves full
  gyromotion (or optionally GCA with RK4 and Littlejohn corrections)
- Coulomb collisions via **Nanbu/Takizuka-Abe** binary collision
  operator (`fix coulomb/background`)
- Sheath models (kick or spatial E-field)
- Thermal forces, cross-field diffusion as separate fixes

**Why this doesn't matter for the 0D benchmark:**
In a periodic box with uniform plasma and no B-field, ion transport
is irrelevant.  Ions created by ionization just sit in the box.  The
comparison is purely about volumetric reaction rates.  The Boris vs
guiding-center difference only matters once you move to a geometry
where ion transport competes with reactions (Step 4+ in the roadmap).

**Why this matters later:**
When you move to real geometry, EIRENE's guiding-center ion tracing
is much cheaper per step (no subcycling) but less accurate near
surfaces and in strong-gradient regions.  OpenEdge's Boris pusher
resolves the gyromotion and gets correct IEADs at walls, but costs
more per step.  This is one of OpenEdge's advantages over EIRENE
for PMI studies.


## After the 0D test passes

Once volume chemistry agrees, the next steps (in order):

1. Add wall recycling (wall.recycle from test_neutral_transport)
2. Add a 1D slab with absorbing walls -> compare neutral penetration depth
3. Add elastic neutral-neutral scattering (collide vss in OpenEdge,
   EL reactions in EIRENE) with matched cross sections
4. Move to 2D tokamak-like geometry with real plasma profiles
5. Full divertor case (much later)
