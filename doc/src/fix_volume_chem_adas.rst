.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix volume/chem/adas

fix volume/chem/adas command
============================

Syntax
------

.. parsed-literal::

   fix ID volume/chem/adas Nevery element reactions keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* volume/chem/adas = style name of this fix command
* Nevery = attempt reactions every this many timesteps
* element = element symbol (for example D, C, W) or atomic number of
  the reacting element
* reactions = source of the reaction list

  * a file path (a word that contains "/" or ends in *.reactions*) is
    read as a reaction file
  * an element symbol selects the reaction catalog of that element in
    the process database
  * *auto* = the same as repeating the *element* argument

* zero or more keyword/value pairs may be appended
* keyword = *mode* or *rate_cache* or *source_species* or
  *stop_on_exhaust* or *exhaust_threshold* or *units* or *output* or
  *volume_source* or *ionization* or *recombination* or *cx* or
  *dissociation*

  * *mode* value = *kinetic* or *neutral*

    * *kinetic* = an ionized particle becomes the product ion species
    * *neutral* = an ionized particle is deleted

  * *rate_cache* value = *particle* or *cell*

    * *particle* = evaluate the rate coefficients for every particle
    * *cell* = evaluate them once per cell and species and reuse them

  * *source_species* values = sp1 sp2 ... = species that this fix
    counts as transported: they are monitored for exhaustion, kept as
    particles when they are charged dissociation products, and given a
    density column with *units flight*
  * *stop_on_exhaust* value = *yes* or *no*
  * *exhaust_threshold* value = Nmin = stop when the number of
    *source_species* particles has fallen to Nmin or below
  * *units* values = *counts* or *rate* or *flight* or *batch* Nb Rpuff
    or *batch_fix* emit-ID Rpuff

    * *counts* = cumulative event tallies per simulation particle
    * *rate* = source densities in SI units over the last
      :math:`N` steps
    * *flight* = cumulative source densities of a single-flight batch
    * *batch* Nb Rpuff = each of Nb trajectories carries Rpuff/Nb
      (1/s)
    * *batch_fix* emit-ID Rpuff = as *batch*, with Nb taken from the
      cumulative particle count of the emit fix emit-ID

  * *output* value = *summary* or *detailed*
  * *volume_source* value = ID of a
    :doc:`compute plasma/fields <compute_plasma_fields>` or
    :doc:`fix background <fix_background>` that supplies the plasma for
    volume recombination of the background ions
  * *ionization*, *recombination*, *cx*, *dissociation* value = *yes*
    or *no* = turn all reactions of that type on or off

Examples
--------

.. code-block:: text

   fix pd background file plasma.h5 static yes
   global bfield_compute pd
   fix fchem volume/chem/adas 1 D auto mode neutral units rate
   fix fchem volume/chem/adas 1 W auto cx no
   fix fchem volume/chem/adas 1 D input/neutral.reactions mode neutral &
       source_species D D2 stop_on_exhaust yes exhaust_threshold 100 &
       units batch_fix fpuff 1.0e21 volume_source pd

Description
-----------

Apply ionization, recombination, charge exchange and dissociation to
the simulation particles with rate coefficients from the OpenEdge
process database, *database/processes.h5*.  The reactions that a
particle can undergo compete as Poisson processes over the interval
:math:`\Delta t_N`.  The fix also tallies, per grid cell, the particle,
momentum and energy that the reactions exchange with the background
plasma.

Symbols not defined here are on the :ref:`notation <notation>` page.

Reaction probability
^^^^^^^^^^^^^^^^^^^^

Let :math:`j` run over the active reactions whose reactant is the
species of the particle.  Each reaction has an expected number of
events in one application of the fix,

.. math::
   :label: chem-lambda

   \lambda_j = k_j\, n_j\, \Delta t_N , \qquad \Delta t_N = N\,\Delta t ,

where :math:`k_j` is the rate coefficient (m\ :sup:`3`/s) and
:math:`n_j` the density of the collision partner.  A particle in
charge state :math:`Z` has

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Type
     - :math:`k_j\, n_j`
     - Rate coefficient
   * - ionization
     - :math:`S_Z(T_e,n_e)\; n_e`
     - ADAS SCD, charge state :math:`Z \to Z+1`
   * - recombination
     - :math:`\alpha_Z(T_e,n_e)\; n_e`
     - ADAS ACD, charge state :math:`Z \to Z-1`
   * - charge exchange, impurity
     - :math:`C_Z(T_e,n_e)\; n_0`
     - ADAS CCD, :math:`Z \to Z-1` on background hydrogen atoms of
       density :math:`n_0`
   * - charge exchange, hydrogen
     - :math:`\langle\sigma v\rangle_\mathrm{cx}(T_i,E)\; n_e`
     - HYDHEL H.3 3.1.8 fit for resonant exchange with the background
       ions, :math:`E = m v^2/2` the energy of the atom
   * - dissociation
     - :math:`k_\mathrm{d}(T_e)\; n_e`
     - polynomial fit given in the reaction file, Eq. :eq:`chem-janev`

Each :math:`\lambda_j` is limited to 50.  With
:math:`\Lambda = \sum_j \lambda_j`, the particle reacts during the
interval with probability

.. math::
   :label: chem-prob

   P = 1 - e^{-\Lambda} .

A reaction occurs if :math:`U < P`.  A second random number then
selects reaction :math:`j` with probability :math:`\lambda_j/\Lambda`.
A particle undergoes at most one reaction per application, so
:math:`\Delta t_N` should stay below the shortest reaction time
:math:`1/(k_j n_j)` wherever a chain of several reactions matters.  A
species can have at most 16 reactions.

The statistical weight of a particle does not enter
:math:`\lambda_j`.  A reaction changes the whole simulation particle,
and the weight enters only the source tallies described under
:ref:`Output <chem-adas-output>`.

Rate coefficients
^^^^^^^^^^^^^^^^^

The ADAS tables hold :math:`\log_{10}` of the coefficient on a grid of
:math:`\log_{10} T_e` and :math:`\log_{10} n_e`, one table per charge
state.  The fix interpolates bilinearly in these logarithms.  Outside
the grid the outermost interval is extrapolated linearly.  If the
database has no table or no row for a requested charge state, the rate
is zero and a warning is printed once.

Two hydrogen rates do not come from the ADAS tables.  When the element
is hydrogen (H, D or T), ionization of the atom uses the AMJUEL H.4
2.1.5 collisional-radiative fit in :math:`T_e` and :math:`n_e`, and
charge exchange uses the HYDHEL H.3 3.1.8 fit of
:math:`\langle\sigma v\rangle_\mathrm{cx}` in :math:`T_i` and the
energy of the atom.  For any other element the charge-exchange partner
is the neutral hydrogen of the background.  Its density :math:`n_0` is
read at the cell centre from the neutral density of the first
:doc:`fix background <fix_background>` in the input script.  The
channel is skipped where that density is absent or zero.

A dissociation rate is the polynomial

.. math::
   :label: chem-janev

   \ln k_\mathrm{d} = \sum_{n=0}^{8} b_n\, (\ln T_e)^n ,

with :math:`T_e` in eV, :math:`k_\mathrm{d}` in cm\ :sup:`3`/s and the
nine coefficients :math:`b_n` given in the reaction file.

With *rate_cache particle* the :math:`\lambda_j` are evaluated for
every particle.  With *rate_cache cell* they are evaluated once per
cell and species, from the plasma values of the first particle of that
species found in the cell, and reused for the others.  The hydrogen
charge-exchange rate is still evaluated for each atom with its own
energy.  The two settings agree where :math:`T_e` and :math:`n_e` do
not vary across a cell.  They differ in sheath cells, where the
electron density depends on the distance from the wall.

Plasma parameters
^^^^^^^^^^^^^^^^^

:math:`T_e` and :math:`n_e` are read from the per-particle plasma
cache, which the particle mover fills at the particle positions.  The
cache needs a provider, a :doc:`fix background <fix_background>` or a
:doc:`compute plasma/fields <compute_plasma_fields>`, and it must be
switched on by the *bfield_compute* option of the
:doc:`global <global>` command, by the sheath model or by the
guiding-centre pusher.  A run with neutral particles only still needs
one of these, for example *global bfield_compute pd*.

When the sheath model of the :doc:`global <global>` command is active
with its surface-geometry compute, the cached density of a particle
inside the sheath is the provider value multiplied by the Boltzmann
factor :math:`\exp(-\phi/T_e)`.  Here :math:`\phi > 0` is the
potential drop of the sheath model at the distance of the particle
from the nearest surface element.  Farther from the wall than the
sheath extent the provider value is kept.

This fix always requests :math:`T_e` and :math:`n_e` from the cache.
It requests :math:`T_i`, :math:`u_\parallel` and :math:`\mathbf{B}`
only if the reaction list contains a charge-exchange reaction, whether
or not *cx no* switches it off.  These three are used for the
charge-exchange rate, for the velocity of the product and for the
momentum and ion-energy entries of recombination in the source
tallies.  If the list has no charge-exchange reaction they are
refreshed only when another command requests them, so the
recombination entries of :math:`\mathbf{S}_m` and :math:`Q_i` should
then not be used.  A particle is skipped where :math:`T_e` or
:math:`n_e` is not positive.

Reaction list
^^^^^^^^^^^^^

The *element* argument names the reacting element.  The isotope labels
H, D and T select the same tables.  The *reactions* argument is
resolved in this order.  A path is read as a reaction file.  An
element symbol, or *auto* for the *element* argument, is looked up as
a reaction catalog in the process database.  If the database has no
catalog for the element, the file
*database/ingest/reactions/<elem>.reactions* is read.

A reaction file lists each reaction on two lines.  Blank lines and
lines that start with "#" are skipped.  The first line is the reaction
formula.  The second line holds the type, the rate style, the rate
coefficients and optional energies:

.. code-block:: text

   D --> D+
   I A 1.0 0.0 0.0 0.0 0.0

   D2 --> D + D
   D J -2.787e+01 1.052e+01 -4.973e+00 1.451e+00 -3.063e-01 &
       4.433e-02 -4.096e-03 2.160e-04 -4.929e-06 fc 3.0 eel 4.478

The continuation character is shown for layout only.  In the file the
second line is a single line.

The type is *I* for ionization, *R* for recombination, *E* for charge
exchange or *D* for dissociation.  The first three take one reactant
and one product.  A dissociation takes one reactant and one or two
products.  The style is *A* or *J*.  Style *A* takes the rate from the
ADAS tables or hydrogen fits described above, and its five
coefficients are placeholders that are read and ignored.  Style *J*
is the nine-coefficient fit of Eq. :eq:`chem-janev` and is used for
dissociation.

Two optional pairs may follow the coefficients of a reaction.  *fc*
sets the Franck-Condon energy :math:`E_\mathrm{fc}` (eV) and *eel* the
electron energy loss per event :math:`E_\mathrm{el}` (eV).  The
defaults are 3.0 eV and 4.478 eV.  The fix warns when an active
dissociation relies on a default.

Species names in a formula must match the :doc:`species <species>` of
the input script.  A reaction that involves an undefined species is
skipped at initialization and reported, so a charge-state ladder that
is truncated in the input script ends at its highest defined state.
The keywords *ionization*, *recombination*, *cx* and *dissociation*
switch off all reactions of one type without editing the list.  Two
reactions of the same type and style with the same reactant and
product are an error.

Products
^^^^^^^^

After ionization or recombination the particle takes the product
species and keeps its velocity.  With *mode neutral* an ionized
particle is deleted instead, which gives a neutral-only calculation of
the kind done by EIRENE.

After charge exchange the particle takes the product species and the
velocity of the background ion it exchanged with.  That velocity is
drawn from the Maxwellian of temperature :math:`T_i` drifting at
:math:`u_\parallel\hat{\mathbf{b}}`.  For hydrogen the draw is
accepted with probability proportional to
:math:`\sigma_\mathrm{cx}(E_\mathrm{rel})\,v_\mathrm{rel}`, where
:math:`v_\mathrm{rel}` is the speed of the ion relative to the atom
and :math:`\sigma_\mathrm{cx}` the HYDHEL H.1 3.1.8 cross-section, so
that fast partners are favoured as the collision rate requires.  If
:math:`T_i` is not available the velocity is kept.

In a dissociation with :math:`E_\mathrm{fc} > 0` the two products
leave back to back in the frame of the parent,

.. math::

   \mathbf{v}_{1,2} = \mathbf{v} \pm v_\mathrm{fc}\,\hat{\mathbf{n}} ,
   \qquad
   v_\mathrm{fc} = \sqrt{2E_\mathrm{fc}/m_1} ,

where :math:`\hat{\mathbf{n}}` is an isotropic random direction and
:math:`m_1` the mass of the first product.  The parent becomes the
first product and a new particle is created for the second.  A
charged product that is not listed in *source_species* is taken to
join the background ions.  It is added to the source tallies and no
particle is created for it.  If both products are of that kind the
parent is deleted.

A molecule whose only dissociation reactions give neutral products
draws a warning, because the molecular-ion channels are then missing.
A charged species listed in *source_species* that has no reaction is
an error, since its particles could never return to the plasma.

Exhaustion
^^^^^^^^^^

With *mode neutral*, *stop_on_exhaust yes* and a *source_species*
list, the fix counts the particles of the listed species over all
processors after each application.  Once the count has exceeded
*exhaust_threshold*, the run ends at the first application where it is
back at or below the threshold.  A threshold above zero cuts off the
slowly decaying tail of a batch.

Volume recombination
^^^^^^^^^^^^^^^^^^^^

With *mode neutral* no ions are followed, so recombination of the
background ions has to be added as a source of neutrals.  The
*volume_source* keyword does this.  In each cell the mean number of
new simulation particles per application is

.. math::

   \bar{N} = \frac{n_e\, n_i\, \alpha_1(T_e,n_e)\, V\, \Delta t_N}{F_\mathrm{num}} ,

where :math:`V` is the cell volume, :math:`\alpha_1` the ACD
coefficient of the singly charged ion and the plasma values are those
of the named source at the cell centre.  If the source has no ion
density or temperature, :math:`n_e` and :math:`T_e` are used.  The
number created is drawn from a Poisson distribution of mean
:math:`\bar{N}`, or from a normal distribution of mean and variance
:math:`\bar{N}` when :math:`\bar{N} \ge 30`.  Each new particle is the
product species of the first active recombination reaction.  It
starts at the cell centre with a velocity drawn from the Maxwellian of
temperature :math:`T_i` drifting at
:math:`u_\parallel\hat{\mathbf{b}}`.  Cells cut by a surface into
several parts are skipped.

.. _chem-adas-output:

Output
^^^^^^

The fix produces a per-grid array that can be used by
:doc:`dump grid <dump>` or :doc:`fix ave/grid <fix_ave_grid>`.  Its
rows follow the cells through load balancing and grid adaptation.

With *output summary* the array has six columns of sources to the
background plasma, summed over the reaction events in the cell.

.. list-table::
   :header-rows: 1
   :widths: 10 14 76

   * - Column
     - Quantity
     - Meaning
   * - 1
     - :math:`S_p`
     - particle source of the ion fluid
   * - 2-4
     - :math:`S_{m,x}, S_{m,y}, S_{m,z}`
     - momentum source, in the coordinate order of the simulation box
   * - 5
     - :math:`Q_e`
     - electron energy source
   * - 6
     - :math:`Q_i`
     - ion energy source

One event adds the following amounts, each multiplied by the event
weight set by *units*.  Here :math:`m` and :math:`\mathbf{v}` are the
mass and velocity of the reacting particle before the event,
:math:`\mathbf{v}'` its velocity after charge exchange, and
:math:`\mathbf{u} = u_\parallel\hat{\mathbf{b}}`.

.. list-table::
   :header-rows: 1
   :widths: 20 8 20 26 26

   * - Type
     - :math:`S_p`
     - :math:`\mathbf{S}_m`
     - :math:`Q_e`
     - :math:`Q_i`
   * - ionization
     - :math:`+1`
     - :math:`m\mathbf{v}`
     - :math:`-(I_Z + L_Z/S_Z)`
     - :math:`m v^2/2`
   * - recombination
     - :math:`-1`
     - :math:`-m\mathbf{u}`
     - :math:`-P_Z/\alpha_Z`
     - :math:`-(m u^2/2 + 3T_i/2)`
   * - charge exchange
     - 0
     - :math:`m(\mathbf{v} - \mathbf{v}')`
     - 0
     - :math:`m(v^2 - v'^2)/2`
   * - dissociation
     - :math:`n_\mathrm{b}`
     - :math:`\sum_k m_k \mathbf{v}_k`
     - :math:`-E_\mathrm{el}`
     - :math:`\sum_k m_k v_k^2/2`

:math:`I_Z` is the ionization potential of charge state :math:`Z`.
:math:`L_Z` is the ADAS PLT line-radiation coefficient of that state,
so :math:`L_Z/S_Z` is the energy radiated per ionization.  :math:`P_Z`
is the ADAS PRB coefficient for recombination and bremsstrahlung
radiation of the recombining ion, so :math:`P_Z/\alpha_Z` is the energy
radiated per recombination.  Both radiation coefficients are stored in
W cm\ :sup:`3` and interpolated like the rate coefficients.  A term
whose table is missing from the database is left out.  In the
dissociation row the sums run over the :math:`n_\mathrm{b}` products
that join the background ions.  A particle created by *volume_source*
is tallied as a recombination.

The sum :math:`L_Z + P_Z` in W m\ :sup:`3` is also what
:doc:`compute grid/weighted <compute_grid_weighted>` uses for its
*prad_w* and *pradtot_w* values.

With *output detailed* the array has 28 columns, seven quantities for
each of the four reaction types in the order ionization,
recombination, charge exchange, dissociation.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Columns
     - Quantity, summed over the events of each type
   * - 1-4
     - event weight (the number of events with *units counts*)
   * - 5-8
     - :math:`m v_x` of the reacting particle
   * - 9-12
     - :math:`m v_y`
   * - 13-16
     - :math:`m v_z`
   * - 17-20
     - :math:`m v^2/2`
   * - 21-24
     - :math:`Q_i` as in the table above
   * - 25-28
     - :math:`Q_e` as in the table above

Columns 1 to 20 are raw sums without sign.  Particles created by
*volume_source* add to columns 2, 6, 10, 14 and 18 only.

In both layouts one more column per *source_species* follows.  It holds
the density of that species (m\ :sup:`-3`) with *units flight* and is
zero otherwise.

The *units* keyword sets the weight of an event.  :math:`w` is the
*pweight* attribute of the reacting particle, the number of real
particles it stands for
(:doc:`fix particle/weight <fix_particle_weight>`), or
:math:`F_\mathrm{num}` if that attribute does not exist.

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - *units*
     - Event weight
     - Result
   * - *counts*
     - 1
     - sums per simulation particle since the start of the run:
       events, kg m/s, J
   * - *rate*
     - :math:`w/(V\,\Delta t_N)`
     - sources over the last application of the fix:
       m\ :sup:`-3` s\ :sup:`-1`, N/m\ :sup:`3`, W/m\ :sup:`3`.  The
       array is zeroed before each application.
   * - *flight*
     - :math:`w/(V\,\Delta t)`
     - cumulative.  When all particles of a batch have ended their
       flights, the array is the steady-state source of a wall source
       that emits one such batch per timestep.
   * - *batch*
     - :math:`R_\mathrm{puff}/(N_\mathrm{b} V)`
     - cumulative.  When all :math:`N_\mathrm{b}` trajectories have
       ended, the array is the steady-state source of a puff of
       :math:`R_\mathrm{puff}` atoms per second.
   * - *batch_fix*
     - :math:`R_\mathrm{puff}/(N_\mathrm{b} V)`
     - as *batch*, with :math:`N_\mathrm{b}` read at each application
       from the cumulative particle count of the emit fix

With *rate* the array holds one sample per application and is meant
to be averaged with *fix ave/grid ... ave running*.  With *flight* the
density columns accumulate :math:`w/V` over the particles of each
source species at every application.  The cumulative tallies are
zeroed at the start of each :doc:`run <run>`.

At initialization the fix prints the number of active and skipped
reactions of each type and the ADAS tables it found.  At the end of
each run it prints the number of events per type and per reaction.

The fix acts at the end of the timestep, after the particle move.

Restrictions
------------

This command is part of the OPENEDGE package.

The per-particle plasma cache must be active, as described above.
Otherwise the fix stops with an error at its first application.

The process database must exist and contain, for the element, the SCD
and ACD tables and the ionization potential :math:`I_Z` of every
charge state.  Otherwise the fix stops with an error.  The fix accepts
H/D/T, He, Li, Be, B, C, N, O, Ne, Ar, Fe, Kr, Mo, Xe and W, and the
database distributed with OpenEdge holds SCD, ACD, CCD, PLT and PRB
tables and ionization potentials for all of them.  It holds reaction
catalogs for D, Li, B, C and W.  Any other element needs a reaction
file.

*volume_source* must name an existing compute plasma/fields or fix
background.  It acts only with *mode neutral* and needs an active
recombination reaction.  *stop_on_exhaust* acts only with *mode
neutral* and a *source_species* list.

Related commands
----------------

:doc:`fix background <fix_background>`,
:doc:`compute plasma/fields <compute_plasma_fields>`,
:doc:`fix surface/emit/puff <fix_surface_emit_puff>`,
:doc:`compute volume/emissivity/grid <compute_volume_emissivity_grid>`,
:doc:`compute grid/weighted <compute_grid_weighted>`,
:doc:`surf_react surface/pwi <surf_react_surface_pwi>`

Default
-------

The option defaults are mode = kinetic, rate_cache = particle,
stop_on_exhaust = no, exhaust_threshold = 0, units = counts, output =
summary, ionization = recombination = cx = dissociation = yes, no
source_species and no volume_source.
