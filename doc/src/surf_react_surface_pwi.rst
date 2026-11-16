.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: surf_react surface/pwi

surf_react surface/pwi command
==============================

Syntax
------

.. parsed-literal::

   surf_react ID surface/pwi infile keyword values ...

* ID = user-assigned name for the surface reaction model
* surface/pwi = style name of this surface reaction model
* infile = file listing the surface reaction channels (format below)
* zero or more keyword/value pairs may be appended
* keyword = *twall* or *twall_surf* or *R_surf* or *ehist* or
  *roughness_dm* or *adens_surf* or *adens_init* or
  *adens_init_file* or *adens_init_group* or *substrate* or
  *deposit_as* or *rzone* or *adens_erosion* or *adens_implant* or
  *conc_feedback* or *adens_nevery* or *strata* or *strata_dens* or
  *ledger_reset*

  * *twall* value = wall temperature (K)
  * *twall_surf* value = name of a custom per-surf vector that holds
    the wall temperature of each element (K)
  * *R_surf* value = name of a custom per-surf vector that holds the
    recycling coefficient of each element.  It replaces R of every *A*
    channel.
  * *ehist* values = file nbins emax every

    * file = output file for the impact energy and angle histograms
    * nbins = number of energy bins between 0 and emax.  The last bin
      also collects impacts above emax.
    * emax = upper bound of the energy axis (eV)
    * every = write and reset the histograms every this many steps, or
      0 for one cumulative histogram

  * *roughness_dm* value = mean surface inclination angle
    :math:`\delta_m` (degrees), :math:`0 \le \delta_m < 90`
  * *adens_surf* value = name of the custom per-surf array that holds
    the areal density of each species (m\ :sup:`-2`).  It is created if
    it does not exist.
  * *adens_init* values = species value

    * species = species ID whose column is initialized
    * value = initial areal density (m\ :sup:`-2`)

  * *adens_init_file* values = file column species scale

    * file = :doc:`dump surf <dump>` text file, or a text file with
      two columns, surface ID and value
    * column = name of the dump column to read
    * species = species ID whose column is initialized
    * scale = factor applied to the file values (> 0)

  * *adens_init_group* value = surface group ID to which the initial
    layers are restricted
  * *substrate* value = species ID of the bulk material under all
    layers
  * *deposit_as* values = element species

    * element = element symbol whose retained atoms are re-labelled
    * species = species ID that holds the redeposited material

  * *rzone* value = areal density of the reaction zone
    :math:`\sigma_\mathrm{z}` (m\ :sup:`-2`)
  * *adens_erosion* values = compute-ID col species [*noconc*]

    * compute-ID = per-surf compute that provides a gross erosion flux
      (m\ :sup:`-2` s\ :sup:`-1`)
    * col = 0 for a per-surf vector, or the column number of a per-surf
      array
    * species = species ID that the flux removes
    * *noconc* = do not multiply the flux by the surface concentration

  * *adens_implant* values = species source [*rcoef* R0] [*cmax* c]
    [*alpha* a] [*depth* d]

    * species = species ID that receives the implanted atoms
    * source = *compute* compute-ID col or *const* flux, the incident
      flux (m\ :sup:`-2` s\ :sup:`-1`)
    * *rcoef* R0 = reflected fraction of the incident flux
    * *cmax* c = saturation concentration (required)
    * *alpha* a = steepness of the saturation switch
    * *depth* d = implantation depth (m), used with *strata*

  * *conc_feedback* value = *yes* or *no*
  * *adens_nevery* value = N = fold the accumulated areal-density
    changes into the per-surf array every N steps
  * *strata* values = K [*minthick* t]

    * K = maximum number of strata per element (2 to 64)
    * *minthick* t = minimum stratum thickness (Angstrom)

  * *strata_dens* values = species density

    * species = species ID.  Any charge state selects the material.
    * density = number density of the solid (m\ :sup:`-3`)

  * *ledger_reset* = no value.  Zero the running tallies of a restart
    file that stored them as totals (see below).

The spellings *sigma_surf*, *sigma_init*, *sigma_erosion*,
*sigma_zone*, *sigma_feedback* and *sigma_nevery* are accepted for
*adens_surf*, *adens_init*, *adens_erosion*, *rzone*, *conc_feedback*
and *adens_nevery*.

Examples
--------

.. code-block:: text

   surf_react wallPWI surface/pwi input/wall.recycle
   surf_react wallPWI surface/pwi input/wall.recycle twall 400.0
   surf_react wallPWI surface/pwi input/wall.recycle R_surf R_rec twall_surf Twall_pfc
   surf_react wallPWI surface/pwi input/wall.pwi twall 500.0 &
              adens_surf adens rzone 1.0e19 adens_init B 5.0e19 &
              adens_erosion cpmiW 0 W adens_erosion cpmiB 0 B
   surf_react wallPWI surface/pwi input/wall.pwi twall 500.0 &
              adens_surf adens deposit_as W Wd &
              adens_init_file state/adens.dump s_adens_net Wd 1.0 &
              adens_erosion cpmiW 0 W adens_erosion cpmiWd 0 Wd

Description
-----------

Define the plasma-wall interaction model of surface elements.  A
particle that hits an element can be reflected as a fast atom,
absorbed and re-emitted thermally, or retained in the wall, and it can
sputter wall atoms.  The channels open to each incident species are
listed in *infile*.  The reflection and re-emission options follow
those of EIRENE: tabulated reflection from binary-collision (TRIM)
calculations for the prompt fraction, and a recycling coefficient with
a molecular fraction for the rest.  The same file grammar serves any
projectile and wall material.

With *adens_surf* the model also keeps the composition of every
element as areal densities :math:`\sigma_s` of the species in a thin
reaction zone.  The composition changes by deposition and erosion and
feeds back on the reflection coefficients and sputter yields, as in
WallDYN.

Symbols are defined on the :ref:`notation <notation>` page.  On this
page :math:`w` is the number of real atoms that a simulation particle
represents.  It is the weight attribute of
:doc:`fix particle/weight <fix_particle_weight>` if that fix is
defined, and :math:`F_\mathrm{num}` otherwise.

Reaction file
^^^^^^^^^^^^^

Lines that start with # and blank lines are ignored.  Each channel
takes two lines:

.. code-block:: text

   reactant --> product1 [+ product2]
   TYPE coefficients

The reactant and the products are species IDs.  A single product NULL
means no product.  TYPE is one letter, in upper or lower case.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Second line
     - Products
     - Meaning
   * - T table [*mat* sp] [*conc* sp] [*rtable* name] [*scale* f]
     - 1
     - Reflection with the probability, energy and angles of the
       tabulated pair *table*
   * - A R [f_mol]
     - 1 or 2
     - Absorption and thermal re-emission with recycling coefficient
       R and molecular fraction f_mol, both between 0 and 1
   * - S entry [*mat* sp] [*conc* sp] [*yscale* f]
     - 1
     - Sputtering with the yield of the pair *entry*, for example
       W_on_W.  The product is the sputtered species.
   * - E prob energy
     - 1
     - Exchange: re-emit the product with a fixed kinetic energy (eV)
   * - D prob e1 e2
     - 2
     - Dissociation: emit two products with fixed kinetic energies
       (eV)
   * - R prob
     - NULL
     - Recombination: remove the particle

A channel whose reactant or product is not a species of the simulation
is ignored.

Impact energy and angle
^^^^^^^^^^^^^^^^^^^^^^^

For an incident particle of mass :math:`m` and velocity
:math:`\mathbf{v}` at an element with unit normal
:math:`\hat{\mathbf{n}}` pointing into the plasma,

.. math::
   :label: pwi-impact

   E_0 = \tfrac{1}{2}\, m v^2 ,
   \qquad
   \cos\theta = -\frac{\mathbf{v}\cdot\hat{\mathbf{n}}}{v} .

The impact angle :math:`\theta` is measured from the surface normal
and is handled in degrees throughout: :math:`\theta = 0` is normal
incidence and :math:`\theta = 90` is grazing incidence.
:math:`\cos\theta` is limited to the interval [0,1].  The velocity is
the one at the wall, so it includes any acceleration that the sheath
model of the :doc:`global <global>` command applied before the
collision.

Outcome of an impact
^^^^^^^^^^^^^^^^^^^^

The impact is processed in this order.

1. Every *S* channel of the incident species emits sputtered atoms.
   Sputtering is additive: it does not change what happens to the
   incident particle.
2. The impact is added to the *ehist* histograms.
3. One uniform random number :math:`U` selects among the *T*, *A*,
   *E*, *D* and *R* channels of the species.  The channels are taken in
   file order and their probabilities :math:`p_k` are accumulated.
   The first channel :math:`k` with :math:`\sum_{j\le k} p_j > U`
   fires.

For *E*, *D* and *R* the probability is the value *prob* of the file.
For *T* it is evaluated for each impact,

.. math::
   :label: pwi-prefl

   p_\mathrm{T} = f\, c_\mathrm{mat}\, R_N(E_0,\theta) ,

where :math:`f` is the *scale* value (default 1), and
:math:`c_\mathrm{mat}` is the surface concentration of the *mat*
species, or 1 without *mat*.  An *A* channel has probability 1.
Listed after *T* it takes every particle that was not reflected, so
that a fraction :math:`R_N` is reflected promptly and the fraction
:math:`1 - R_N` is absorbed.  With *scale* :math:`f` equal to a
recycling coefficient, an *R* channel of probability :math:`1 - f`
listed first and an *A* channel with R = 1 listed last, the prompt
reflection is counted inside the recycling coefficient.

If no channel fires, the particle is passed to the
:doc:`surf_collide <surf_collide>` model of the element.

Reflection
^^^^^^^^^^

A *T* table holds the particle reflection coefficient
:math:`R_N(E_0,\theta)` on a grid of 12 energies and 7 angles,
together with quantiles of the distributions of the reflected energy
and of the polar and azimuthal exit angles.  :math:`R_N` is
interpolated linearly in :math:`E_0` and :math:`\theta` and held
constant outside the grid.  The reflected energy and direction are
sampled from the quantiles with three uniform random numbers, so the
shape of the tabulated distributions is kept.  The polar angle is
relative to the surface normal.  The azimuth is relative to the plane
of incidence, and its sign is chosen at random.

Tables are read from the group /surface/reflection/<pair> of the
file processes.h5 in the OpenEdge database directory.  The pair name
is the *table* name in lower case, for example d_on_w.  A missing
table is an error.

With *conc* sp and *rtable* name, the probability
:math:`R_N(E_0,\theta_\mathrm{eff},c)` is read from the
three-dimensional table *name* (group /surface/sputter of
processes.h5) at the local concentration :math:`c` of species sp, and
:math:`c_\mathrm{mat}` is not applied.  The angle is shifted as in Eq.
:eq:`pwi-rough`.  The energy and angles of the reflected particle are
still sampled from the pure-target table *table*.

Absorption and re-emission
^^^^^^^^^^^^^^^^^^^^^^^^^^

An *A* channel returns the absorbed particle as a thermal atom or
molecule, or retains it.  The first product is the molecular species.
The atomic species is the second product if there is one, the
reactant if the reactant is neutral, and the first product if the
reactant is an ion.  If the two differ, the outcome is

.. math::

   P_\mathrm{atom} = R\,(1 - f_\mathrm{mol}) , \qquad
   P_\mathrm{mol} = \tfrac{1}{2}\, R\, f_\mathrm{mol} , \qquad
   P_\mathrm{ret} = 1 - P_\mathrm{atom} - P_\mathrm{mol} .

The factor 1/2 accounts for two atoms leaving as one molecule.  If the
atomic and molecular species are the same, the particle returns with
probability :math:`R` and is retained with probability
:math:`1 - R`.  An ion reactant with a molecular fraction between 0
and 1 needs both products, for example

.. code-block:: text

   D+ --> D2 + D
   A 0.99 0.5

When *R_surf* is set, the value of the element, limited to [0,1],
replaces R of every *A* channel of this model.

A re-emitted particle of mass :math:`m` leaves with the velocity
distribution of a Maxwellian flux at the wall temperature
:math:`T_\mathrm{wall}` (K),

.. math::

   v_n = v_\mathrm{th}\sqrt{-\ln U} , \qquad
   v_{t1,t2} = \sqrt{\frac{k_\mathrm{B}T_\mathrm{wall}}{m}}\;\xi_{1,2} ,
   \qquad
   v_\mathrm{th} = \sqrt{\frac{2k_\mathrm{B}T_\mathrm{wall}}{m}} ,

where :math:`v_n` is along the normal and :math:`v_{t1}`,
:math:`v_{t2}` are the two tangential components.  The mean kinetic
energy of the emitted particles is :math:`2k_\mathrm{B}T_\mathrm{wall}`.
A retained particle is deleted and, with *adens_surf*, deposited on
the element.

Fixed-energy channels
^^^^^^^^^^^^^^^^^^^^^

The products of *E* and *D* channels leave with the kinetic energies
given in the file and a cosine distribution of directions about the
normal, :math:`\cos\vartheta = \sqrt{U}` with uniform azimuth.  An *R*
channel deletes the particle and, with *adens_surf*, deposits it on
the element.

Sputtering
^^^^^^^^^^

An *S* channel emits

.. math::
   :label: pwi-nsput

   N = \lfloor Y_\mathrm{eff} \rfloor +
   \begin{cases} 1 & \text{with probability }
        Y_\mathrm{eff} - \lfloor Y_\mathrm{eff} \rfloor \\
        0 & \text{otherwise} \end{cases}

simulation particles of the product species at the impact point.
Each carries the weight :math:`w` of the incident particle, so the
sputtered flux of real atoms is the yield times the incident flux.
The effective yield is

.. math::
   :label: pwi-yeff

   Y_\mathrm{eff} = f_Y\, c_\mathrm{mat}\, Y(E_0,\theta_\mathrm{eff}) ,

with :math:`f_Y` the *yscale* value (default 1) and
:math:`c_\mathrm{mat}` the surface concentration of the *mat* species,
or 1 without *mat*.  The *roughness_dm* keyword evaluates the yield at
a reduced angle,

.. math::
   :label: pwi-rough

   \theta_\mathrm{eff} = \max(0,\; \theta - \delta_m) ,

which flattens the angular dependence as a rough surface does.

No atoms are emitted if the incident particle cannot transfer more
than the binding energy, :math:`\gamma E_0 \le E_\mathrm{s}`, with
:math:`\gamma` defined in Eq. :eq:`pwi-thompson`.  :math:`N` is
limited to 20 per impact, and a warning is printed the first time the
limit is reached.

Sputter yield
^^^^^^^^^^^^^

The yield of an *S* channel is taken from a table if processes.h5 has
the group /surface/sputter/<entry>, with the entry name in lower
case.  Otherwise the analytic fit below is used and a warning is
printed.

A table gives :math:`Y` on a grid of energies and angles.  It is
interpolated linearly in :math:`\ln E_0` and in :math:`\theta` and
held constant outside the grid.  Two kinds of table have a third
axis.

* A table with a temperature axis, used for liquid metals, is
  interpolated linearly at the wall temperature of the element
  (*twall_surf* or *twall*).  It is evaluated at its lowest
  temperature if no wall temperature is set.
* A compound table has a composition axis.  It is interpolated at the
  concentration of the *conc* species of the channel, and
  :math:`c_\mathrm{mat}` is not applied.  Without *conc* it is
  evaluated at concentration 1.

The analytic fit is the Bohdansky formula for normal incidence with
the Yamamura angular factor, with the parameters :math:`Q`,
:math:`E_\mathrm{th}`, :math:`E_\mathrm{TF}` and :math:`E_\mathrm{s}`
of [Eckstein1993]_ for the pair *entry*:

.. math::
   :label: pwi-bohdansky

   Y(E_0,0) = Q\, s_n\!\left(\frac{E_0}{E_\mathrm{TF}}\right)
      \left[1 - \left(\frac{E_\mathrm{th}}{E_0}\right)^{2/3}\right]
      \left(1 - \frac{E_\mathrm{th}}{E_0}\right)^{2} ,
   \qquad E_0 > E_\mathrm{th} ,

.. math::

   s_n(\varepsilon) = \frac{0.5\,\ln(1 + 1.2288\,\varepsilon)}
      {\varepsilon + 0.1728\sqrt{\varepsilon} + 0.008\,\varepsilon^{0.1504}} ,

.. math::
   :label: pwi-yamamura

   Y(E_0,\theta) = Y(E_0,0)\, (\cos\theta)^{-f}
      \exp\!\left[f\left(1 - \frac{1}{\cos\theta}\right)
      \cos\theta_\mathrm{opt}\right] ,
   \qquad f = 2,\quad \cos\theta_\mathrm{opt} = 0.26 .

The yield is zero for :math:`E_0 \le E_\mathrm{th}`.  The values of
:math:`f` and :math:`\cos\theta_\mathrm{opt}` are the same for every
pair, and :math:`\theta` is limited to 89.5 degrees.  Eq.
:eq:`pwi-yamamura` falls steeply towards grazing incidence, where it
underestimates the yield.  Tables should be used for walls that the
magnetic field meets at a shallow angle.

The entry must exist either in the built-in parameter list of the
analytic fit or as a table that carries its surface binding energy.
Otherwise the command stops with an error.

Energy and angle of sputtered atoms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The energy :math:`E` of a sputtered atom is sampled from the Thompson
distribution with a cutoff at the largest energy that the projectile
can transfer,

.. math::
   :label: pwi-thompson

   f(E) \propto \frac{E}{(E + E_\mathrm{s})^3}
      \left[1 - \sqrt{\frac{E + E_\mathrm{s}}{E_\mathrm{max} + E_\mathrm{s}}}\,\right] ,
   \quad 0 \le E \le E_\mathrm{max} ,
   \qquad
   E_\mathrm{max} = \gamma E_0 - E_\mathrm{s} ,
   \quad
   \gamma = \frac{4\, m_1 m_2}{(m_1 + m_2)^2} ,

where :math:`m_1` and :math:`m_2` are the masses of the incident and
the sputtered species.  :math:`E_\mathrm{s}` is the full surface
binding energy of the target, 8.68 eV for tungsten.  The distribution
without the cutoff factor has its maximum at :math:`E_\mathrm{s}/2`.
:math:`E_\mathrm{s}` is taken from the built-in parameter list if the
entry is there, and otherwise from the binding-energy attribute of
the table.

The sampling is by rejection.  A trial energy is drawn from the
distribution without the bracket, truncated at :math:`E_\mathrm{max}`,
by inversion of its cumulative distribution,

.. math::

   E = E_\mathrm{s}\,\frac{s}{1 - s} , \qquad
   s = \sqrt{U}\;\frac{E_\mathrm{max}}{E_\mathrm{max} + E_\mathrm{s}} ,

and it is accepted with the probability given by the bracket of Eq.
:eq:`pwi-thompson`.  The direction follows a cosine distribution about
the surface normal, :math:`\cos\vartheta = \sqrt{U}` with uniform
azimuth.

Wall temperature
^^^^^^^^^^^^^^^^

*twall* sets one wall temperature for all elements.  *twall_surf*
reads the temperature of each element from a custom per-surf vector,
which can be set by the :doc:`custom <custom>` command, read by
:doc:`read_surf <read_surf>` or updated by a fix.  The two keywords
cannot be used together.

The wall temperature is used for the velocity of re-emitted particles
(*A*), for yield tables with a temperature axis, and for the
rotational and vibrational energy of the products of *T*, *A*, *E*
and *D* channels, which is sampled at that temperature.  If neither
keyword is given, the *A* channel uses 300 K.  The internal energy is
then kept by an *E* channel that does not change the species, and set
to zero by *T* and *D* channels and by an *E* channel that changes the
species.

Surface composition
^^^^^^^^^^^^^^^^^^^

With *adens_surf* the areal density :math:`\sigma_s` of every species
:math:`s` is stored for every element in a custom per-surf array with
one column per species, in the order of the
:doc:`species <species>` command.  For an element of area :math:`A` it
evolves as

.. math::
   :label: pwi-ledger

   \frac{d\sigma_s}{dt} =
      \frac{1}{A}\sum_{\mathrm{retained}} w\,\delta(t - t_i)
    - \frac{1}{A}\sum_{\mathrm{sputtered}} N w\,\delta(t - t_i)
    - c_s\,\Gamma_{\mathrm{ero},s}
    + \Gamma_{\mathrm{imp},s} .

The first sum runs over the particles that the wall retains at times
:math:`t_i`.  An *R* channel credits the incident species and the
retained branch of an *A* channel credits its atomic species.  The
second sum runs over the
impacts that sputter :math:`N` atoms of species :math:`s`, Eq.
:eq:`pwi-nsput`.  The last two terms are the erosion and implantation
by the background plasma, described below.  :math:`A` is the triangle
area in three dimensions, the area of the ring swept by the line
segment in an axisymmetric simulation, and the segment length in a
planar two-dimensional simulation.

All charge states of one element form one material.  The species
names are split at the first character that is not a letter, so W,
W+ and W2+ belong to the material W.  The concentration of material
:math:`m` in the reaction zone is

.. math::
   :label: pwi-conc

   c_m = \frac{1}{\sigma_\mathrm{z}}\,
         \max\Bigl(0,\; \sum_{s\in m}\sigma_s\Bigr) ,

with :math:`\sigma_\mathrm{z}` the *rzone* value.  If the
concentrations add up to more than 1 they are divided by their sum.
Otherwise the remainder :math:`1 - \sum_m c_m` is added to the
substrate material.  These concentrations are the factors
:math:`c_\mathrm{mat}` of Eqs. :eq:`pwi-prefl` and :eq:`pwi-yeff`,
the coordinate of compound tables, and the factor :math:`c_s` of Eq.
:eq:`pwi-ledger`.  A surface covered by a deposit therefore reflects
and sputters as the deposit, and the substrate below it is shielded.
*conc_feedback no* switches this off: :math:`c_\mathrm{mat} = 1` and
compound tables are evaluated at concentration 1.

Four more per-surf custom attributes are created with the array.
Their names start with the *adens_surf* name, written <attr> here.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Attribute
     - Content
   * - <attr>_net
     - net change of the areal density summed over species since the
       start of the first run (m\ :sup:`-2`)
   * - <attr>_dep
     - gross deposition summed over species (m\ :sup:`-2`)
   * - <attr>_ero
     - gross erosion summed over species (m\ :sup:`-2`)
   * - <attr>_conc
     - array with one column per species.  Every column of a material
       holds its concentration :math:`c_m`.

The first three are vectors that start at zero and do not contain the
initial layers.  The array <attr> itself holds the total inventory,
including the initial layers.  All of them are written to restart
files and continue to accumulate after a restart.  They can be
written by :doc:`dump surf <dump>`.

A redeposited ion is credited to the column of its own charge state,
whereas sputtering debits the column of the sputtered species and
*adens_erosion* debits the column of the species it names, usually
the neutral.  A single column can therefore become negative while the
material is present.  The inventory of a material is the sum of the
columns of all its charge states.  A column of <attr>_conc already
holds the concentration of the whole material, so one column should
be read and the columns should not be added.

The impacts are accumulated on each processor and added to the
per-surf arrays every *adens_nevery* steps, on the last step of a run
and on steps that write a dump or a restart file.  The concentrations
are recomputed at the same time, so between two updates all impacts
see the composition of the previous update.  The update requires a
reduction over all processors and all elements.  In parallel runs a
value such as *adens_nevery* 100 keeps its cost small.

Initial layers and substrate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*adens_init* adds a uniform layer of a species to every element, for
example a boronization layer.  *adens_init_file* adds a layer that
differs from element to element.  Its file is either a
:doc:`dump surf <dump>` text file, from which the named column of the
last snapshot is read by surface ID, or a text file with a surface ID
and a value on each line.  The values are multiplied by *scale*, and
values that are not positive are ignored.  Both keywords may be
repeated.  They act once, when the array is created, and not when it
is read from a restart file.  *adens_init_group* restricts them to
one surface group.

*substrate* names the bulk material under the reaction zone and under
the strata.  Without it the substrate is W if a species W is defined
and otherwise the first species.  The choice is written to the screen
and the log file.  If the deepest initial layer of an element is the
material of the substrate itself, erosion of that layer never exposes
a different material.  A warning gives the number of such elements
when the layers are created.  A *deposit_as* species such as Wd is a
material of its own and differs from W.  For a boron layer on a
graphite tile, define a species C and use *substrate* C.

Redeposited material
^^^^^^^^^^^^^^^^^^^^

*deposit_as* keeps redeposited atoms of an element apart from the
bulk of the same element.  Retained atoms of every species whose name
starts with *element* are credited to the column of *species*.
Sputtered atoms of the element are debited from the element and its
deposit species in proportion to their concentrations, so a deposit
that fills the reaction zone is consumed before the bulk below it.
*species* must be
defined with the mass of the element, its name must not start with
the element symbol (Wd for W, for example), and no particle of it is
ever created.

The deposit can be given its own yield.  For incident simulation
particles use a channel "S entry *mat* species [*yscale* f]".  For
the background plasma use
:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`
with *target* species *target_like* element, bound by
*adens_erosion*.

Erosion and implantation by the background
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*adens_erosion* removes a species at the gross erosion flux
:math:`\Gamma_{\mathrm{ero},s}` that a per-surf compute provides for
each element, usually the *erosion_flux* of
:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`.
At each update the column is reduced by

.. math::

   \Delta\sigma_s = c_s\,\Gamma_{\mathrm{ero},s}\,\Delta t\, n ,

where :math:`n` is the number of steps since the previous update.
*noconc* sets :math:`c_s = 1`.  It is meant for a compute that
already evaluates composition-resolved tables.  The keyword may be
repeated for several materials.

*adens_implant* adds atoms of a species from an incident background
flux :math:`\Gamma`, a per-surf compute column or a constant,

.. math::

   \Gamma_{\mathrm{imp},s} = \Gamma\,(1 - R_0)\,(1 - S) ,
   \qquad
   S = \tfrac{1}{2}\left\{\tanh\left[a\,(c_s - c_\mathrm{max})\right] + 1\right\} ,

where :math:`R_0`, :math:`c_\mathrm{max}` and :math:`a` are the
*rcoef*, *cmax* and *alpha* values and :math:`c_s` is the current
concentration of the species.  Implantation stops as the
concentration approaches :math:`c_\mathrm{max}`.  The species must be
defined even if no particle of it is created.

Strata
^^^^^^

Without *strata* the reaction zone is homogeneous: Eq. :eq:`pwi-conc`
does not know in which order materials arrived.  *strata* replaces it
by a stack of up to K strata per element.  Each stratum has a
thickness, a number density and a composition by material.  The stack
starts as 1 mm of substrate with the initial layers on top.

Net deposition of a material at an update is added to the top
stratum if that stratum is thinner than *minthick* or already
consists mainly of the same material, and otherwise forms a new
stratum on top.  Net erosion removes the material from the top of the
stack downwards, but only within the depth window

.. math::

   d_\mathrm{z} = \frac{\sigma_\mathrm{z}}{n_\mathrm{top}} ,

where :math:`n_\mathrm{top}` is the number density of the top
stratum.  A material buried deeper than :math:`d_\mathrm{z}` is not
eroded.  If the window holds less than the requested amount, only the
available amount is removed, and the column of the per-surf array is
corrected so that array and stack agree.  The concentrations
:math:`c_m` are the atom-weighted composition of the same window.
*adens_implant* mixes the implanted atoms into the stratum at
*depth*, or deposits them on top if *depth* is 0.

The stack is compacted at every update.  A buried stratum thinner
than *minthick* is merged into the stratum below it, with the
composition averaged by atom content.  The top stratum and the
deepest one are exempt from this.  If more than K strata remain, the
two deepest are merged until K are left.  The stack is stored in the per-surf custom array
<attr>_strata with :math:`4 + K(2 + N_\mathrm{mat})` columns, where
:math:`N_\mathrm{mat}` is the number of materials, and is written to
restart files.  On a restart a warning reports stacks whose deepest
stratum is not the substrate.

The number densities used to convert areal density to thickness are
6.306e28 (W), 1.32e29 (B), 4.29e28 (O), 1.13e29 (C), 4.63e28 (Li) and
1.23e29 (Be) m\ :sup:`-3`.  Other elements use 6.3e28
m\ :sup:`-3` with a warning.  *strata_dens* overrides the value of a
material.

Restart files with running totals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

<attr>_net, <attr>_dep and <attr>_ero are changes since the start of
the first run.  A marker attribute <attr>_ledgerv is stored with
them.  A restart file that has <attr>_net but no marker holds total
areal densities in that attribute and is refused.  *ledger_reset*
accepts such a file: the three vectors are set to zero on every
element and the marker is created, while the array <attr> and the
strata are kept.

Impact histograms
^^^^^^^^^^^^^^^^^

*ehist* accumulates histograms of the impact energy :math:`E_0` and the impact angle
:math:`\theta` of Eq. :eq:`pwi-impact` for the impacts of all species that have at
least one channel in *infile*.  Each impact adds its weight
:math:`w`.  The energy axis has *nbins* bins of equal width up to
*emax*.  The angle axis has 18 bins of 5 degrees from the normal (0)
to grazing incidence (90).

With *every* > 0 a block is appended to the file and the histograms
are reset every *every* steps.  With *every* = 0 the histograms are
never reset, and a block is written on steps that write a dump and at
the end of the run.  A block begins with a header line

.. code-block:: text

   # step S nbin N emax E nang 18 dang 5 cumulative C columns all sput transit

and has these lines:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Line
     - Content
   * - E centre w_all w_sput w_transit
     - energy histogram: bin centre (eV) and summed weights
   * - A centre w_all w_sput w_transit
     - angle histogram: bin centre (degrees) and summed weights
   * - J Ecentre Acentre w_all w_transit
     - joint histogram in energy and angle.  Only cells with
       non-zero weight are written.
   * - S species centre w
     - energy histogram of one incident species.  Species with no
       impact since the last reset are omitted.

w_all counts every impact, w_sput the impacts that sputtered at least
one atom, and w_transit the impacts of particles marked by
:doc:`fix particle/tag <fix_particle_tag>`.

Output info
^^^^^^^^^^^

This model computes a global vector of length :math:`2 + 2n`, as
described on the :doc:`surf_react <surf_react>` page.  Here :math:`n`
is the number of channels in *infile*, including channels that name
an undefined species.  The first element is the number of reactions
on the last step and the second is the cumulative number since the
start of the run.  Elements 3 to :math:`2 + n` are the numbers of
reactions of the individual channels on the last step, in the order
of the file.  The last :math:`n` elements are the cumulative numbers
of the individual channels.  Each channel that fires and each
sputtered atom counts as one reaction.  The statistics printed
at the end of a run list the total and the count of every channel
that fired at least once.  The label of a
channel is its first line in *infile* followed by a tag such as
[T:reflect], [A:absorb], [S:sputter], [E:exchange], [D:dissoc] or
[R:recomb].  An *A* channel counts every particle that reaches it,
whichever of its three outcomes occurs.

Restrictions
------------

This command is part of the OPENEDGE package.

The model cannot be assigned to a boundary of the simulation box.

*adens_surf* requires explicit surface elements that are not
distributed.  The *mat* and *conc* keywords of the reaction file and
the *deposit_as* keyword require *adens_surf*.

The *twall_surf* and *R_surf* attributes must be custom per-surf
floating-point vectors.  *adens_erosion* and *adens_implant* with
*compute* require per-surf computes that have the requested column.

Related commands
----------------

:doc:`surf_react <surf_react>`, :doc:`surf_collide <surf_collide>`,
:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`,
:doc:`fix surface/emit/source <fix_surface_emit_source>`,
:doc:`fix surface/emit/recycle <fix_surface_emit_recycle>`,
:doc:`fix particle/weight <fix_particle_weight>`,
:doc:`custom <custom>`, :doc:`read_surf <read_surf>`

Default
-------

The option defaults are rzone = 1.0e19, conc_feedback = yes,
adens_nevery = 1, roughness_dm = 0.0, no strata, minthick = 0.5 and
substrate = W if that species is defined, otherwise the first
species.  For adens_implant the defaults are rcoef = 0.0, alpha =
100.0 and depth = 0.0, and cmax has no default.  No wall temperature,
R_surf, ehist or adens_surf is set by default.
