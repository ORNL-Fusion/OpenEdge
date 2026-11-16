.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix surface/emit/source

fix surface/emit/source command
===============================

Syntax
------

.. parsed-literal::

   fix ID surface/emit/source mix-ID group-ID source keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* surface/emit/source = style name of this fix command
* mix-ID = ID of the mixture to use when creating particles
* group-ID = ID of the surface group that emits particles
* source = compute-ID [*flux_index* M] or *constant* F or *file* path
  [*section* NAME]

  * compute-ID = ID of a compute with per-surf output that gives the
    emitted flux of each element (m\ :sup:`-2` s\ :sup:`-1`)
  * *flux_index* M = column of the per-surf array of the compute.
    Allowed only directly after compute-ID.
  * *constant* F = the same flux F (m\ :sup:`-2` s\ :sup:`-1`, > 0) on
    every element
  * *file* path = read density and stream velocity from an inflow file
  * *section* NAME = section of the file to read

* zero or more keyword/value pairs may be appended
* keyword = *conc_scale* or *n* or *nlaunch* or *nlaunch_total* or
  *flux_thresh* or *normal* or *model* or *nevery* or *perspecies* or
  *region*

  * *conc_scale* values = attr col

    * attr = custom per-surf array, for example the concentrations of
      surf_react surface/pwi
    * col = column, counted from 1, that multiplies the flux of each
      element

  * *n* value = Np = emit Np particles per window of nevery steps with the
    default weight (0 = follow the flux)
  * *nlaunch* value = N = emit N weighted particles from every piece
    of an emitting element
  * *nlaunch_total* value = N = emit N weighted particles per window of nevery
    steps in total, divided among the elements by source strength
  * *flux_thresh* value = F = skip elements whose flux is below F
    (m\ :sup:`-2` s\ :sup:`-1`)
  * *normal* value = *yes* or *no* = direct the stream velocity of the
    mixture along the element normal
  * *model* values = *thermal* or *thermal_tsurf* name or *thompson* Es
    [*cos_n* n] [*emax* E] or *fixed_energy* E

    * *thermal* = Maxwellian flux at the mixture temperature
    * *thermal_tsurf* name = Maxwellian flux at the surface temperature
      of each element, stored in the custom per-surf vector *name*
      (degrees Celsius)
    * *thompson* Es = Thompson distribution with surface binding energy
      Es (eV)
    * *cos_n* n = exponent of the angular distribution (> 0)
    * *emax* E = cutoff energy (eV), larger than Es
    * *fixed_energy* E = every particle leaves with energy E (eV)

  * *nevery* value = Nevery = evaluate the source every this many timesteps
  * *perspecies* value = *yes* or *no*
  * *region* value = region-ID = emit only at points inside the region

Examples
--------

.. code-block:: text

   compute cpmiO surface/physical/sputter wall background pd target W projectiles O erosion_flux
   fix femit surface/emit/source tungstenSource wall cpmiO perspecies no &
       nlaunch_total 200 model thompson 8.68 emax 80 nevery 100
   fix femit surface/emit/source tungstenSource wall cpmiWd conc_scale adens_conc 3 &
       perspecies no nlaunch_total 200 model thompson 8.68 emax 80
   fix fpuff surface/emit/source dgas valve constant 1.0e22 n 20 normal yes perspecies no
   fix finflow surface/emit/source dgas inlet file input/inflow.dat section ZLO &
       n 50 perspecies no

Description
-----------

Emit particles of a mixture from the surface elements of a group,
with the emitted flux of each element given by a per-surf compute, a
constant or an inflow file.  The usual source is the erosion flux of
:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`,
which turns the sputtering of the wall by the background plasma into
simulation particles.

Symbols are defined on the :ref:`notation <notation>` page.  On this
page :math:`w` is the number of real atoms that a simulation particle
represents.

Source strength
^^^^^^^^^^^^^^^

The fix works on tasks.  A task is the piece of one surface element
that lies inside one grid cell, with area :math:`A`.  On a step where
the fix acts, the number of real particles that a task with flux
:math:`\Gamma` has to emit is

.. math::
   :label: emit-source-strength

   S = c\,\Gamma\, A\,\Delta t_N ,
   \qquad \Delta t_N = N\,\Delta t ,

where :math:`N` is the *nevery* value and :math:`c` is the
*conc_scale* factor, or 1.  The area is the polygon area in three
dimensions, the area of the ring swept by the line segment in an
axisymmetric simulation, and the segment length in a planar
two-dimensional simulation.  A task is skipped if its flux, including
:math:`c`, is not positive or is below *flux_thresh*.

With a compute as the source, its per-surf vector, or column
*flux_index* of its per-surf array, is evaluated at every source
evaluation.  If the compute is a
:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`
with *static yes*, its flux is read once and kept.

*conc_scale* multiplies the flux of each element by one column of a
custom per-surf array.  With the concentration array of
:doc:`surf_react surface/pwi <surf_react_surface_pwi>`, the emitted
flux of a material is its erosion flux times its local concentration.
The concentration is read at each source evaluation, also when the
flux itself is kept from a static compute.

With *constant* every element has the flux F.

With *file* the flux of an element is
:math:`\Gamma = n\,|\mathbf{u}\cdot\hat{\mathbf{n}}|`, with density
:math:`n` and stream velocity :math:`\mathbf{u}` interpolated
bilinearly from the file at the midpoint of the element.  The file
has the format of the SPARTA inflow files:

.. code-block:: text

   NAME
   NIJ ni nj
   NV nv
   VALUES name1 name2 ...
   IMESH x1 x2 ... xni
   JMESH y1 y2 ... ynj
   i j value1 value2 ...

with ni\*nj rows of values.  NAME is the *section* name and must be
one of XLO, XHI, YLO, YHI, ZLO and ZHI.  It selects the two box
coordinates of the mesh: all but x for XLO and XHI, all but y for YLO
and YHI, all but z for ZLO and ZHI.  The value names are nrho, vx, vy,
vz, temp and gamma.  Fields that are not in the file keep the value
of the mixture.  The velocity of the file also replaces the stream
velocity of the mixture in the velocity model.  The temp and gamma
columns are read, but the thermal speed of the emitted particles is
that of the mixture temperature.

Number of particles and weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Four modes convert :math:`S` into simulation particles.  A fractional
number :math:`n` of particles is rounded to
:math:`\lfloor n + U \rfloor`.

Without *n*, *nlaunch* or *nlaunch_total*, a task emits
:math:`n = S/F_\mathrm{num}` particles, each with the default weight
:math:`F_\mathrm{num}`, as :doc:`fix emit/surf <fix_emit_surf>` does.

*n* Np emits Np particles per window of nevery steps with the default weight.
They are divided among the tasks in proportion to area.  The flux of
the source is not used in this mode.

*nlaunch* N emits N particles from every task that has :math:`S > 0`,
each with the weight

.. math::

   w = \frac{S}{N} .

Strong and weak elements get the same number of particles, and the
total number grows with the number of emitting tasks.  This suits
small emitting areas.

*nlaunch_total* N emits N particles per window of nevery steps in total.  With
:math:`S_\mathrm{tot}` the sum of :math:`S` over all tasks, task
:math:`i` is given the expected number
:math:`n_i = N S_i/S_\mathrm{tot}` and emits
:math:`k_i = \lfloor n_i + U \rfloor` particles of weight

.. math::
   :label: emit-source-weight

   w_i =
   \begin{cases}
      S_i/k_i , & n_i \ge 1 , \\
      S_\mathrm{tot}/N , & n_i < 1 .
   \end{cases}

A task that expects at least one particle emits exactly its own
source in every window.  A weaker task emits one particle with
probability :math:`n_i`, and the weight :math:`S_\mathrm{tot}/N` makes
its mean emission equal to :math:`S_i`.  The total emission is
therefore the same for any N, which only sets the cost and the noise.
This suits whole walls with a wide range of fluxes.

*nlaunch* and *nlaunch_total* store the weight in the attribute of
:doc:`fix particle/weight <fix_particle_weight>`, which must be
defined.

With *perspecies yes* the particle count of each species is the count
of the task times the mixture fraction of the species.  With
*perspecies no* the species of each particle is drawn from the mixture
fractions.

Position and time
^^^^^^^^^^^^^^^^^

Particles are created at the start of the timestep, at points
distributed uniformly over the task area.  A point outside the
*region*, if one is given, is rejected.  Each new particle is then
moved for a random fraction :math:`U\,\Delta t` of the timestep in
which it is launched.  This first move is a straight line at the launch
velocity.  Surface collisions are detected along it, but the fixes
that act once per timestep see the particle only at its end.

The *nevery* keyword sets how often the source is evaluated, not how
often particles are launched.  The source is evaluated on the first
step of each window of *nevery* timesteps and held for that window
(the *conc_scale* factor is read at the same time, in every launch
mode).  The particles of the window are released over its
timesteps: an element with :math:`N` particles per window launches
them at evenly spaced steps, starting at an offset that differs from
element to element and from window to window.  The number of particles
per window and their weights are the same as for a single launch, so
*nevery* sets the cost of the source evaluation and the number of
particles, while every new particle moves at most one timestep before
the other fixes act on it.

The source of a window, the flux and the *conc_scale* factor of each
element, is kept across successive :doc:`run <run>` commands and across
load balancing, also with distributed surfaces.  It is not stored in
restart files.  A run restarted inside a window evaluates the source
again at its first step and uses that value for the rest of the window,
with a warning; the continuation is then not the uninterrupted run.  If
the source varies in time, the amount emitted in that window differs by
the change of the source since the start of the window, which can be
large for a rapidly varying source.  Write restart files at multiples
of *nevery* to keep the source history; the emitted amount is then the
same as in an uninterrupted run.

Velocity
^^^^^^^^

The *model* keyword selects the velocity distribution.  In all models
:math:`\vartheta` is the angle between the velocity and the element
normal, and the azimuth about the normal is uniform.

*thermal* samples a Maxwellian flux at the temperature
:math:`T` of the mixture, as :doc:`fix emit/surf <fix_emit_surf>`
does.  With :math:`v_\mathrm{th} = (2k_\mathrm{B}T/m)^{1/2}` and
:math:`u_n` the stream velocity along the normal,

.. math::

   v_n = u_n + \beta\, v_\mathrm{th} , \qquad
   f(\beta) \propto \left(\beta + \frac{u_n}{v_\mathrm{th}}\right)
      e^{-\beta^2} , \qquad
   v_t = v_\mathrm{th}\sqrt{-\ln U} ,

where :math:`v_n > 0` is the normal component and :math:`v_t` the
magnitude of the tangential component.  :math:`\beta` is limited to
the interval [-3,3].  With *normal no*, :math:`u_n` is the projection
of the mixture *vstream* on the normal and the tangential part of
*vstream* is added.  With *normal yes*, :math:`u_n` is the magnitude
of *vstream* and nothing is added to the tangential components.
Without a stream velocity the mean kinetic energy of the emitted
particles is :math:`2k_\mathrm{B}T`.

The other models sample a kinetic energy :math:`E` and a direction
from a power-cosine distribution,

.. math::

   v = \sqrt{\frac{2E}{m}} , \qquad
   \cos\vartheta = U^{1/(n+1)} ,

and then add the mixture *vstream*.  The exponent is :math:`n = 1`,
the cosine distribution, unless *cos_n* is given with *thompson*.

*thermal_tsurf* samples the energy of a Maxwellian flux at the
surface temperature :math:`T_\mathrm{s}` of the element,

.. math::

   f(E) \propto E\, e^{-E/k_\mathrm{B}T_\mathrm{s}} , \qquad
   E = -k_\mathrm{B}T_\mathrm{s}\,\ln(U_1 U_2) .

The custom vector holds the temperature in degrees Celsius, as
written by :doc:`fix surface/state/lm <fix_surface_state_lm>`, and
273.15 is added to obtain :math:`T_\mathrm{s}`.  The rotational and
vibrational energies are sampled at the same temperature.  An element
whose temperature is not a positive finite number emits nothing.

*thompson* samples the Thompson distribution of sputtered atoms,

.. math::
   :label: emit-source-thompson

   f(E) \propto \frac{E}{(E + E_\mathrm{s})^3}
      \left[1 - \sqrt{\frac{E + E_\mathrm{s}}{E_\mathrm{max} + E_\mathrm{s}}}\,\right] ,
   \qquad 0 \le E \le E_\mathrm{max} ,

where :math:`E_\mathrm{max}` is the *emax* value.  Without *emax* the
bracket is dropped and the energy is unbounded,
:math:`E = E_\mathrm{s}\sqrt{U}/(1 - \sqrt{U})`.  With *emax* the
sampling is the rejection method described for
:doc:`surf_react surface/pwi <surf_react_surface_pwi>`.

.. note::

   The *thompson* argument is the full surface binding energy
   :math:`E_\mathrm{s}`, 8.68 eV for tungsten.  The distribution has
   its maximum at :math:`E_\mathrm{s}/2` by itself, so a halved value
   must not be entered.

Without *emax* the distribution falls off as :math:`E^{-2}` and a
small fraction of the atoms leaves with very high energy.  Production
runs should set *emax* to the largest energy the projectiles can
transfer, about :math:`\gamma E_0 - E_\mathrm{s}` with
:math:`\gamma` of Eq. :eq:`pwi-thompson`.

*fixed_energy* gives every particle the same energy.

In all models the rotational and vibrational energies are sampled at
the mixture temperatures, except with *thermal_tsurf*.

The keyword *source_thresh* is no longer supported and produces an
error.

Restart, output info
^^^^^^^^^^^^^^^^^^^^

No information about this fix is written to
:doc:`binary restart files <restart>`.

This fix computes a global vector of length 2, which can be accessed
by various output commands.  The first element is the number of
particles emitted on the most recent timestep.  The second is
the cumulative number of particles emitted.

Restrictions
------------

This command is part of the OPENEDGE package.

It requires explicit surface elements.  The source compute must
produce per-surf output.

*n* > 0 and *nlaunch_total* cannot be used with *perspecies yes*,
which is the default.  *n* > 0 cannot be used with *nlaunch_total*,
and *nlaunch* cannot be used with *nlaunch_total*.  *nlaunch* and
*nlaunch_total* require
:doc:`fix particle/weight <fix_particle_weight>` and cannot be used
with a *file* source.

*conc_scale* requires a compute as the source.  The *conc_scale* and
*thermal_tsurf* custom attributes must exist when the run starts.

Related commands
----------------

:doc:`fix emit/surf <fix_emit_surf>`,
:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`,
:doc:`surf_react surface/pwi <surf_react_surface_pwi>`,
:doc:`fix particle/weight <fix_particle_weight>`,
:doc:`fix surface/emit/puff <fix_surface_emit_puff>`

Default
-------

The option defaults are flux_index = 1, section = ZLO, n = 0,
flux_thresh = 0.0, normal = no, model = thermal, cos_n = 1, no emax,
nevery = 1 and perspecies = yes.  There is no default conc_scale or
region.
