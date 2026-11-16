.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix surface/emit/puff

fix surface/emit/puff command
=============================

Syntax
------

.. parsed-literal::

   fix ID surface/emit/puff mix-ID group-ID keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* surface/emit/puff = style name of this fix command
* mix-ID = ID of the mixture to use when creating particles
* group-ID = ID of the surface group that emits particles
* zero or more keyword/value pairs may be appended
* keyword = *n* or *normal* or *stop_at_np* or *slave* or
  *nlaunch_total* or *nevery* or *perspecies* or *region*

  * *n* value = Np = emit Np particles per emission step (0 = flow
    mode)
  * *normal* value = *yes* or *no* = direct the stream velocity of the
    mixture along the element normal
  * *stop_at_np* value = Ntot = stop once Ntot particles have been
    emitted in total (0 = no limit)
  * *slave* values = R Gamma sc-IDs

    * R = fraction of the atoms absorbed by the listed models that is
      emitted again (>= 0)
    * Gamma = external feed rate (atoms/s, >= 0)
    * sc-IDs = comma-separated IDs of
      :doc:`surf_collide vanish <surf_collide>` models

  * *nlaunch_total* value = N = emit N weighted particles per emission
    step (*slave* mode only)
  * *nevery* value = Nevery = emit every this many timesteps
  * *perspecies* value = *yes* or *no*
  * *region* value = region-ID = emit only at points inside the region

Examples
--------

.. code-block:: text

   fix puff surface/emit/puff dgas valve n 50 normal yes stop_at_np 100000 &
       perspecies no
   fix puff surface/emit/puff dgas valve slave 0.9 1.0e20 pump1,pump2 &
       nlaunch_total 200 nevery 10 perspecies no

Description
-----------

Emit particles of a mixture from the surface elements of a group at a
prescribed rate, for example a gas puff from a valve.  No plasma
background and no surface compute are needed.  The rate is set in one
of three ways: by the density and temperature of the mixture, by a
fixed number of particles per step, or by the rate at which particles
are removed at pumping surfaces.

Symbols are defined on the :ref:`notation <notation>` page.  On this
page :math:`w` is the number of real atoms that a simulation particle
represents.

The fix works on tasks.  A task is the piece of one surface element
that lies inside one grid cell, with area :math:`A`: the polygon area
in three dimensions, the area of the ring swept by the line segment
in an axisymmetric simulation, and the segment length in a planar
two-dimensional simulation.  The fix acts at the start of every
:math:`N`-th timestep, with :math:`N` the *nevery* value and
:math:`\Delta t_N = N\,\Delta t`.  A fractional number :math:`n` of
particles is rounded to :math:`\lfloor n + U \rfloor`.

Flow mode
^^^^^^^^^

Without *n* and *slave*, a task emits the particles that an
equilibrium gas with the density :math:`n_\mathrm{mix}`, temperature
:math:`T` and stream velocity of the mixture would send through its
area, as :doc:`fix emit/surf <fix_emit_surf>` does.  For species
:math:`k` with mixture fraction :math:`f_k`, the number is

.. math::
   :label: puff-flow

   n_k = \frac{\Gamma_k\, A\,\Delta t_N}{F_\mathrm{num}} ,
   \qquad
   \Gamma_k = f_k\, n_\mathrm{mix}\,\frac{v_{\mathrm{th},k}}{2\sqrt{\pi}}
      \left[e^{-s_k^2} + \sqrt{\pi}\, s_k\,(1 + \mathrm{erf}\, s_k)\right] ,

with :math:`v_{\mathrm{th},k} = (2k_\mathrm{B}T/m_k)^{1/2}` and
:math:`s_k = u_n/v_{\mathrm{th},k}`, where :math:`u_n` is the stream
velocity along the normal.  Without a stream velocity this is the
thermal flux :math:`f_k n_\mathrm{mix}\bar{v}_k/4`, with
:math:`\bar{v}_k` the mean thermal speed.  With *perspecies yes* each
species emits :math:`n_k` particles.  With *perspecies no* the task
emits :math:`\sum_k n_k` particles whose species are drawn from the
mixture fractions.

Fixed number
^^^^^^^^^^^^

*n* Np emits Np particles per emission step, summed over all
processors.  They are divided among the tasks in proportion to area.

Slave mode
^^^^^^^^^^

*slave* ties the puff to the particles removed at pumping surfaces.
The number of atoms to inject on an emission step is

.. math::
   :label: puff-slave

   N_\mathrm{atoms} = R\, W_\mathrm{abs} + \Gamma\,\Delta t_N ,

where :math:`W_\mathrm{abs}` is the number of real atoms that the
listed :doc:`surf_collide vanish <surf_collide>` models have absorbed
since the previous emission step, and :math:`\Gamma` here is the
external feed rate in atoms per second.  A pumped flux is thus fed
back in part at the puff location.  :math:`R = 0` gives a purely
external source.  The absorbed atoms are those of the moves before
the emission step, so the feedback lags by one step.

Without *nlaunch_total*, :math:`N_\mathrm{atoms}/F_\mathrm{num}`
particles of the default weight are emitted.  With *nlaunch_total* N,
N particles are emitted, each with the weight

.. math::

   w = \frac{N_\mathrm{atoms}}{N} ,

stored in the attribute of
:doc:`fix particle/weight <fix_particle_weight>`.  The number of
simulation particles is then independent of :math:`F_\mathrm{num}`
and of the pumped flux.  Nothing is emitted on a step with
:math:`N_\mathrm{atoms} = 0`.  In both cases the particles are divided
among the tasks in proportion to area.

In the fixed-number and slave modes the species of each particle is
drawn from the mixture fractions.

Position and velocity
^^^^^^^^^^^^^^^^^^^^^

Particles are created at points distributed uniformly over the task
area.  A point outside the *region*, if one is given, is rejected.
Each new particle is then moved for a random fraction
:math:`U\,\Delta t_N` of the emission interval.

The velocity is sampled from a Maxwellian flux at the mixture
temperature, as in the *thermal* model of
:doc:`fix surface/emit/source <fix_surface_emit_source>`:

.. math::

   v_n = u_n + \beta\, v_\mathrm{th} , \qquad
   f(\beta) \propto \left(\beta + \frac{u_n}{v_\mathrm{th}}\right)
      e^{-\beta^2} , \qquad
   v_t = v_\mathrm{th}\sqrt{-\ln U} ,

where :math:`v_n > 0` is the component along the normal and
:math:`v_t` the magnitude of the tangential component, whose
direction is uniform.  With *normal no*, :math:`u_n` is the
projection of the mixture *vstream* on the normal and the tangential
part of *vstream* is added.  With *normal yes*, :math:`u_n` is the
magnitude of *vstream* and nothing is added to the tangential
components.  The rotational and vibrational energies are sampled at
the mixture temperatures.

Stopping the puff
^^^^^^^^^^^^^^^^^

*stop_at_np* ends the emission when the cumulative number of emitted
particles reaches the limit.  The last emission step is cut short so
that the limit is met, which gives a single puff that can be followed
until all its particles are gone.  The count is that of the particles
emitted by this fix, so products of later reactions do not affect it.
In a parallel run the final count can exceed the limit slightly,
because each processor limits its emission against the count at the
start of the step.

Restart, output info
^^^^^^^^^^^^^^^^^^^^

No information about this fix is written to
:doc:`binary restart files <restart>`.

This fix computes a global vector of length 2, which can be accessed
by various output commands.  The first element is the number of
particles emitted on the most recent emission step.  The second is
the cumulative number of particles emitted.

Restrictions
------------

This command is part of the OPENEDGE package.

It requires explicit surface elements.

*n* > 0 and *slave* cannot be used with *perspecies yes*, which is
the default.  *nlaunch_total* requires *slave* and
:doc:`fix particle/weight <fix_particle_weight>`.  The *slave* IDs
must be :doc:`surf_collide vanish <surf_collide>` models.

Related commands
----------------

:doc:`fix emit/surf <fix_emit_surf>`,
:doc:`fix surface/emit/source <fix_surface_emit_source>`,
:doc:`fix surface/emit/recycle <fix_surface_emit_recycle>`,
:doc:`fix particle/weight <fix_particle_weight>`,
:doc:`surf_collide <surf_collide>`

Default
-------

The option defaults are n = 0 (flow mode), normal = no, stop_at_np = 0
(no limit), nevery = 1 and perspecies = yes.  There is no default
slave, nlaunch_total or region.
