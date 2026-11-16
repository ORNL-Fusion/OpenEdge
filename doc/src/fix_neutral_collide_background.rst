.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix neutral/collide/background

fix neutral/collide/background command
======================================

Syntax
------

.. parsed-literal::

   fix ID neutral/collide/background Nevery background fix-ID channel keyword value ... [channel keyword value ...]

* ID is documented in the :doc:`fix <fix>` command
* neutral/collide/background = style name of this fix command
* Nevery = attempt collisions every this many timesteps
* *background* fix-ID = ID of a :doc:`fix background <fix_background>`
  that supplies the density, temperature and flow of the collision
  partners
* one or two channel blocks must follow
* channel = *ion* or *neutral*

  * *ion* = collide with the main ions of the background
  * *neutral* = collide with the fluid neutrals of the background

* each channel is followed by its own keyword/value pairs
* keyword = *sigma* or *gref* or *omega* or *mass_amu*

  * *sigma* value = reference cross section
    :math:`\sigma_\mathrm{ref}` at the relative speed *gref*
    (m\ :sup:`2`), required
  * *gref* value = reference relative speed :math:`g_\mathrm{ref}`
    (m/s)
  * *omega* value = exponent :math:`\omega` of the cross-section law,
    between 0.5 and 1.5
  * *mass_amu* value = mass of the collision partner (amu)

Examples
--------

.. code-block:: text

   fix pd background file plasma.h5 static yes
   fix fnc neutral/collide/background 1 background pd &
       ion sigma 5.2e-19 gref 1e4 omega 1.0 mass_amu 2.014 &
       neutral sigma 3.0e-19 gref 1e4 omega 0.5 mass_amu 2.014

Description
-----------

Collide neutral simulation particles elastically with the ions and the
fluid neutrals of the background plasma.  The background has no
particles of its own.  Every :math:`N` steps a virtual partner is
drawn for each neutral particle and each active channel, a collision
is accepted with a probability set by the local collision frequency,
and the partner is discarded.  Only the simulation particle changes
velocity, so the background acts as a reservoir of momentum and
energy.  Charged particles are not affected.

Symbols not defined here are on the :ref:`notation <notation>` page.

Collision model
^^^^^^^^^^^^^^^

The partner has mass :math:`m_\mathrm{bg}`, given by *mass_amu*.  Its
velocity is sampled from a Maxwellian of density
:math:`n_\mathrm{bg}`, temperature :math:`T_\mathrm{bg}` and drift
:math:`\mathbf{u}_\mathrm{bg}`,

.. math::

   \mathbf{v}_\mathrm{bg} = \mathbf{u}_\mathrm{bg}
      + \sqrt{\frac{T_\mathrm{bg}}{m_\mathrm{bg}}}\;\boldsymbol{\xi} ,

where :math:`\boldsymbol{\xi}` is a vector of three independent unit
normal numbers.  With the relative velocity
:math:`\mathbf{g} = \mathbf{v} - \mathbf{v}_\mathrm{bg}`, the cross
section follows the power law

.. math::
   :label: neutcoll-sigma

   \sigma(g) = \sigma_\mathrm{ref}
      \left(\frac{g_\mathrm{ref}}{g}\right)^{2\omega - 1} .

For :math:`\omega = 0.5` the cross section does not depend on speed
(hard spheres).  For :math:`\omega = 1` it varies as :math:`1/g`, the
Langevin limit of a polarization interaction, and the rate coefficient
:math:`\sigma g = \sigma_\mathrm{ref}\, g_\mathrm{ref}` is constant.

The collision frequency of the sampled pair and the probability of a
collision within the interval :math:`\Delta t_N = N\,\Delta t` are

.. math::
   :label: neutcoll-prob

   \nu = n_\mathrm{bg}\,\sigma(g)\,g ,
   \qquad
   P = 1 - \exp(-\nu\,\Delta t_N) .

The collision takes place if :math:`U < P`.  The relative velocity is
then turned into a direction :math:`\hat{\mathbf{n}}` distributed
uniformly over the unit sphere, with its magnitude unchanged, and the
particle velocity becomes

.. math::
   :label: neutcoll-post

   \mathbf{v}' = \frac{m\,\mathbf{v} + m_\mathrm{bg}\,\mathbf{v}_\mathrm{bg}}
                      {m + m_\mathrm{bg}}
      + \frac{m_\mathrm{bg}}{m + m_\mathrm{bg}}\; g\,\hat{\mathbf{n}} .

This is isotropic scattering in the centre-of-mass frame.  At most one
collision per channel is made in an interval, so :math:`\Delta t_N`
should be small compared with :math:`1/\nu`.

Channels
^^^^^^^^

The two channels are independent and each has its own cross-section
parameters.

For the *ion* channel :math:`n_\mathrm{bg} = n_i` and
:math:`T_\mathrm{bg} = T_i` at the particle position.  If the
background has no ion density or ion temperature, the electron value
is used in its place.  The partners drift along the magnetic field,
:math:`\mathbf{u}_\mathrm{bg} = u_\parallel\,\hat{\mathbf{b}}`, if the
background has both a parallel flow and a magnetic field, and do not
drift otherwise.

For the *neutral* channel :math:`n_\mathrm{bg}` and
:math:`T_\mathrm{bg}` are the neutral density and temperature of the
background, and :math:`\mathbf{u}_\mathrm{bg} = 0`.  If the background
has no neutral temperature, :math:`T_i` is used.  If it has no neutral
density, the channel does nothing and a warning is printed when the
run starts.

A channel block starts with *ion* or *neutral* and ends at the next
channel name or at the end of the command.  *sigma* must be given in
every block.

When both channels are active, the *ion* collision is attempted first
and the *neutral* collision then uses the updated velocity.  A channel
is skipped for a particle where :math:`n_\mathrm{bg}` or
:math:`T_\mathrm{bg}` is not positive.  The fix acts at the end of the
timestep, after the particle move.

The code holds no cross-section data.  The channel parameters must
be chosen for the species pair at hand.  As an example of the
input for tungsten atoms in a deuterium plasma, *ion sigma 5.2e-19
gref 1e4 omega 1.0* gives a rate coefficient
:math:`\sigma g = 5.2 \times 10^{-15}` m\ :sup:`3`/s that does not
depend on :math:`g`, and *neutral sigma 3.0e-19 gref 1e4 omega 0.5*
gives a constant cross section of
:math:`3.0 \times 10^{-19}` m\ :sup:`2`.  These numbers are example
values.  Take the values for a study from the literature.

Output info
-----------

This fix computes a global scalar, the cumulative number of collisions
made on all processors.  It can be accessed by the output commands
that read global fix values.

Restrictions
------------

This command is part of the OPENEDGE package.

It requires a :doc:`fix background <fix_background>` with
*format legacy* or a *constant* source.  Only species with zero charge
are scattered.

Each channel needs *sigma* > 0, *gref* > 0 and 0.5 <= *omega* <= 1.5.
At least one channel must be given.

Related commands
----------------

:doc:`fix coulomb/background <fix_coulomb_background>`,
:doc:`fix background <fix_background>`

Default
-------

The keyword defaults for each channel are gref = 1.0e4, omega = 0.5
and mass_amu = 2.014.  A channel is not active unless it is named.
