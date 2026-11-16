.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix particle/tag

fix particle/tag command
========================

Syntax
------

.. parsed-literal::

   fix ID particle/tag group-ID keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* particle/tag = style name of this fix command
* group-ID = surface group whose elements mark a particle that hits
  them, or *none*
* zero or more keyword/value pairs may be appended
* keyword = *psi* or *nevery*

  * *psi* values = psiN fix-ID

    * psiN = a particle is marked where the normalized poloidal flux
      is below this value
    * fix-ID = ID of a :doc:`fix reflect/psi <fix_reflect_psi>` that
      supplies :math:`\psi_N` at the particle position

  * *nevery* value = N = test the *psi* condition every this many
    timesteps

Examples
--------

.. code-block:: text

   fix ftag particle/tag core
   fix ftag particle/tag none psi 0.9 fcore nevery 5

Description
-----------

Mark particles that have visited a chosen region.  The fix creates a
custom per-particle integer attribute named *transit*.  It is 0 for a
new particle and is set to 1 when one of two conditions is met.  It is
never reset to 0.

The first condition is a hit on a surface element of *group-ID*, for
example the core boundary of a wedge.  The flag is set in the particle
move when the collision with the element is processed, whatever
collision model the surface carries.  Particles that never hit the
group cost nothing.  With *group-ID* = *none* this condition is off.

The second condition is set by the *psi* keyword.  Every *nevery*
steps, at the end of the step, each particle that is not yet marked is
tested, and it is marked if :math:`\psi_N < \mathrm{psiN}` at its
position.  This costs one flux-map lookup per unmarked particle on the
steps where the test runs.  A negative psiN turns the test off.

Both conditions can be given, and the flag is set by whichever is met
first.  At least one must be given.

The flag is a custom per-particle attribute.  It moves with the
particle between processors, including a rebalancing of the grid, and
it is written to :doc:`binary restart files <restart>`.  If the
attribute already exists when the fix is defined, for example after
:doc:`read_restart <read_restart>`, its values are kept.  A reaction
of :doc:`fix volume/chem/adas <fix_volume_chem_adas>` that changes the
species of a particle in place, such as ionization, recombination or
charge exchange, keeps the flag.  A second product of a reaction is a
new particle and starts with 0.

The *ehist* impact histograms of
:doc:`surf_react surface/pwi <surf_react_surface_pwi>` have a separate
column for marked particles.  The energy and angle distribution of the
ions that reach a wall area is thereby split into ions that came
through the marked region and ions that did not.  The attribute can
also be written by the :doc:`dump particle <dump>` command as
*p_transit*.

Restrictions
------------

This command is part of the OPENEDGE package.

Only one fix particle/tag can be defined.  The *psi* keyword needs a
:doc:`fix reflect/psi <fix_reflect_psi>` on a background that has a
flux map.

Related commands
----------------

:doc:`surf_react surface/pwi <surf_react_surface_pwi>`,
:doc:`fix reflect/psi <fix_reflect_psi>`

Default
-------

nevery = 10.  The *psi* condition is off unless the keyword is given.
