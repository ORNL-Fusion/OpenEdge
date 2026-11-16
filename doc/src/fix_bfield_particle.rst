.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix bfield/particle

fix bfield/particle command
===========================

Syntax
------

.. parsed-literal::

   fix ID bfield/particle Bx By Bz

* ID is documented in the :doc:`fix <fix>` command
* bfield/particle = style name of this fix command
* Bx, By, Bz = names of :doc:`particle-style variables <variable>`
  that give the components of the magnetic field (T), or NULL

Examples
--------

.. code-block:: text

   variable Bx particle 0.0
   variable By particle 0.0
   variable Bz particle 1.0
   fix fb bfield/particle Bx By Bz
   global bfield particle fb 1

   fix fb bfield/particle NULL NULL Bz0
   global bfield particle fb 0

Description
-----------

Define a magnetic field that is evaluated for each particle from
particle-style variables.  The three arguments name the variables,
without the "v\_" prefix, that give the x, y and z components of
:math:`\mathbf{B}` in the coordinates of the simulation box.  A
component given as NULL is not evaluated and is taken as zero.

The fix does not act on particles itself.  It becomes the magnetic
field of the charged-particle pusher through the *bfield particle*
setting of the :doc:`global <global>` command, which names this fix
and sets the interval Nfreq, in timesteps, at which the variables are
evaluated again.  At each evaluation every named variable is computed
for all particles owned by the processor, and the results are stored
in a per-particle array.

Because the field is an arbitrary formula of the particle position
and velocity, this fix suits verification cases with an analytic
field and a known solution.  A simulation of a device takes its
magnetic field from a :doc:`fix background <fix_background>` through
the *plasma* value of the *pusher* keyword of the
:doc:`global <global>` command.

Restart, output info
--------------------

No information about this fix is written to
:doc:`binary restart files <restart>`.

This fix computes a per-particle array with one column for each
non-NULL component, in x, y, z order.  The columns hold the field
values from the last evaluation and are zero before the first one.
They are read by the *bfield* setting of the :doc:`global <global>`
command and can be written by the :doc:`dump particle <dump>` command
as f_ID[N].

Restrictions
------------

This command is part of the OPENEDGE package.  It is only enabled if
OpenEdge was built with that package.  See
:ref:`Section 2.3 <start_3>` for more info.

Exactly three arguments must be given.  Each non-NULL argument must
name a particle-style variable that exists when the run begins.

Related commands
----------------

:doc:`fix efield/particle <fix_efield_particle>`,
:doc:`global <global>`, :doc:`variable <variable>`

Default
-------

none
