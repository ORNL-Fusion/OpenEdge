.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix efield/particle

fix efield/particle command
===========================

Syntax
------

.. parsed-literal::

   fix ID efield/particle Ex Ey Ez

* ID is documented in the :doc:`fix <fix>` command
* efield/particle = style name of this fix command
* Ex, Ey, Ez = names of :doc:`particle-style variables <variable>`
  that give the components of the electric field (V/m), or NULL

Examples
--------

.. code-block:: text

   variable Ex particle 1.0e3
   fix fe efield/particle Ex NULL NULL
   fix fb bfield/particle NULL NULL Bz0
   global efield particle fe 1
   global bfield particle fb 1

Description
-----------

Define an electric field that is evaluated for each particle from
particle-style variables.  The three arguments name the variables,
without the "v\_" prefix, that give the x, y and z components of
:math:`\mathbf{E}` in the coordinates of the simulation box.  A
component given as NULL is not evaluated and is taken as zero.

The fix does not act on particles itself.  It becomes the electric
field of the charged-particle pusher through the *efield particle*
setting of the :doc:`global <global>` command, which names this fix
and sets the interval Nfreq, in timesteps, at which the variables are
evaluated again.  At each evaluation every named variable is computed
for all particles owned by the processor, and the results are stored
in a per-particle array.

This fix is the counterpart of
:doc:`fix bfield/particle <fix_bfield_particle>` for verification
cases with analytic fields, such as the
:math:`\mathbf{E}\times\mathbf{B}` and polarization-drift tests.  A
simulation of a device takes its electric field from a
:doc:`fix background <fix_background>` through the *plasma* value of
the *pusher* keyword of the :doc:`global <global>` command.

Restart, output info
--------------------

No information about this fix is written to
:doc:`binary restart files <restart>`.

This fix computes a per-particle array with one column for each
non-NULL component, in x, y, z order.  The columns hold the field
values from the last evaluation and are zero before the first one.
They are read by the *efield* setting of the :doc:`global <global>`
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

:doc:`fix bfield/particle <fix_bfield_particle>`,
:doc:`global <global>`, :doc:`variable <variable>`

Default
-------

none
