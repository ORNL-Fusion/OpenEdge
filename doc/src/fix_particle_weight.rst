.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix particle/weight

fix particle/weight command
===========================

Syntax
------

.. parsed-literal::

   fix ID particle/weight

* ID is documented in the :doc:`fix <fix>` command
* particle/weight = style name of this fix command

Examples
--------

.. code-block:: text

   fix pw particle/weight
   fix femit surface/emit/source LiSource divertor cevap perspecies no &
       nlaunch_total 200 model thermal
   compute cgw grid/weighted all LiAll nrho_w u_w temp_w

Description
-----------

Create the per-particle statistical weight :math:`w`, the number of
real particles that one simulation particle represents.  The weight is
stored in a custom per-particle floating-point attribute named
*pweight* (see the :doc:`custom <custom>` command).

Emission commands that launch particles of unequal weight, such as
:doc:`fix surface/emit/source <fix_surface_emit_source>`, write
:math:`w` when they insert a particle.  The weighted diagnostics, such
as :doc:`compute grid/weighted <compute_grid_weighted>`, read it.  An
emitter in a weighted launch mode stops with an error if the attribute
does not exist, so this fix must be defined before such an emitter is
used.

A particle whose weight is zero receives the default
:math:`w = F_\mathrm{num}`, the *fnum* setting of the
:doc:`global <global>` command.  This is done at the start of a run
for all existing particles, and at insertion for every new particle
that its emitter did not weight, for example particles from
:doc:`create_particles <create_particles>`,
:doc:`fix emit/face <fix_emit_face>` or a second reaction product of
:doc:`fix volume/chem/adas <fix_volume_chem_adas>`.  A
non-zero weight is never overwritten.  With this default the weighted
diagnostics of a run without weighted emitters equal the unweighted
ones.  An atom sputtered by
:doc:`surf_react surface/pwi <surf_react_surface_pwi>` does not take
the default.  It inherits the weight of the incident particle.

Like any custom attribute, *pweight* follows the particle through
sorting and migration between processors.  A clone made by the cell
weighting of the :doc:`global <global>` *weight* keyword receives a
copy of the weight of the particle it is cloned from.

Restart and output
^^^^^^^^^^^^^^^^^^

The *pweight* attribute is written to
:doc:`binary restart files <restart>` with the particles.  If it
already exists when the fix is defined, for example after
:doc:`read_restart <read_restart>`, the existing attribute and its
values are kept.

The fix produces no output of its own.  The weights can be written by
the :doc:`dump particle <dump>` command as *p_pweight*.

Restrictions
------------

This command is part of the OPENEDGE package.

The command takes no arguments.  Only one fix of this style should be
defined in a simulation.

Related commands
----------------

:doc:`compute grid/weighted <compute_grid_weighted>`,
:doc:`fix surface/emit/source <fix_surface_emit_source>`,
:doc:`custom <custom>`

Default
-------

none
