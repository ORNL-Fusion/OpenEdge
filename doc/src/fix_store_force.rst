.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix store/force

fix store/force command
=======================

Syntax
------

.. parsed-literal::

   fix ID store/force channel keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* store/force = style name of this fix command
* channel = *electric/plasma* or *electric/sheath* or *magnetic* or
  *thermal/ion* or *thermal/electron* or *coulomb/background* or
  *coulomb/binary*
* zero or more keyword/value pairs may be appended
* keyword = *id_max*

  * *id_max* value = largest particle ID to record, or 0 for every
    particle

Examples
--------

.. code-block:: text

   fix fEplasma store/force electric/plasma
   fix fEsheath store/force electric/sheath
   fix fBmag store/force magnetic
   fix fTion store/force thermal/ion
   fix fTelec store/force thermal/electron
   fix fNanbu store/force coulomb/background id_max 10000

   dump dforce particle all 10 forces.dump id type x y z &
       f_fEplasma[1] f_fEplasma[2] f_fEplasma[3] &
       f_fEsheath[1] f_fEsheath[2] f_fEsheath[3]

Description
-----------

Store, for each particle, the force that one transport mechanism
exerted on it during the current timestep.  The *channel* argument
names the mechanism, so each contribution is kept in its own fix
instead of one lumped vector.

Each operator that feeds a channel reports the momentum change
:math:`\Delta\mathbf{p}_k` it applied to the particle and the time
interval :math:`\tau_k` over which that change was integrated.  The
stored force is

.. math::

   \mathbf{F} = \sum_k \frac{\Delta\mathbf{p}_k}{\tau_k} ,

where the sum runs over all applications of the operator to the
particle within the timestep, including half-kicks and subcycles.  The
value is reset to zero at the start of each timestep.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Channel
     - Recorded momentum change
   * - *electric/plasma*
     - electric kicks of the Boris scheme,
       :math:`q\mathbf{E}\,\delta t` per substep, with the electric
       field the pusher uses: that of the plasma provider or, where it
       returns none, that of the *efield* keyword
   * - *electric/sheath*
     - velocity changes that the sheath model makes inside the Boris
       push: the potential update of every substep in *spatial* mode
       and the reflection or slowing down of an outgoing particle in
       *boundary* mode
   * - *magnetic*
     - velocity rotation of the Boris scheme, taken as the velocity
       change of a substep minus its electric kicks
   * - *thermal/ion*
     - ion thermal force of
       :doc:`fix force/thermal <fix_force_thermal>`
   * - *thermal/electron*
     - electron thermal force of
       :doc:`fix force/thermal <fix_force_thermal>`
   * - *coulomb/background*
     - collision kick of
       :doc:`fix coulomb/background <fix_coulomb_background>`
   * - *coulomb/binary*
     - collision kick of
       :doc:`fix coulomb/binary <fix_coulomb_binary>`

For the three pusher channels the momentum changes of all substeps
of a push are summed and divided by the timestep :math:`\Delta t`.
The wall kick of the *kick* and *boundary* sheath modes is applied in
the surface collision and is not recorded.  The modes and
:math:`\delta t` are described on the :doc:`global <global>` page.

The two thermal channels are recorded separately even when one
:doc:`fix force/thermal <fix_force_thermal>` applies both terms.
Each of its two half-kicks is recorded with the full interval
:math:`\Delta t_N`, so their sum is the force of that step.

The Coulomb channels hold the momentum kick of the collision divided
by the collision interval :math:`\Delta t_N`.  The value is zero on a
timestep on which the collision fix does not act.  The kick is a
stochastic quantity.  A single
value is not the deterministic friction force of a fluid model.
Average it over particles or over time, keeping the sign, before
comparing it with a mean friction force.

Wall collisions, surface reactions, reflection by
:doc:`fix reflect/psi <fix_reflect_psi>`, cross-field diffusion and
pinch are not recorded.  They are boundary events or displacements in
position and do not enter the velocity update as forces.

With *id_max 0* all particles are recorded.  With a positive *id_max*
the particles with a larger ID keep zero force.  This reduces the
diagnostic work.  The storage is still allocated for every particle,
so that the values migrate with the particles between processors.

Output info
-----------

This fix computes a per-particle array with three columns, the force
components in N.  They can be accessed as f_ID[1], f_ID[2] and f_ID[3]
by :doc:`dump particle <dump>` and by other commands that read
per-particle fix output.  The components are in the coordinate order
of the simulation box.  In a 2d axisymmetric simulation
:math:`(x, y, z) = (Z, R, \phi)`, so the columns are
:math:`(F_Z, F_R, F_\phi)`.

The values are held in per-particle custom attributes named
*store_force_ID* and *store_force_step_ID*, with ID the ID of this
fix.

Restrictions
------------

This command is part of the OPENEDGE package.  It is not available
with the KOKKOS package.

Only one fix store/force may store a given channel.

The *electric/plasma*, *electric/sheath* and *magnetic* channels
require the Boris mode of the *pusher* keyword of the
:doc:`global <global>` command.  The force on a particle advanced by
the guiding-centre pusher is not decomposed.

Related commands
----------------

:doc:`dump <dump>`, :doc:`global pusher <global>`,
:doc:`fix force/thermal <fix_force_thermal>`,
:doc:`fix coulomb/background <fix_coulomb_background>`,
:doc:`fix coulomb/binary <fix_coulomb_binary>`

Default
-------

id_max = 0
