.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix dt/monitor

fix dt/monitor command
======================

Syntax
------

.. parsed-literal::

   fix ID dt/monitor N

* ID is documented in the :doc:`fix <fix>` command
* dt/monitor = style name of this fix command
* N = report every N timesteps

Examples
--------

.. code-block:: text

   fix dtm dt/monitor 1000
   stats_style step np f_dtm[1] f_dtm[2] f_dtm[3] f_dtm[4]

Description
-----------

Monitor how well the timestep resolves the gyromotion of charged
particles and the volume reactions.  The numbers are accumulated
inside the operators while they run, so particles that hit a wall or
react between two reports are included.  The fix reads them every N
timesteps, sums them over processors, and prints a table at the end of
each run.

Monitored quantities
^^^^^^^^^^^^^^^^^^^^

Two quantities are monitored.  For gyromotion, in each Boris substep of
the :doc:`pusher <global>`,

.. math::
   :label: dtmon-gyro

   h_g = \left|\frac{q}{m}\right| |\mathbf{B}|\, \Delta t_{sub} ,
   \qquad \Delta t_{sub} = \frac{\Delta t}{N_{sub}} ,

where :math:`N_{sub}` is the number of subcycles.  For reactions, in
each update of :doc:`fix volume/chem/adas <fix_volume_chem_adas>`,

.. math::
   :label: dtmon-react

   h_r = \Delta t_{chem} \sum_k \nu_k ,

the sum of the reaction rates :math:`\nu_k` of the particle times the
interval :math:`\Delta t_{chem}` of that fix.

Each is compared with an advisory level :math:`L`: the *bad_dt_limit*
of the :doc:`global pusher <global>` command for gyromotion (default
0.1) and 0.1 for reactions.  For each quantity the fix reports the
largest value and the fraction above the level,

.. math::
   :label: dtmon-fractions

   F_g = \frac{\sum w\,\Delta t_{sub}\,[h_g > L]}{\sum w\,\Delta t_{sub}} ,
   \qquad
   F_r = \frac{\sum w\,[h_r > L]}{\sum w} ,

where :math:`w` is the particle weight (the *pweight* attribute of
:doc:`fix particle/weight <fix_particle_weight>`, or *fnum*).
:math:`F_g` is a fraction of the Boris exposure, weight times time,
summed over the substeps.  :math:`F_r` is a fraction of the weight
summed over the updates of the chemistry fix, with or without a
reaction, and is attributed to the species the particle had before the
update.  Neither is a fraction of the number of particles.  Sums over
processors are taken before the division.

Walls and guiding centres
^^^^^^^^^^^^^^^^^^^^^^^^^

When a particle hits a wall, the time left in its step is removed from
its Boris exposure, on whichever processor the impact occurs and with
the species and weight it had before the impact; periodic caps and
transparent surfaces remove nothing.  The part above the level is
reduced in the same proportion as the whole push, and the largest value
still includes every substep that was integrated.  A particle that
leaves through an open boundary of the box keeps the exposure of its
full step.  The exposure and :math:`F_g` are therefore approximate, and
the table says so.  The fix keeps the state it needs in a custom
per-particle attribute named *dtm_push*.

Particles pushed as guiding centres do not resolve the gyromotion by
design: their exposure is listed in the table and is not assessed.
Neutral particles are not part of the gyromotion numbers.

Warnings and the run table
^^^^^^^^^^^^^^^^^^^^^^^^^^

A warning is printed the first time in a run that a report finds a
value above a level.  It gives the measured value, the level, the
species and what to change.  For gyromotion that is the number of
subcycles that would meet the level for the fields seen up to then, or
a smaller timestep; this addresses the gyromotion only.  The reaction
interval does not change with the subcycles: it is reduced by the
timestep or by the interval of
:doc:`fix volume/chem/adas <fix_volume_chem_adas>`.

The table printed at the end of a run covers the whole run, including a
last window shorter than N, and lists the values per species with one
of these states for each quantity:

* below the advisory level everywhere assessed
* EXCEEDED, with the fraction and the largest value
* not assessed: no charged particle was pushed by Boris, or no
  chemistry update took place
* INVALID: non-finite values were found

Other processes that are active in the run but have no check yet
(Coulomb collisions, cross-field diffusion, the sheath, the surface
updates of :doc:`surf_react surface/pwi <surf_react_surface_pwi>`) are
listed in the table as not assessed.

.. note::

   The advisory levels flag a timestep that is clearly coarse.  They
   are not an accuracy guarantee: a run below both levels can still be
   unconverged in the quantities of interest.  In the verification
   case ``examples/verification/surface_emission/prompt_redeposition``,
   where both quantities are equal, the redeposited fraction is within
   1 % of the exact value at 0.02 and 5 % off at 0.2.  A timestep study
   of the quantities of interest remains the test of convergence.

:doc:`fix volume/chem/adas <fix_volume_chem_adas>` caps the rate times
interval of one reaction channel at 50.  Updates with a capped channel
are counted in the table, and the largest value is then a lower bound.

Restart, output info
--------------------

No information about this fix is written to
:doc:`binary restart files <restart>`.

This fix computes a global vector of length 4 for the last reporting
window, which can be accessed by various output commands every N
timesteps: (1) the largest :math:`h_g`, (2) :math:`F_g`, (3) the
largest :math:`h_r`, (4) :math:`F_r`.  A value of -1 means the
quantity was not assessed in that window; it is never reported as
zero.

Restrictions
------------

This command is part of the OPENEDGE package.  Only one fix dt/monitor
can be defined, and it must be defined after
:doc:`fix volume/chem/adas <fix_volume_chem_adas>` so that a report
includes the chemistry of its timestep.  The tallies of a run start at
zero.  Sheath traversal, Coulomb collisions, cross-field diffusion and
surface updates are not monitored.

Related commands
----------------

:doc:`global pusher <global>`,
:doc:`fix volume/chem/adas <fix_volume_chem_adas>`,
:doc:`compute dt/grid <compute_dt_grid>`,
:doc:`fix dt/reset <fix_dt_reset>`

Default
-------

none
