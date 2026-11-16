.. index:: fix dt/reset

fix dt/reset command
====================

Syntax
------

.. parsed-literal::

   fix ID dt/reset Nfreq step weight resetflag

* ID is documented in :doc:`fix <fix>` command

* dt/reset = style name of this fix command

* Nfreq = perform timestep calculation every this many steps

* step = compute or fix column for per-grid cell timestep, prefaced by "c\_" or "f\_"

* weight = weight (0.0 to 1.0) applied to average per-cell timestep when calculating global timestep

* resetflag = 1 to overwrite global timestep with new timestep, 0 to just calculate new timestep

Examples
--------

.. parsed-literal::

   compute 1 grid all mymixture nrho temp usq vsq wsq
   fix 1 ave/grid all 10 50 500 c_1[\*]
   compute mct lambda/grid f_1[1] f_1[2] tau
   compute tstep dt/grid all 0.25 0.1 c_mct f_1[2] f_1[3] f_1[4] f_1[5]

.. parsed-literal::

   fix 2 dt/reset 500 c_tstep 0.1 1

Description
-----------

Calculate a new global timestep for the simulation based on per grid
cell timesteps calculated by a compute or fix.  The new global
timestep can be output by the :doc:`stats_style <stats_style>` command.
Or it can be used to overwrite the current global timestep for a
variable time simulation.  See this
:ref:`section <howto_14>` of the manual for more
information on variable timestep simulations.

The *Nfreq* argument specifies how often the global timestep is calculated.

The *step* argument specifies a compute which calculates a per grid
cell timestep.  Or it specifies a fix which time averages a per grid
cell timestep.  Currently the only compute that calculates a per grid
cell timestep is :doc:`compute dt/grid <compute_dt_grid>`.  The :doc:`fix ave/grid <fix_ave_grid>` command could perform a time average of
the compute.

This is done by specifying the *step* argument like this:

* c_ID = compute with ID that calculates a per grid cell timestep as a vector output
* c_ID[m] = compute with ID that calculates a timestep as its Mth column of array output
* f_ID[m] = fix with ID that calculates a time-averaged timestep as a vector output
* f_ID[m] = fix with ID that calculates a time-averaged timestep as its Mth column of array output

.. warning::

   If the ID of a :doc:`fix ave/grid <fix_ave_grid>`
   command is used as the *step* argument, it only produces output on
   timesteps that are multiples of its *Nfreq* argument.  Thus this fix
   can only be invoked on those timesteps.

Note that some of the per-cell timesteps may be zero for several reasons.  First,
data used to calculate the timestep, such as mean collision time, temperature, or particle speed, may be zero.
Also, some cells may not contain particles, either due to their type or to local flow conditions.
For example, split cells (in which sub cells store the particles) and cells interior to surface
objects do not store particles.  See :ref:`Section 4.10 <howto_12>` of the manual for
details of how OpenEdge defines child, unsplit, split, and sub cells.

From the per-cell timesteps, 3 values are extracted by this fix.  They
are the minimum positive timestep (DTmin) for all cells, the maximum positive timestep
(DTmax) for all cells, and the average positive timestep (DTave) over all
cells.  Cells with a timestep value of zero are not included in the mininum,
maximum, and average timestep calculations.

A new global timestep is than calculated by this formula, using
the specified *weight* argument:

.. parsed-literal::

   DTnew = (1-weight)\*DTmin + weight\*DTave

If the *resetflag* argument is specified as 1, then the global
timestep for the simulation, initially specified by the
:doc:`timestep <timestep>` command, is overwritten with the new DTnew
value.  If *resetflag* is 0, then the global timestep is not changed.

----------

Restart, output info
--------------------

No information about this fix is written to :doc:`binary restart files <restart>`.

This fix computes a global scalar which is the new global timestep
(DTnew above) after the most recent timestep re-calculation.  This
value is accessible to other commands whether or not the global
timestep is overwritten with the new value.

It also computes a global vector of length 3 with these values:

* 1 = DTmin
* 2 = DTmax
* 3 = DTave

Related commands
----------------

:doc:`compute dt/grid <compute_dt_grid>`

**Default:** none
