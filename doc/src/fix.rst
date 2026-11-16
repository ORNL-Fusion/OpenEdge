.. index:: fix

fix command
===========

Syntax
------

.. parsed-literal::

   fix ID style args

* ID = user-assigned name for the fix
* style = one of a long list of possible style names (see below)
* args = arguments used by a particular style

Examples
--------

.. parsed-literal::

   fix 1 grid/check 100 warn
   fix 1 ave/time all 100 5 1000 c_myTemp c_thermo_temp file temp.profile

Description
-----------

Set a fix that will be applied to the system.  In OpenEdge, a "fix" is
an operation that is applied to the system during timestepping.
Examples include adding particles via inlet boundary conditions or
computing diagnostics.  Code for new fixes can be added to OpenEdge; see
the source code of the manual for details.

Fixes perform their operations at different stages of the timestep.
If 2 or more fixes operate at the same stage of the timestep, they are
invoked in the order they were specified in the input script.

The ID for a fix is used to identify the fix in other commands.  Each
fix ID must be unique; see an exception below.  The ID can only
contain alphanumeric characters and underscores.  You can specify
multiple fixes of the same style so long as they have different IDs.
A fix can be deleted with the :doc:`unfix <unfix>` command, after which
its ID can be re-used.

.. warning::

   The :doc:`unfix <unfix>` command is the only way to turn
   off a fix; simply specifying a new fix with the same style and a
   different ID will not turn off the first one.

If you specify a new fix with the same ID and style as an existing
fix, the old fix is deleted and the new one is created (presumably
with new settings).  This is the same as if an "unfix" command were
first performed on the old fix, except that the new fix is kept in the
same order relative to the existing fixes as the old one originally
was.

Some fixes store an internal "state" which is written to binary
restart files via the :doc:`restart <restart>` or
:doc:`write_restart <write_restart>` commands.  This allows the fix to
continue on with its calculations in a restarted simulation.  See the
:doc:`read_restart <read_restart>` command for info on how to re-specify
a fix in an input script that reads a restart file.  See the doc pages
for individual fixes for info on which ones can be restarted.

----------

Each fix style has its own doc page which describes its arguments and
what it does, as listed below.  Here is an alphabetic list of fix
styles available in OpenEdge:

* :doc:`ablate <fix_ablate>` - alter implicit surfaces within each grid cell
* :doc:`adapt <fix_adapt>` - on-the-fly grid adaptation
* :doc:`ambipolar <fix_ambipolar>` - ambipolar approximation for ionized plasmas
* :doc:`ave/grid <fix_ave_grid>` - compute per grid cell time-averaged quantities
* :doc:`ave/histo <fix_ave_histo>` - compute/output time averaged histograms
* :doc:`ave/histo/weight <fix_ave_histo>` - compute/output weighted histograms
* :doc:`ave/surf <fix_ave_surf>` - compute per surface element time-averaged quantities
* :doc:`ave/time <fix_ave_time>` - compute/output global time-averaged quantities
* :doc:`background <fix_background>` - load the background plasma and magnetic equilibrium shared by other commands
* :doc:`balance <fix_balance>` - perform dynamic load-balancing
* :doc:`bfield/particle <fix_bfield_particle>` - gyro-motion of charged particles in the background magnetic field
* :doc:`coulomb/background <fix_coulomb_background>` - Coulomb collisions of charged particles with the background plasma
* :doc:`coulomb/binary <fix_coulomb_binary>` - binary Coulomb collisions between charged particles in a cell
* :doc:`core/transport <fix_core_transport>` - 1d core impurity transport model coupled to the particles at a flux surface
* :doc:`cross_field_diffusion <fix_cross_field_diffusion>` - anomalous cross-field diffusion of charged particles
* :doc:`dt/monitor <fix_dt_monitor>` - monitor the timestep against gyromotion and reaction rates
* :doc:`dt/reset <fix_dt_reset>` - adjust global timestep dynamically
* :doc:`efield/particle <fix_efield_particle>` - acceleration of charged particles by the background electric field
* :doc:`emit/face <fix_emit_face>` - emit particles at global boundaries
* :doc:`emit/surf <fix_emit_surf>` - emit particles at surfaces
* :doc:`force/thermal <fix_force_thermal>` - parallel friction and thermal forces on impurity ions
* :doc:`grid/check <fix_grid_check>` - check if particles are in the correct grid cell
* :doc:`halt <fix_halt>` - stop the simulation early if a condition is met
* :doc:`move/surf <fix_move_surf>` - move surfaces dynamically during a simulation
* :doc:`neutral/collide/background <fix_neutral_collide_background>` - collisions of neutrals with the background plasma (ionization, charge exchange)
* :doc:`particle/tag <fix_particle_tag>` - flag particles that hit a surface group or enter psiN below a threshold
* :doc:`particle/weight <fix_particle_weight>` - per-particle statistical weights
* :doc:`particulate/charge <fix_particulate_charge>` - charging of dust grains and droplets by the background plasma
* :doc:`particulate/drag <fix_particulate_drag>` - plasma drag forces on dust grains and droplets
* :doc:`particulate/emit <fix_particulate_emit>` - inject dust grains or droplets
* :doc:`particulate/thermal <fix_particulate_thermal>` - heating, evaporation and sublimation of dust grains and droplets
* :doc:`print <fix_print>` - print text and variables during a simulation
* :doc:`reflect/psi <fix_reflect_psi>` - reflect particles that cross a normalized-flux (psi) boundary
* :doc:`store/force <fix_store_force>` - store one named per-particle transport-force contribution
* :doc:`surf/temp <fix_surf_temp>` - compute per-surf temperatures dynamically
* :doc:`surface/emit/puff <fix_surface_emit_puff>` - emit a gas puff from surfaces at a prescribed rate
* :doc:`surface/emit/recycle <fix_surface_emit_recycle>` - emit recycled neutrals from surfaces at the plasma wall flux
* :doc:`surface/emit/source <fix_surface_emit_source>` - emit particles from surfaces at a per-element source rate (e.g. sputtering)
* :doc:`surface/state/lm <fix_surface_state_lm>` - liquid-metal film and surface temperature model on a divertor strip
* :doc:`temp/global/rescale <fix_temp_global_rescale>` - rescale particle temperatures
* :doc:`temp/rescale <fix_temp_rescale>` - rescale particle temperatures within each grid cell
* :doc:`vibmode <fix_vibmode>` - discrete vibrational energy modes
* :doc:`volume/chem/adas <fix_volume_chem_adas>` - ionization and recombination of impurities with ADAS rate coefficients

There are also additional accelerated compute styles included in the
OpenEdge distribution for faster performance on specific hardware.  The
list of these with links to the individual styles are given in the
pair section of :ref:`this page <cmd_5>`.

----------

In addition to the operation they perform, some fixes also produce one
of four styles of quantities: global, per-particle, per-grid, or
per-surf.  These can be used by other commands or output as described
below.  A global quantity is one or more system-wide values, e.g. the
temperature of the system.  A per-particle quantity is one or more
values per particle, e.g. the kinetic energy of each particle.  A
per-grid quantity is one or more values per grid cell.  A per-surf
quantity is one or more values per surface element.

Global, per-particle, per-grid, and per-surf quantities each come in
two forms: a single scalar value or a vector of values.  Additionaly,
global quantities can also be a 2d array of values.  The doc page for
each fix describes the style and kind of values it produces, e.g. a
per-particle vector.  Some fixes can produce more than one form of a
single style, e.g. a global scalar and a global vector.

When a fix quantity is accessed, as in many of the output commands
discussed below, it can be referenced via the following bracket
notation, where ID is the ID of the fix:

.. list-table::

   * - f_ID
     - entire scalar, vector, or array
   * - f_ID[I]
     - one element of vector, one column of array
   * - f_ID[I][J]
     - one element of array

In other words, using one bracket reduces the dimension of the
quantity once (vector -> scalar, array -> vector).  Using two brackets
reduces the dimension twice (array -> scalar).  Thus a command that
uses scalar fix values as input can also process elements of a vector
or array.

Note that commands and :doc:`variables <variable>` which use fix
quantities typically do not allow for all kinds, e.g. a command may
require a vector of values, not a scalar.  This means there is no
ambiguity about referring to a fix quantity as f_ID even if it
produces, for example, both a scalar and vector.  The doc pages for
various commands explain the details.

----------

Any values generated by a fix can be used in several ways:

* Global values can be output via the :doc:`stats_style <stats_style>`
  command.  Or the values can be referenced in a :doc:`variable equal <variable>` or :doc:`variable atom <variable>` command.

* Per-particle values can be output via the :doc:`dump particle <dump>`
  command.  Or the per-particle values can be referenced in an
  :doc:`particle-style variable <variable>`.

* Per-grid values can be output via the :doc:`dump grid <dump>` command.
  Or the per-grid values can be referenced in a :doc:`grid-style variable <variable>`.

----------

**Restrictions:** none

Related commands
----------------

:doc:`unfix <unfix>`

**Default:** none
