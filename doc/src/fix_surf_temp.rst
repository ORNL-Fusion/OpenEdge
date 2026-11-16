.. index:: fix surf/temp

fix surf/temp command
=====================

Syntax
------

.. parsed-literal::

   fix ID surf/temp surf-ID Nevery source Tinit emisurf custom-name

* ID is documented in :doc:`fix <fix>` command

* surf/temp = style name of this fix command

* surf-ID = group ID for which surface elements to consider

* Nevery = adjust surface temperature once every Nevery steps

* source = computeID or fixID

  .. parsed-literal::

       computeID = c_ID or c_ID[N] for a compute that calculates per surf values
       fixID = f_ID or f_ID[N] for a fix that calculates per surf values

* Tinit = initial temperature of surface (temperature units)

* emisurf = emissivity of the surface (unitless, 0 < emisurf <= 1)

* custom-name = name of a custom per-surf variable to create

Examples
--------

.. parsed-literal::

   compute 1 surf all all etot
   fix 1 surf/temp all 1000 c_1 250 0.9 temperature
   surf_collide 1 diffuse s_temperature 0.5

Description
-----------

Compute a new temperature for each surface element in the group once
every Nevery steps.  This command uses a compute or fix ID which
should compute the flux of total energy onto each surface element,
e.g. *etot* from the :doc:`compute surf <compute_surf>` command.  Note
that OpenEdge does not check that the specified compute/fix calculates
an energy flux.

This fix creates a custom per-surf floating point vector with the
specified name of *custom-name*.  It can be accessed by other commands
which use the temperature of surface elements.  An example is the
:doc:`surf_collide diffuse <surf_collide>` command, as shown above.

The per-surface element temperatures computed by this fix can be
output via the :doc:`dump surf <dump>` command, using its *s_name*
syntax.  See the examples/adjust_temp directory for input scripts that
use this fix.

The specified *group-ID* must be the name of a surface element group,
as defined by the :doc:`group surf <group>` command, which contains a
set of surface elements.

The specfied *Nevery* determines how often the surface temperatures
are re-computed.

The *source* can be specified as a per-surf quantity calculated by a
compute, such as the :doc:`compute surf <compute_surf>` command.  Or it
can be specified a per-surf quantity calculated by a fix, e.g. one
which performs time-averaging of per-surf quantities, such as :doc:`fix ave/surf <fix_ave_surf>`.

If the specified *source* has no bracketed term appended, the compute
or fix must calculate a per-surf vector.  If *c_ID[N]* or
*f_ID[N]* is used, then N must be in the range from 1-M, which will
use the Nth column of the M-column per-surf array calculated by the
compute or fix.

The temperature of each surface element is calculated from the
Stefan-Boltzmann law for a gray-body as follows:

.. parsed-literal::

   q_wall = sigma \* emisurf \* Tsurf^4

where q_wall is the heat flux to the surface (provided by the compute
or fix), sigma is the Stefan-Boltzmann constant appropriate to the
:doc:`units <units>` being used, *emisurf* is the surface emissivity,
and *Tsurf* is the resulting surface temperature.

The specified emissivity *emisurf* is a unitless coefficient > 0.0 and
<= 1.0, which determines the emissivity of the surface.  An emissivity
coefficient of 1.0 means the surface is a black-body that radiates all
the energy it receives.

The specified *Tinit* value is used to set the initial temperature of
every surface element in the system.  New temperature values for only
the surface elements in the *surf-ID* group will be reset every
*Nevery* timesteps by the formula above.  Thus temperature values for
surfaces not in the *surf-ID* group will always be *Tinit*.

Note that commands which use these temperature values can determine
which surface element values they access by their own *surf-ID* group.
E.g. the :doc:`surf_collide diffuse <surf_collide>` command is assigned
to a group of surface elements via the :doc:`surf_modify <surf_modify>`
command.  It its Tsurf value is set to the custom vector defined by
this fix, then you likely want the two surface groups to be
consistent.  Note that it also possible to define multiple
:doc:`surf_collide diffuse <surf_collide>` and multiple fix surf/temp
commands, each pair of which use a different surface group and
different custom per-surf vector name.

----------

Restart, output info
--------------------

No information about this fix is written to :doc:`binary restart files <restart>`.

However, the values of the custom particle attribute defined by this
fix is written to the restart file.  Namely the floating-point vector
of temperature values for each surface with the name assigned by this
command.  As explained on the :doc:`read_restart <read_restart>` doc
page these values will be re-assigned to surface when a restart file
is read.  If a new fix surf/temp command is specified in the restart
script as well as a surface collision model which uses the custom
attribute updated by this fix, then the per-surf temperatures and
updating process will continue to be used in the continued run.

No global or per-surf quantities are stored by this fix for access by
various output commands.

However, the custom per-surf attribute defined by this fix can be
accessed by the :doc:`dump surf <dump>` command, as s_name.  That means
those per-surf values can be written to surface dump files.

Restrictions
------------

This fix can only be used in simulations that define explicit
surfaces, not for implicit surface models.

**Related commands:** none

**Default:** none
