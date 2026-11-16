.. index:: compute temp

compute temp command
====================

.. index:: compute temp/kk

compute temp/kk command
=======================

Syntax
------

.. parsed-literal::

   compute ID temp

* ID is documented in :doc:`compute <compute>` command
* temp = style name of this compute command

Examples
--------

.. parsed-literal::

   compute 1 temp
   compute myTemp temp

Description
-----------

Define a computation that calculates the temperature of all particles.

The temperature is calculated by the formula KE = dim/2 N kB T, where
KE = total kinetic energy of the particles (sum of 1/2 m v^2), dim =
dimensionality of the simulation, N = number of particles, kB =
Boltzmann constant, and T = temperature.

Note that this definition of temperature does not subtract out a net
streaming velocity for particles, so it is not a thermal temperature
when the particles have a non-zero streaming velocity.  See the
:doc:`compute thermal/grid <compute_thermal_grid>` command for
calculation of thermal temperatures on a per grid cell basis.

----------

Output info
-----------

This compute calculates a global scalar (the temperature).  This value
can be used by any command that uses global scalar values from a
compute as input.  See :ref:`Section 4.8 <howto_10>` for an
overview of OpenEdge output options.

The scalar value will be in temperature :doc:`units <units>`.

----------

Styles with a *kk* suffix are functionally the same as the
corresponding style without the suffix.  They have been optimized to
run faster, depending on your available hardware, as discussed in the
:ref:`Building OpenEdge <start_2>` section of the manual.
The accelerated styles take the same arguments and should produce the
same results, except for different random number, round-off and
precision issues.

These accelerated styles are part of the KOKKOS package. They are only
enabled if OpenEdge was built with that package.  See the :ref:`Making OpenEdge <start_3>` section for more info.

You can specify the accelerated styles explicitly in your input script
by including their suffix, or you can use the :ref:`-suffix command-line switch <start_5>` when you invoke OpenEdge, or you can
use the :doc:`suffix <suffix>` command in your input script.

See the :ref:`Building OpenEdge <start_2>` section of the
manual for more instructions on how to use the accelerated styles
effectively.

----------

**Restrictions:** none

**Related commands:** none

**Default:** none
