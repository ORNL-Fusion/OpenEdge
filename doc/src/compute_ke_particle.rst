.. index:: compute ke/particle

compute ke/particle command
===========================

.. index:: compute ke/particle/kk

compute ke/particle/kk command
==============================

Syntax
------

.. parsed-literal::

   compute ID ke/particle

* ID is documented in :doc:`compute <compute>` command
* ke/particle = style name of this compute command

Examples
--------

.. parsed-literal::

   compute 1 ke/particle

Description
-----------

Define a computation that calculates the per-atom translational
kinetic energy for each particle.

The results of this compute can be used by different commands in
different ways.  The values for a single timestep can be output by the
:doc:`dump particle <dump>` command.

The kinetic energy is

.. parsed-literal::

   Vsq = Vx\*Vx + Vy\*Vy + Vz\*Vz
   KE = 1/2 m Vsq

where m is the mass and (Vx,Vy,Vz) are the velocity components of the
particle.

Output info
-----------

This compute calculates a per-particle vector, which can be accessed
by any command that uses per-particle values from a compute as input.

The vector can be accessed by any command that uses per-particle
values from a compute as input.  See :ref:`Section 4.4 <howto_8>` for an overview of OpenEdge output
options.

The per-particle vector values will be in energy :doc:`units <units>`.

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

Related commands
----------------

:doc:`dump particle <dump>`

**Default:** none
