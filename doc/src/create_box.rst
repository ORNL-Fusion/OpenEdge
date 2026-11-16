.. index:: create_box

create_box command
==================

Syntax
------

.. parsed-literal::

   create_box xlo xhi ylo yhi zlo zhi

.. parsed-literal::

   xlo,xhi = box bounds in the x dimension (distance units)
   ylo,yhi = box bounds in the y dimension (distance units)
   zlo,zhi = box bounds in the z dimension (distance units)

Examples
--------

.. parsed-literal::

   create_box 0 1 0 1 0 1
   create_box 0 1 0 1 -0.5 0.5
   create_box 0 10.0 0 5.0 -4.0 0.0

Description
-----------

Set the size of the simulation box.

For a 2d simulation, as specifed by the :doc:`dimension <dimension>`
command, *zlo* < 0.0 and *zhi* > 0.0 is required.  This means the z
dimensions straddle 0.0.  Typical values are -0.5 and 0.5, but this is
not required.  See :ref:`Section 4.1 <howto_1>` of the
manual for more information about 2d simulations.

For 2d axisymmetric simulations, as set by the
:doc:`dimension <dimension>` and :doc:`boundary <boundary>` commands, the
ylo setting must be 0.0.  See :ref:`Section 4.2 <howto_2>`
of the manual for more information about axisymmetric simulations.

**Restrictions:** none

**Related commands:** none

**Default:** none
