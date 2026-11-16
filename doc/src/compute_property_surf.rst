.. index:: compute property/surf

compute property/surf command
=============================

Syntax
------

.. parsed-literal::

   compute ID property/surf group-ID input1 input2 ...

* ID is documented in :doc:`compute <compute>` command

* property/surf = style name of this compute command

* group-ID = group ID for which surface elements to perform calculation on

* input = one or more surface element attributes

  .. parsed-literal::

       possible attributes = id, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v2y, v3z, xc, yc, zc, area, normx, normy, normz, s_name, s_name[N]

  .. parsed-literal::

       id = surface element ID
       v1x,v1y,v1z = coords of first line end point or triangle corner point
       v3x,v2y,v2z = coords of second line end point or triangle corner point
       v3x,v3y,v3z = coords of third triangle corner point
       xc,yc,zc = coords of center of line segment or triangle
       area = length of line segment or area of triangle
       normx, normy, normz = unit normal vector for line segment or triangle
       s_name = custom per-surf floating-point vector with name
       s_name[N] = Nth column of custom per-surf floating-point array with name

Examples
--------

.. parsed-literal::

   compute 1 property/surf all id xc yc zc

Description
-----------

Define a computation that simply stores surface element attributes for
each explicit surface element in a surface group.  This is useful for
values which can be used by other :ref:`output commands <howto_8>` that take computes as inputs.
See for example, the :doc:`compute reduce <compute_reduce>`, :doc:`fix ave/surf <fix_ave_surf>`, :doc:`dump surf <dump>`, and :doc:`surf-style variable <variable>` commands.

Only surface elements in the surface group specified by *group-ID* are
included in the calculation.  See the :doc:`group surf <group>` command
for info on how surface elements can be assigned to surface groups.

This command can only be used for simulations with explicit surface
elements.  Explicit surface elements are triangles for 3d simulations
and line segments for 2d simulations.  Unlike implicit surface
elements, each explicit triangle or line segment may span multiple
grid cells.  See :ref:`Section 4.9 <howto_11>` of the
manual for details.

----------

*Id* is the surface element ID, as defined in the surface data file
read by the :doc:`read_surf <read_surf>` comand.

The *v1x*, *v1y*, *v1z* attributes are the coordinates of the first
end point of a line segment (2d) or first corner point of a triangle
(3d).  Likewise, the *v2x*, *v2y*, *v2z* attributes are the
coordinates of the second end point of a line segment (2d) or second
corner point of a triangle (3d).  The *v3x*, *v3y*, *v23z* attributes
are the coordinates of the third corner point of a triangle (3d).

The *xc*, *yc*, *zc* attributes are the coordinates of the center
point of a line segment or tringle.

The *area* attribute is the length of a line segment (distance units
in 2d), or area of a triangle (area units in 3d).

The *normx*, *normy*, *normz* attributes are components of a unit
normal perpendicular to the line segment or face of the trangle. It
points into the flow volume of the simulation.

----------

Output info
-----------

This compute calculates a per-surf vector or per-surf array depending
on the number of input values.  If a single input is specified, a
per-surf vector is produced.  If two or more inputs are specified, a
per-surf array is produced where the number of columns = the number of
inputs.

This compute performs calculations for each explicit surface element
in the simulation.

Surface elements not in the specified *group-ID* will output zeroes
for all their values.

The vector or array can be accessed by any command that uses per-surf
values from a compute as input.  See :ref:`Section 4.4 <howto_8>` for an overview of OpenEdge output
options.

The vector or array values will be in whatever :doc:`units <units>` the
corresponding attribute is in, e.g. distance units for *v1x* or *xc*,
length units for *area* in 2d, area units for *area* in 3d.

----------

Restrictions
------------

For 2d simulations, none of the attributes which refer to the 3rd
dimension may be used.  Likewise *v3x*, *v3y*, *v3z* may not be used
since they refer to triangles.

Related commands
----------------

:doc:`dump surf <dump>`, :doc:`fix ave/surf <fix_ave_surf>`

**Default:** none
