.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: compute nearest_surf/grid

compute nearest_surf/grid command
=================================

Syntax
------

.. parsed-literal::

   compute ID nearest_surf/grid group-ID sgroup-ID value1 value2 ...

* ID is documented in the :doc:`compute <compute>` command
* nearest_surf/grid = style name of this compute command
* group-ID = group ID for which grid cells to perform calculation on
* sgroup-ID = surface group ID of the surface elements to search
* two or more values must be appended
* value = *dist* or *surfid* or *surfidx* or *nx* or *ny* or *nz*

  * *dist* = distance from the cell centre to the nearest point of the
    nearest surface element (m)
  * *surfid* = surface ID of that element
  * *surfidx* = index of that element in the surface list of the
    processor
  * *nx*, *ny*, *nz* = components of the unit normal of that element,
    oriented towards the cell centre

Examples
--------

.. code-block:: text

   compute cgeom nearest_surf/grid all wall dist surfid nx ny nz
   global pusher mode boris plasma pd sheath boundary geom cgeom

   compute cnear nearest_surf/grid all divertor dist surfid
   dump dgrid grid all 1000 nearest.txt id c_cnear[1] c_cnear[2]

Description
-----------

Find, for every grid cell of the group, the nearest surface element of
the surface group, and store its distance, identity and normal.  The
elements are line segments in 2d and triangles in 3d.  The sheath
model and the *boris_near* and *gc_wall* options of the *pusher*
keyword of the :doc:`global <global>` command take the wall geometry
from a compute of this style, named by their *geom* value.  The
values can also be dumped as a wall-distance map.

For a cell with centre :math:`\mathbf{x}_c`, the distance to a surface
element :math:`s` is

.. math::
   :label: nearest-dist

   d_s = \min_{\mathbf{y} \in s} \left| \mathbf{x}_c - \mathbf{y} \right| ,

where :math:`\mathbf{y}` runs over the points of the finite segment or
triangle.  The closest point can lie in the interior of the element,
on an edge or at a vertex.  The compute selects the element of the
surface group with the smallest :math:`d_s`.  The distance to the
infinite line or plane through an element is not used, because it
would select a short element that lies far beyond its end points.

The *dist* value is :math:`d_s` of the selected element.  The *surfid*
value is its ID as assigned when the surface was read.  The *surfidx*
value is its index in the surface list of the processor.  The *nx*,
*ny* and *nz* values are the components of its unit normal
:math:`\hat{\mathbf{n}}`, with the sign chosen such that

.. math::

   \hat{\mathbf{n}} \cdot (\mathbf{x}_c - \mathbf{x}_s) \ge 0 ,

where :math:`\mathbf{x}_s` is the midpoint of the segment or the
centroid of the triangle.  The normal therefore points from the wall
towards the cell.  In 2d *nz* is zero.

The search runs once, when the values are first requested.  It tests
every element of the surface group for every cell.  The results are
kept in per-cell custom grid attributes that move with the cells, so a
load balance reuses them on the new processor, provided the surface
elements are not distributed.  Cells created later, for example by
:doc:`fix adapt <fix_adapt>` or :doc:`adapt_grid <adapt_grid>`, are
searched when the grid changes.

Output
^^^^^^

The compute produces a per-grid array with one column per value, in
the order the values were given.  It can be written by the
:doc:`dump grid <dump>` command.

Cells that are not in the grid group hold zeros in all columns, and so
do the sub-cells of a split cell.  The split cell itself holds the
result, and the pusher looks the values up there for a particle in one
of its sub-cells.  A cell for which the surface group has no element
holds *dist* = 1.0e20, *surfid* = -1, *surfidx* = -1 and a zero
normal.

Restrictions
------------

This command is part of the OPENEDGE package.

At least two values must be given.

The search indexes the complete list of surface elements on every
processor.  The surface elements must therefore be explicit and not
distributed, which is the default *surfs explicit* setting of the
:doc:`global <global>` command.  The command does not test this.  If
no surface elements are defined, every cell holds the values of a cell
without an element.

Related commands
----------------

:doc:`global <global>`, :doc:`group <group>`,
:doc:`compute grid <compute_grid>`

Default
-------

none
