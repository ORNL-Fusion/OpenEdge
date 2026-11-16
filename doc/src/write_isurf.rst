.. index:: write_isurf

write_isurf command
===================

Syntax
------

.. parsed-literal::

   write_isurf group-ID Nx Ny Nz filename ablateID keyword args ...

* group-ID = group ID for which grid cells store the implicit surfs

* Nx,Ny,Nz = grid cell extent of the grid cell group

* filename = name of file to write grid corner point info to

* ablateID = ID of the :doc:`fix ablate <fix_ablate>` command which stores the corner points

* zero or more keyword/args pairs may be appended

* keyword = *precision*

  .. parsed-literal::

       precision arg = int or double

Examples
--------

.. parsed-literal::

   write_isurf block 100 100 200 isurf.material.\* ablation

Description
-----------

Write a grid corner point file in binary format describing the current
corner point values which define the current set of implicit surface
elements.  See the :doc:`read_isurf <read_surf>` command for a
definition of implicit surface elements and how they are defined from
grid conner point values.  The surface file can be used for later
input to a new simulation or for post-processing and visualization.

The specified *group-ID* is the name of a grid cell group, as defined
by the :doc:`group grid <group>` command, which contains a set of grid
cells, all of which are the same size, and which comprise a contiguous
3d array, with specified extent *Nx* by *Ny* by *Nz*.  These should be
the same parameters that were used by the :doc:`read_isurf <read_isurf>`
command, when the original grid corner point values were read in and
used to define a set of implicit surface elements.  For 2d
simulations, *Nz* must be specified as 1, and the group must comprise
a 2d array of cells that is *Nx* by *Ny*.  These are the grid cells
that contain implicit surfaces.

Similar to :doc:`dump <dump>` files, the *filename* can contain a "\*"
wildcard character.  The "\*" character is replaced with the current
timestep value.  For example isurf.material.0 or
isurf.material.100000.

The specified *ablateID* is the fix ID of a :doc:`fix ablate <fix_ablate>` command which has been previously specified in
the input script for use with the :doc:`read_isurf <read_isurf>` command
and (optionally) to perform ablation during a simulation.  It stores
the grid corner point values for each grid cell.

The output file is written in the same binary format as the
:doc:`read_isurf <read_isurf>` command reads in.

**Restrictions:** none

Related commands
----------------

:doc:`read_isurf <read_isurf>`

Default
-------

The optional keyword default is precision double.
