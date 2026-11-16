.. index:: remove_surf

remove_surf command
===================

Syntax
------

.. parsed-literal::

   remove_surf surfID

* surfID = group ID for which surface elements to remove

Examples
--------

.. parsed-literal::

   remove_surf topsurf

Description
-----------

Remove a group of surface elements that have previously been read-in
via the :doc:`read_surf <read_surf>` command.  The :doc:`group surf <group>` or :doc:`read_surf <read_surf>` can be used to assign
each surface element to one or more groups.  This command removes all
surface elements in the specified *surfID* group.

Note that the remaining surface elements must still constitute a
"watertight" surface or an error will be generated.  The definition of
watertight is explained in the Restrictions section of the
:doc:`read_surf <read_surf>` doc page.

After surface elements have been deleted the IDs of the remaining
surface points and elements are renumbered so that the remaining N
elements have IDs from 1 to N.  The new list of surface elements can
be output via the :doc:`write_surf <write_surf>` or :doc:`dump surf <dump>` commands.

**Restrictions:** none

Related commands
----------------

:doc:`read_surf <read_surf>`

**Default:** none
