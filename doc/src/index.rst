.. hand-edited: this page is maintained here, not generated from doc/*.txt

OpenEdge documentation
======================

OpenEdge is a kinetic Monte Carlo code for impurity transport and
plasma-material interaction in the edge of magnetic fusion devices.
It follows neutrals, impurity ions, and dust grains or droplets
through a plasma background taken from an edge code such as
SOLPS-ITER, SOLEDGE3X, OEDGE or EMC3-EIRENE, and it evolves the wall
surface that these particles erode and coat.

OpenEdge is built on the `SPARTA <https://sparta.github.io>`_ Direct
Simulation Monte Carlo code and keeps its input-script language, grid,
surface handling and parallelism.  The source is on
`GitHub <https://github.com/ORNL-Fusion/OpenEdge>`_.  OpenEdge is
developed by Abdourahmane (Abdou) Diaw at Oak Ridge National
Laboratory; questions and bug reports can go to diawa@ornl.gov or
to the GitHub issue tracker.

New users should read :doc:`Section_intro` and :doc:`Section_start`,
then run one of the cases in :doc:`Section_example`.  The physics
models are documented on the pages of the individual commands, which
are listed by category in :ref:`Section 3.4 <cmd_4>`.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   Section_intro
   Section_start
   Section_commands
   Section_howto
   Section_example
   Section_errors

.. toctree::
   :maxdepth: 1
   :caption: Reference

   commands
   notation
   references
