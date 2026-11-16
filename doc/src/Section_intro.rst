.. hand-edited: this page is maintained here, not generated from doc/*.txt

1. Introduction
===============

This chapter says what OpenEdge is, lists its features with links to
the commands that provide them, and gives the license and the
references to cite.

| 1.1 :ref:`What is OpenEdge <intro_1>`
| 1.2 :ref:`OpenEdge features <intro_2>`
| 1.3 :ref:`Grids, surfaces and plasma backgrounds <intro_3>`
| 1.4 :ref:`Open source distribution <intro_4>`
| 1.5 :ref:`Acknowledgments and citations <intro_5>`

----------

.. _intro_1:

1.1 What is OpenEdge
--------------------

OpenEdge is a kinetic Monte Carlo code for the edge of magnetically
confined plasmas and for plasma-material interaction.  It follows
test particles through a plasma and magnetic-field background that is
prescribed, not solved for.  The test particles are neutral atoms and
molecules, impurity ions, and solid or liquid particulates.  The
background comes from an edge code such as SOLPS-ITER, SOLEDGE3X,
OEDGE or EMC3-EIRENE, or from constant or analytic fields in
verification cases.

Ions are advanced with the Boris scheme [Boris1970]_ or, where the
Larmor radius is small compared with the scale length of the magnetic
field, with a guiding-centre integrator.  A hybrid mode switches
between the two for each particle.  Ionization, recombination and
charge exchange use ADAS rate coefficients.  Ions also experience
Coulomb collisions with the background, the thermal forces, anomalous
cross-field diffusion and the electric field of the wall sheath.  A
particle that reaches the wall can be reflected or implanted, and it
can sputter wall atoms that are then followed in turn.  The surface
composition of each wall element changes as material is eroded and
deposited.  Dust grains and droplets carry a radius, mass, charge and
temperature, and the plasma charges, drags, heats and evaporates
them.

OpenEdge is built on `SPARTA <https://sparta.github.io>`_, the Direct
Simulation Monte Carlo code from Sandia National Laboratories
[Plimpton2019]_.  It takes from SPARTA the hierarchical Cartesian
grid, the cut-cell treatment of surfaces, the particle data
structures, the MPI parallelization and the Kokkos layer.  Every
SPARTA input command is still available.  This manual documents the
SPARTA commands together with the commands that OpenEdge adds.

OpenEdge needs a C++17 compiler and an `MPI <https://www.mpi-forum.org>`_
library.  It runs on anything from a laptop to a parallel cluster, and
a Kokkos build runs on multi-core CPUs and on NVIDIA GPUs.

----------

.. _intro_2:

1.2 OpenEdge features
---------------------

The lists below cover the commands that OpenEdge adds to SPARTA.  The
grid, surface, output and load-balancing commands that it inherits are
described on their own pages.

Plasma backgrounds
^^^^^^^^^^^^^^^^^^

* plasma density, temperature, parallel flow and magnetic equilibrium
  read from a converted plasma file,
  :doc:`fix background <fix_background>`
* two-dimensional axisymmetric meshes and three-dimensional
  EMC3-EIRENE zones
* constant and analytic backgrounds for verification
* plasma quantities, magnetic field and gradients sampled at each
  particle, :doc:`compute plasma/fields <compute_plasma_fields>`

Ion transport
^^^^^^^^^^^^^

* Boris pusher with subcycling, :doc:`global pusher <global>`
* guiding-centre pusher with second- or fourth-order Runge-Kutta
  integration, and a hybrid mode that returns to Boris near walls
* wall sheath as an energy gain at impact, a spatially resolved
  potential, or a boundary barrier
* Coulomb collisions with the background plasma,
  :doc:`fix coulomb/background <fix_coulomb_background>`, and between
  simulation particles, :doc:`fix coulomb/binary <fix_coulomb_binary>`
* ion and electron thermal forces,
  :doc:`fix force/thermal <fix_force_thermal>`
* anomalous cross-field diffusion and pinch,
  :doc:`fix cross_field_diffusion <fix_cross_field_diffusion>`
* user-supplied electric and magnetic fields per particle,
  :doc:`fix efield/particle <fix_efield_particle>` and
  :doc:`fix bfield/particle <fix_bfield_particle>`
* absorption or reflection at a flux surface,
  :doc:`fix reflect/psi <fix_reflect_psi>`

Atomic processes
^^^^^^^^^^^^^^^^

* ADAS ionization, recombination, charge exchange and dissociation,
  with one channel chosen among those that compete,
  :doc:`fix volume/chem/adas <fix_volume_chem_adas>`
* collisions of neutrals with the background plasma,
  :doc:`fix neutral/collide/background <fix_neutral_collide_background>`
* line emission from photon emissivity coefficients,
  :doc:`compute volume/emissivity/grid <compute_volume_emissivity_grid>`

Plasma-wall interaction
^^^^^^^^^^^^^^^^^^^^^^^

* incident flux, impact energy and angle, sputter yield and gross
  erosion per wall element, from tabulated or analytic yields,
  :doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`
* reflection, implantation, re-emission and sputtering, with the
  areal density of each species tracked per wall element and optional
  layering (strata), :doc:`surf_react surface/pwi <surf_react_surface_pwi>`
* neutral sources from recycling, sputtering and gas puffs,
  :doc:`fix surface/emit/recycle <fix_surface_emit_recycle>`,
  :doc:`fix surface/emit/source <fix_surface_emit_source>` and
  :doc:`fix surface/emit/puff <fix_surface_emit_puff>`
* liquid-metal surfaces with evaporation and adatom desorption,
  :doc:`fix surface/state/lm <fix_surface_state_lm>`

Particulates
^^^^^^^^^^^^

* dust grains and droplets with their own radius, mass and
  temperature, :doc:`material <material>`
* injection from surfaces,
  :doc:`fix particulate/emit <fix_particulate_emit>`
* orbital-motion-limited charging with thermionic and secondary
  electron emission,
  :doc:`fix particulate/charge <fix_particulate_charge>`
* ion drag, neutral drag, electric force and gravity,
  :doc:`fix particulate/drag <fix_particulate_drag>`
* heating, melting, evaporation and breakup,
  :doc:`fix particulate/thermal <fix_particulate_thermal>`
* sticking or bouncing at walls,
  :doc:`surf_collide particulate/bounce <surf_collide_particulate_bounce>`

Diagnostics and weighting
^^^^^^^^^^^^^^^^^^^^^^^^^

* statistical weights per particle and weighted grid moments,
  :doc:`fix particle/weight <fix_particle_weight>` and
  :doc:`compute grid/weighted <compute_grid_weighted>`
* distance to the nearest wall element for each grid cell, and its
  index, :doc:`compute nearest_surf/grid <compute_nearest_surf_grid>`
* all SPARTA dump, statistics and averaging commands

----------

.. _intro_3:

1.3 Grids, surfaces and plasma backgrounds
------------------------------------------

Particles are stored and sorted on the SPARTA hierarchical Cartesian
grid, in three dimensions, two dimensions or two-dimensional
axisymmetric geometry.  The vessel wall and internal components are
surfaces made of triangles, or of line segments in two dimensions,
that cut through the grid cells.  When a particle hits a surface
element, the surface collision and reaction models attached to that
element decide what happens to it.

The plasma background does not live on the particle grid.  It stays
on the mesh of the code that produced it: the quadrilateral B2 mesh
and its EIRENE triangulation for SOLPS-ITER, a triangular mesh for
SOLEDGE3X, curvilinear zones for EMC3-EIRENE.  Each particle samples
the background at its own position through
:doc:`fix background <fix_background>`.  The particle grid can
therefore be refined or coarsened to suit the transport problem
without resampling the plasma solution.

The magnetic field is computed either from the poloidal flux map of
the equilibrium, which also gives its derivatives, or from field
components stored with the background.  Atomic rate coefficients and
surface-interaction tables are read from a separate database file,
:ref:`processes.h5 <howto_6>`.

----------

.. _intro_4:

1.4 Open source distribution
----------------------------

OpenEdge is distributed under the GNU General Public License version
2, the license of SPARTA.  The LICENSE and LICENSE.SPARTA files in the
distribution contain the full text and the attribution of the SPARTA
components.  The NOTICES file lists the third-party data and code
shipped with OpenEdge and their licenses.

Under the GPL you may use, modify and redistribute OpenEdge.  A
derivative work that you distribute must also be released under the
GPL.  Improvements are welcome as pull requests to the
`repository <https://github.com/ORNL-Fusion/OpenEdge>`_.

----------

.. _intro_5:

1.5 Acknowledgments and citations
---------------------------------

OpenEdge is developed at Oak Ridge National Laboratory by
Abdourahmane (Abdou) Diaw (diawa@ornl.gov).  It depends on SPARTA, written by Steve Plimpton, Michael Gallis and their
colleagues at Sandia National Laboratories, and this manual reuses
the SPARTA documentation for the commands that OpenEdge inherits.

Atomic data come from `OPEN-ADAS <https://open.adas.ac.uk>`_.
Sputter yield and reflection tables are computed with the
`RustBCA <https://github.com/lcpp-org/RustBCA>`_ binary collision
code.  Plasma backgrounds are produced with SOLPS-ITER, SOLEDGE3X,
OEDGE and EMC3-EIRENE.

If you use OpenEdge, please cite the code paper [Diaw2025]_, which
describes the impurity transport model.  Work that uses the
particulate models for dust grains and liquid-metal droplets should
also cite [Diaw2026]_.
