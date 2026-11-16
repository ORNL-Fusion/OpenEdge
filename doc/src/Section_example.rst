.. hand-edited: this page is maintained here, not generated from doc/*.txt

5. Example problems
===================

The examples directory of the distribution holds verification cases,
workflow cases and one visualization case.  Each case has its own
directory with one or more input scripts named in.\*.  Most cases keep
their input data in an input sub-directory and their checking and
plotting scripts in a scripts sub-directory, and write to an output
sub-directory.  The README.md file of a case describes the physics,
the commands to run it and the acceptance criteria.

The paths below are relative to the examples directory.

Verification cases
------------------

A verification case compares OpenEdge with an analytic solution, a
gyro-resolved or otherwise converged reference run, or the published
equations of another code.  The cases are small.  Most have a run.sh
script that runs the case, applies the checks and prints PASS or FAIL.
Run the cases that cover a command after changing its source.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Directory
     - Content
   * - verification/collisions/coulomb
     - slowing-down of C\ :sup:`3+` on a deuterium background with
       :doc:`fix coulomb/background <fix_coulomb_background>` and
       thermalization of two species with
       :doc:`fix coulomb/binary <fix_coulomb_binary>`, against the
       temperature-equilibration rate of the NRL formulary
   * - verification/d2_chemistry
     - dissociation of D\ :sub:`2`, ionization of D and charge
       exchange of D with D\ :sup:`+` in a uniform plasma
   * - verification/efield_polarization
     - polarization drift of a proton in a uniform magnetic field and
       an oscillating electric field
   * - verification/ionization_recombination
     - relaxation of oxygen to its charge-state equilibrium under the
       ADAS rates of
       :doc:`fix volume/chem/adas <fix_volume_chem_adas>`
   * - verification/kokkos_parity
     - particle-by-particle and cell-by-cell comparison of the CPU
       and Kokkos versions of the OpenEdge styles, started from one
       restart file
   * - verification/particulates/dustt2005/uniform_plasma_benchmark
     - charging, drag and motion of a grain in a uniform plasma with
       the DUSTT-2005 model
   * - verification/particulates/dustt2005/cat_solps_droplet_transport
     - lithium droplets launched from the outer divertor of the CAT
       geometry in a SOLPS-ITER background
   * - verification/particulates/dis2021/uniform_plasma_comparison
     - one boron grain in a uniform plasma with the DUSTT-2005 and the
       DIS-2021 models; each is checked against its own equations
   * - verification/particulates/psi_reflect_inside
     - :doc:`fix reflect/psi <fix_reflect_psi>` with particles that
       are created inside the reflecting flux surface
   * - verification/pushers/orbit
     - orbit of a proton in the analytic tokamak field of Khan et al.;
       the guiding-centre pusher is compared with the Boris pusher in
       3d, 2d and axisymmetric geometry
   * - verification/pushers/hybrid
     - hybrid pusher near an absorbing wall: switch from
       guiding-centre to Boris, sheath barrier, coupling to the
       thermal force, cross-field diffusion and Coulomb collisions,
       and an axisymmetric ring target
   * - verification/surface_emission/constant_flux
     - number of particles emitted at constant flux for several
       values of the emission interval
   * - verification/surface_pwi/deposit_tagging
     - the *deposit_as* and initial-layer keywords of
       :doc:`surf_react surface/pwi <surf_react_surface_pwi>` with a
       tungsten beam on a tungsten plate
   * - verification/surface_pwi/emit_conc_scale
     - :doc:`fix surface/emit/source <fix_surface_emit_source>` with a
       source strength that follows the surface concentration of an
       eroding boron layer
   * - verification/surface_pwi/ledger_fold
     - areal densities of
       :doc:`surf_react surface/pwi <surf_react_surface_pwi>` at dump
       and restart times for two values of *adens_nevery* and two
       processor counts
   * - verification/surface_pwi/strata_buried_erosion
     - erosion of a boron layer that lies under a tagged tungsten
       deposit
   * - verification/surface_pwi/transit_tag
     - :doc:`fix particle/tag <fix_particle_tag>`: wall impacts of
       particles that crossed a marked surface are tallied separately
   * - verification/thermal_force_coefficient
     - one-step velocity change from the ion thermal force of
       :doc:`fix force/thermal <fix_force_thermal>` for boron and
       carbon ions, against the DIVIMP coefficient
   * - verification/wedge_periodic_caps
     - conservation of particles that cross the toroidal caps of a
       10 degree wedge with *surf_collide toroidal* on several
       processors

A verification case is run from its directory:

.. code-block:: text

   cd examples/verification/pushers/orbit
   ./run.sh /path/to/spa_mpi

Workflow cases
--------------

A workflow case is a device-scale simulation that goes from a
converted plasma background to impurity or particulate transport and
the response of the wall.  The cases are meant as templates for new
simulations.  The README of each case gives the run command.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Directory
     - Content
   * - workflows/impurity_transport/west_tungsten_transport
     - axisymmetric tungsten erosion and transport in WEST on a
       SOLEDGE3X background: sputtering by oxygen, Boris pusher with
       a spatially resolved sheath, Coulomb collisions, thermal
       forces, cross-field diffusion, ADAS ionization and a
       plasma-wall interaction model at the wall
   * - workflows/impurity_transport/rfpie_tungsten_transport
     - three-dimensional tungsten sputtering by a helium plasma on a
       biased target in the RFPIE geometry, with measured plasma
       profiles as the background
   * - workflows/neutral_transport/ltx_lithium_neutral_sources
     - lithium and hydrogen neutral sources in LTX-beta on a SOLEDGE3X
       background: particle, momentum and energy source moments on an
       R-Z grid
   * - workflows/particulates/cat_liquid_metal_divertor
     - lithium release from a liquid-metal divertor in SOLPS-ITER
       backgrounds by evaporation, adatom desorption and sputtering,
       and transport of lithium droplets
   * - workflows/particulates/st40_lithium_powder_dropper
     - lithium powder injection in ST40 on a SOLPS-ITER background,
       with an axisymmetric and a full-torus input script and the
       scripts that couple the lithium sources back to SOLPS-ITER
   * - workflows/particulates/west_boron_powder_dropper
     - boron powder injection in WEST: grain heating and sublimation
       and the transport of the released boron

A workflow case is run with its in.openedge script:

.. code-block:: text

   cd examples/workflows/impurity_transport/west_tungsten_transport
   mpirun -np 4 /path/to/spa_mpi -in in.openedge

Visualization case
------------------

The visualization/paraview directory holds a small flow around a
circular obstacle that writes the grid, the surface and the particles
as VTK files.  It needs an executable built with the VTK package; see
:ref:`Section 4.13 <howto_13>`.

Running all cases
-----------------

The regression script at the top of the distribution runs every
registered case for a limited number of steps and reports which ones
pass:

.. code-block:: text

   ./regression/run_regression.sh --exe /path/to/spa_mpi

:ref:`Section 2.3 <start_3>` lists its options.
