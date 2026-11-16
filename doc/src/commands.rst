.. hand-edited: this page is maintained here, not generated from doc/*.txt

Command reference
=================

Every input-script command has its own page.  The commands that
OpenEdge adds to SPARTA are grouped by topic below, followed by the
SPARTA commands in alphabetical order.  The ion pusher and the sheath
model are settings of the :doc:`global <global>` command.
:ref:`Section 3.5 <cmd_5>` lists all commands and styles in tables.

.. toctree::
   :maxdepth: 1
   :caption: Plasma background and fields

   fix_background
   compute_plasma_fields
   fix_bfield_particle
   fix_efield_particle

.. toctree::
   :maxdepth: 1
   :caption: Ion transport and collisions

   fix_coulomb_background
   fix_coulomb_binary
   fix_force_thermal
   fix_cross_field_diffusion
   fix_core_transport
   fix_reflect_psi
   fix_neutral_collide_background
   fix_store_force

.. toctree::
   :maxdepth: 1
   :caption: Atomic processes

   fix_volume_chem_adas
   compute_volume_emissivity_grid

.. toctree::
   :maxdepth: 1
   :caption: Plasma-wall interaction

   surf_react_surface_pwi
   compute_surface_physical_sputter
   fix_surface_emit_source
   fix_surface_emit_recycle
   fix_surface_emit_puff
   fix_surface_state_lm
   compute_surface_chemical_adatom
   compute_surface_chemical_evaporation

.. toctree::
   :maxdepth: 1
   :caption: Particulates

   material
   fix_particulate_emit
   fix_particulate_charge
   fix_particulate_drag
   fix_particulate_thermal
   surf_collide_particulate_bounce

.. toctree::
   :maxdepth: 1
   :caption: Weighting and diagnostics

   fix_particle_weight
   fix_particle_tag
   fix_dt_monitor
   compute_grid_weighted
   compute_nearest_surf_grid

.. toctree::
   :maxdepth: 1
   :caption: SPARTA commands

   adapt_grid
   balance_grid
   bound_modify
   boundary
   clear
   collide_modify
   collide
   compute_boundary
   compute_count
   compute_distsurf_grid
   compute_dt_grid
   compute_eflux_grid
   compute_grid
   compute_isurf_grid
   compute_ke_particle
   compute_lambda_grid
   compute_pflux_grid
   compute_property_grid
   compute_property_surf
   compute_react_boundary
   compute_react_isurf_grid
   compute_react_surf
   compute_reduce
   compute_sonine_grid
   compute_surf
   compute_temp
   compute_thermal_grid
   compute_tvib_grid
   compute
   create_box
   create_grid
   create_isurf
   create_particles
   custom
   dimension
   dump_image
   dump_modify
   dump
   echo
   fix_ablate
   fix_adapt
   fix_ambipolar
   fix_ave_grid
   fix_ave_histo
   fix_ave_surf
   fix_ave_time
   fix_balance
   fix_dt_reset
   fix_emit_face
   fix_emit_surf
   fix_grid_check
   fix_halt
   fix_move_surf
   fix_print
   fix_surf_temp
   fix_temp_global_rescale
   fix_temp_rescale
   fix_vibmode
   fix
   global
   group
   if
   include
   jump
   label
   log
   mixture
   move_surf
   next
   package
   partition
   print
   quit
   react_modify
   react
   read_grid
   read_isurf
   read_particles
   read_restart
   read_surf
   region
   remove_surf
   reset_timestep
   restart
   run
   scale_particles
   seed
   shell
   species_modify
   species
   stats_modify
   stats_style
   stats
   suffix
   surf_collide
   surf_modify
   surf_react_adsorb
   surf_react
   timestep
   uncompute
   undump
   unfix
   units
   variable
   write_grid
   write_isurf
   write_restart
   write_surf
