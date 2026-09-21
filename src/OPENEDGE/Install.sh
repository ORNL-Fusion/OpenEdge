#!/bin/sh
# Install/unInstall OpenEdge package files in SPARTA
# mode = 0/1/2 for uninstall/install/update

mode=$1

# Override files: these replace base SPARTA files.
# On install: back up originals to *.sparta_orig
# On uninstall: restore originals from *.sparta_orig

override () {
  if (test $mode = 0) then
    # Uninstall: restore original if backup exists, else just remove
    if (test -e ../$1.sparta_orig) then
      mv ../$1.sparta_orig ../$1
    else
      rm -f ../$1
    fi
  elif (! cmp -s $1 ../$1) then
    # Install/update: back up original (if not already backed up), then copy
    if (test -e ../$1 && test ! -e ../$1.sparta_orig) then
      cp ../$1 ../$1.sparta_orig
    fi
    cp $1 ..
    if (test $mode = 2) then
      echo "  updating src/$1"
    fi
  fi
}

# New files: no backup needed, just add/remove

action () {
  if (test $mode = 0) then
    rm -f ../$1
  elif (! cmp -s $1 ../$1) then
    cp $1 ..
    if (test $mode = 2) then
      echo "  updating src/$1"
    fi
  fi
}

# --- Override files (exist in base SPARTA) ---
override update.cpp
override update.h
override input.cpp
override input.h
override particle.cpp
override particle.h
override variable.cpp
override variable.h
override compute_grid.cpp
override compute_grid.h
override dump_particle.cpp
override dump_particle.h
override sparta.cpp
override sparta.h
override surf_collide_diffuse.cpp
override surf_collide_diffuse.h

# --- New OpenEdge files ---
action background_zones3d.cpp
action background_zones3d.h
action compute_grid_weighted.cpp
action compute_grid_weighted.h
action compute_impact_energy.cpp
action compute_impact_energy.h
action compute_nearest_surf_grid.cpp
action compute_nearest_surf_grid.h
action compute_plasma_fields.cpp
action compute_plasma_fields.h
action compute_surface_chemical_adatom.cpp
action compute_surface_chemical_adatom.h
action compute_surface_chemical_evaporation.cpp
action compute_surface_chemical_evaporation.h
action compute_surface_physical_sputter.cpp
action compute_surface_physical_sputter.h
action compute_surf_weighted.cpp
action compute_surf_weighted.h
action compute_volume_emissivity_grid.cpp
action compute_volume_emissivity_grid.h
action database_paths.cpp
action database_paths.h
action eckstein_sputter_data.h
action eckstein_sputter.h
action fix_background.cpp
action fix_background.h
action fix_bfield_particle.cpp
action fix_bfield_particle.h
action fix_coulomb_background.cpp
action fix_coulomb_background.h
action fix_coulomb_base.cpp
action fix_coulomb_base.h
action fix_coulomb_binary.cpp
action fix_coulomb_binary.h
action fix_cross_field_diffusion.cpp
action fix_cross_field_diffusion.h
action fix_efield_particle.cpp
action fix_efield_particle.h
action fix_force_gravity.cpp
action fix_force_gravity.h
action fix_force_thermal.cpp
action fix_force_thermal.h
action fix_particle_weight.cpp
action fix_particle_weight.h
action fix_population_control.cpp
action fix_population_control.h
action fix_reflect_psi.cpp
action fix_reflect_psi.h
action fix_surface_emit_puff.cpp
action fix_surface_emit_puff.h
action fix_surface_emit_recycle.cpp
action fix_surface_emit_recycle.h
action fix_surface_emit_source.cpp
action fix_surface_emit_source.h
action fix_surface_state_lm.cpp
action fix_surface_state_lm.h
action fix_volume_chem_adas.cpp
action fix_volume_chem_adas.h
action grid_src.h
action iead_table.cpp
action iead_table.h
action liquid_metal_strip.cpp
action liquid_metal_strip.h
action openedge_geom.h
action particulate_model_kernels.cpp
action particulate_model_kernels.h
action process_library.cpp
action process_library.h
action pusher.cpp
action pusher.h
action reflection_tables.h
action sheath_models.cpp
action sheath_models.h
action surface_incidence.cpp
action surface_incidence.h
action surf_collide_partial_recycle.cpp
action surf_collide_partial_recycle.h
action surf_react_mpex.cpp
action surf_react_mpex.h
action surf_react_surface_pwi.cpp
action surf_react_surface_pwi.h
action surf_state_multilayer.cpp
action surf_state_multilayer.h
