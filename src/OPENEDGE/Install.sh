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
action pusher.h
action pusher.cpp
action sheath_models.cpp
action sheath_models.h
action database_paths.cpp
action database_paths.h
action process_library.cpp
action process_library.h
action compute_plasma_fields.cpp
action compute_plasma_fields.h
action compute_surface_physical_sputter.cpp
action compute_surface_physical_sputter.h
action iead_table.cpp
action iead_table.h
action compute_nearest_surf_grid.cpp
action compute_nearest_surf_grid.h
action fix_bfield_grid.cpp
action fix_bfield_grid.h
action fix_bfield_particle.cpp
action fix_bfield_particle.h
action fix_volume_chem_adas.cpp
action fix_volume_chem_adas.h
action fix_coll_background.cpp
action fix_coll_background.h
action fix_coulomb_base.cpp
action fix_coulomb_binary.cpp
action fix_coulomb_background.cpp
action fix_coulomb_base.h
action fix_coulomb_binary.h
action fix_coulomb_background.h
action fix_drag.cpp
action fix_drag.h
action fix_efield_grid.cpp
action fix_efield_grid.h
action fix_efield_particle.cpp
action fix_efield_particle.h
action fix_emit_droplet.cpp
action fix_emit_droplet.h
action fix_surface_emit_source.cpp
action fix_surface_emit_source.h
action fix_evaporation.cpp
action fix_evaporation.h
action fix_force_gravity.cpp
action fix_force_gravity.h
action fix_viscous.cpp
action fix_viscous.h
action surf_react_surface_pwi.cpp
action surf_react_surface_pwi.h
action fix_surface_emit_recycle.cpp
action fix_surface_emit_recycle.h
action fix_surface_emit_puff.cpp
action fix_surface_emit_puff.h
