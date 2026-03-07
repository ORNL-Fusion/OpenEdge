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
override fix_emit_face_file.cpp
override fix_emit_face_file.h
override fix_emit_surf.cpp
override fix_emit_surf.h
override fix_field_grid.cpp
override fix_field_grid.h
override fix_field_particle.cpp
override fix_field_particle.h
override surf_collide_diffuse.cpp
override surf_collide_diffuse.h

# --- New OpenEdge files ---
action boris_grid.h
action nanbu_scatter_table.h
action sheath_models.cpp
action sheath_models.h
action compute_iead.cpp
action compute_iead.h
action compute_incident_plasma_flux.cpp
action compute_incident_plasma_flux.h
action compute_bfield_timedep.cpp
action compute_bfield_timedep.h
action compute_plasma_fields.cpp
action compute_plasma_fields.h
action compute_pmi_surf_data.cpp
action compute_pmi_surf_data.h
action compute_sheath_geometry_grid.cpp
action compute_sheath_geometry_grid.h
action compute_surf_ead.cpp
action compute_surf_ead.h
action compute_thermal_sheath_grid.cpp
action compute_thermal_sheath_grid.h
action fix_bfield_grid.cpp
action fix_bfield_grid.h
action fix_bfield_particle.cpp
action fix_bfield_particle.h
action fix_chem_adas.cpp
action fix_chem_adas.h
action fix_coll_background.cpp
action fix_coll_background.h
action fix_coll_nanbu.cpp
action fix_coll_nanbu.h
action fix_drag.cpp
action fix_drag.h
action fix_efield_grid.cpp
action fix_efield_grid.h
action fix_efield_particle.cpp
action fix_efield_particle.h
action fix_emit_droplet.cpp
action fix_emit_droplet.h
action fix_emit_surf_file.cpp
action fix_emit_surf_file.h
action fix_emit_surf_pmi.cpp
action fix_emit_surf_pmi.h
action fix_evaporation.cpp
action fix_evaporation.h
action fix_gravity.cpp
action fix_gravity.h
action fix_thermal_force_e.cpp
action fix_thermal_force_e.h
action fix_thermal_force_i.cpp
action fix_thermal_force_i.h
action fix_viscous.cpp
action fix_viscous.h
action surf_react_mpex.cpp
action surf_react_mpex.h
action surf_react_pmi.cpp
action surf_react_pmi.h
