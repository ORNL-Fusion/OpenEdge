# Domain, species, physical surfaces, and background plasma.

seed        12345
dimension   2
boundary    o ao p
global      gridcut 0.0 comm/sort no
global      fnum 1.0e14
timestep    ${dt}

create_box  -1.0 1.0  0.0 1.05  -0.05 0.05
create_grid 130 72 1
balance_grid rcb cell
global      weight cell radius

species     input/grains.species droplet
species     input/atoms.species Li Li+ Li2+ Li3+
mixture     droplet droplet frac 1.0
mixture     livapor Li frac 1.0
mixture     allLi Li Li+ Li2+ Li3+ group SELF

read_surf   input/st40_wall_axi.surf group wall particle check
read_surf   input/st40_core.surf group core
group       dropper surf id <> 16 17

surf_collide wallc diffuse 300.0 1.0
surf_react   liwall surface/pwi input/li.recycle twall 300 &
             adens_surf adens rzone 1e19
surf_modify  wall collide wallc react liwall

# Numerical sink at the inner SOLPS boundary. The transparent-core study uses
# a separate setup include and never treats this contour as a physical wall.
surf_collide corec diffuse 300.0 1.0
surf_react   licore surface/pwi input/li.recycle twall 300 &
             adens_surf adensc rzone 1e19
surf_modify  core collide corec react licore

adapt_grid   all refine surf wall 0.001 maxlevel 3 cells 2 2 1
balance_grid rcb cell

# This tracked file preserves every native, orientation-verified SOLPS/EFIT
# run_dir dataset and adds a labelled regular-grid continuation across the
# central mesh hole.  The physical core surface below prevents the canonical
# core-loss control from sampling that continuation.
fix         pd background file input/plasma_st40_solps_corefill.h5 static yes
global      phi_track yes
