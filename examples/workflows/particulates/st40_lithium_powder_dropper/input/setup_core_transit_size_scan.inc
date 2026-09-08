# Core-transit setup for the deterministic multi-diameter response scan.

seed        12345
dimension   2
boundary    o ao p
global      gridcut 0.0 comm/sort no
global      fnum 1.0e14
timestep    ${dt}

create_box  -1.0 1.0  0.0 1.05  -0.05 0.05
create_grid 130 72 1
balance_grid rcb cell

# Preserve one ID for each diameter/launch tuple. OpenEdge pweight remains the
# physical-inventory weight in later coupled-source runs.
global      weight cell none

species input/grains_size_scan.species &
        grain_d040um grain_d060um grain_d080um grain_d100um grain_d125um &
        grain_d150um grain_d200um grain_d250um grain_d300um
species input/atoms.species Li Li+ Li2+ Li3+
mixture droplet grain_d040um grain_d060um grain_d080um grain_d100um &
        grain_d125um grain_d150um grain_d200um grain_d250um grain_d300um
mixture livapor Li frac 1.0
mixture allLi Li Li+ Li2+ Li3+ group SELF

read_surf   input/st40_wall_axi.surf group wall particle check
group       dropper surf id <> 16 17

surf_collide wallc diffuse 300.0 1.0
surf_react   liwall surface/pwi input/li_size_scan.recycle twall 300 &
             adens_surf adens rzone 1e19
surf_modify  wall collide wallc react liwall

adapt_grid   all refine surf wall 0.001 maxlevel 3 cells 2 2 1
balance_grid rcb cell

fix         pd background file input/plasma_st40_solps_corefill.h5 static yes
fix         fcore reflect/psi background pd psi_norm 0.51 &
            action absorb mixture allLi

global      phi_track yes
