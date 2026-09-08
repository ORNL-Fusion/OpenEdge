# Core-transit domain, species, physical wall, and filled background.
#
# The numerical st40_core.surf contour is deliberately absent. Li atoms and
# ions are absorbed at the same flux contour by fix reflect/psi, while grains
# are outside mixture allLi and therefore cross it without interruption.

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
group       dropper surf id <> 16 17

surf_collide wallc diffuse 300.0 1.0
surf_react   liwall surface/pwi input/li.recycle twall 300 &
             adens_surf adens rzone 1e19
surf_modify  wall collide wallc react liwall

adapt_grid   all refine surf wall 0.001 maxlevel 3 cells 2 2 1
balance_grid rcb cell

# Native SOLPS cells retain priority. The labelled regular-grid extension is
# queried only in the central hole; it is a bounded numerical model, not a
# measured core profile.
fix         pd background file input/plasma_st40_solps_corefill.h5 static yes

# The verified old inner contour is psi_N = 0.5057 +/- 0.0007. A threshold of
# 0.51 places the atomic sink just inside the native mesh. Grains are not in
# allLi, so they continue into the filled core and keep ablating.
fix         fcore reflect/psi background pd psi_norm 0.51 &
            action absorb mixture allLi

global      phi_track yes
