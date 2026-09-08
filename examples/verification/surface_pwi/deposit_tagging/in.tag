# surf_react surface/pwi deposit_as test: a hot W beam deposits on a W plate
# while W self-sputtering erodes it.  tag=1 credits the retained W to the
# deposit material Wd and debits erosion from whatever is exposed.

variable    case     index untagged
variable    tag      index 0
variable    recycle  index input/w_plain.recycle
variable    nsteps   index 2000
variable    nlaunch  index 200

shell       mkdir output
seed        12345
dimension   3
boundary    o o o
global      gridcut 0.0 comm/sort yes
global      fnum 1e14
timestep    1e-6

create_box  0 1 0 1 0 1
create_grid 4 4 4
balance_grid rcb cell

species     input/w.species W Wd
mixture     wbeam W frac 1.0 nrho 1.0e19 temp 3.0e6 vstream 0.0 0.0 0.0

read_surf   input/plate.surf group plate particle check
read_surf   input/source.surf group source particle check
surf_collide vac vanish
surf_collide wall diffuse 773.15 1.0
if "${tag} == 1" then &
  "surf_react pwi surface/pwi ${recycle} twall 773.15 adens_surf adens rzone 1e17 strata 12 minthick 0.05 deposit_as W Wd" &
else &
  "surf_react pwi surface/pwi ${recycle} twall 773.15 adens_surf adens rzone 1e17 strata 12 minthick 0.05"
surf_modify plate collide wall react pwi
surf_modify source collide vac

fix         pw particle/weight
fix         femit surface/emit/source wbeam source constant 1.0e22 perspecies no &
            normal no nlaunch_total ${nlaunch} nevery 1 model thermal

run         0 post no
dump        dsig surf plate ${nsteps} output/${case}.*.dump id s_adens[*] &
            s_adens_net s_adens_dep s_adens_ero s_adens_conc[*] s_adens_strata[*]
stats       500
stats_style step cpu np nscoll nsreact
run         ${nsteps}
