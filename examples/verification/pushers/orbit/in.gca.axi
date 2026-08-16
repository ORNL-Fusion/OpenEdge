# Khan-orbit pusher verification — pure GCA, 2D axisymmetric (x=Z, y=R).
# Same field, launch point and dt as in.gca; compare against the 3D
# Boris reference (output/traj.boris) with plot_trajectories.py --mode axi.

seed                12345
dimension           2
boundary            o ao p
global              gridcut 0.0 comm/sort no
global              fnum 1

create_box          -2.0 2.0 0.0 2.0 -0.5 0.5
create_grid         10 10 1
balance_grid        rcb part

species             input/plasma.species H+
mixture             ions H+ frac 1.0

read_particles      input/source.axi 0

fix                 pd background file khan_plasma.h5 static yes

variable gcaIntegrator index rk2
global pusher plasma pd mode gca &
       gca_integrator ${gcaIntegrator} &
       subcycles 1 bad_dt_check yes bad_dt_limit 0.5 dump no

variable dt         equal 5e-10
variable numStep    index 600000
variable dumpFreq   index 300

# v is the GC chord in 2D; physical state rides in the custom attrs
dump dpart particle all ${dumpFreq} output/traj.gcaaxi.${gcaIntegrator} &
     id type x y z vx vy vz p_gca_x p_gca_y p_gca_z p_gca_vpar p_gca_mu
dump_modify dpart first yes

timestep            ${dt}
stats               5000
stats_style         step cpu np
run                 ${numStep}
