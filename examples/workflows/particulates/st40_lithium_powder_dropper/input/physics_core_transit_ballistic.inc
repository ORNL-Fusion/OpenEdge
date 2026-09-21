# Gravity-only control: same wall, starts, timestep, and trajectory identities.

global weight cell none
read_particles input/core_transit_cohort.particles 0

global pusher plasma pd mode boris skip droplet
fix fgravity force/gravity droplet 0 -9.8 0
