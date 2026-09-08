# Deterministic grain-only physics for the Stage-1 core-transit question.

# Disable SPARTA's stochastic axisymmetric cell-weight cloning/deletion.  A
# deterministic cohort requires one persistent identity per physical launch;
# OpenEdge pweight, not cell sampling weight, carries inventory later.
global weight cell none

read_particles input/core_transit_cohort.particles 0

# Grains are unmagnetized. The normal OpenEdge mover still advances them and
# resolves wall collisions; plasma drag and gravity update their velocity.
global pusher plasma pd mode boris skip droplet

fix fcharge particulate/charge 1 background pd model dustt2005 ion_mass_amu 2.0 &
            thermionic no mixture droplet material Li
fix fdrag particulate/drag 1 2.0 1 background pd gravity 0 -9.8 0 &
          model dustt2005 coulomb/self yes material Li mixture droplet

# Deliberately omit emit_into. The cumulative evaporation and DUSTT terminal
# remainder are retained in f_fevap[1:2], without stochastic Li markers.
fix fevap particulate/thermal 1 droplet background pd heatflux/scale 1.0 &
          rocket_eta 0.0 heating oml ion_mass_amu 2.0 material Li &
          model dustt2005
