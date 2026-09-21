# DUSTT-2005/OML physics for the deterministic multi-diameter response scan.

read_particles input/core_transit_size_scan.particles 0

global pusher plasma pd mode boris skip droplet

fix fcharge particulate/charge 1 background pd model dustt2005 ion_mass_amu 2.0 &
            thermionic no mixture droplet material Li
fix fdrag particulate/drag 1 2.0 1 background pd gravity 0 -9.8 0 &
          model dustt2005 coulomb/self yes material Li mixture droplet

# The thermal solver adaptively substeps within each OpenEdge half-step, limiting
# local temperature changes to 25 K and radius loss to 2%. The outer 2/1/0.5-us
# scan still tests motion/background sampling and complete coupled integration.
fix fevap particulate/thermal 1 droplet background pd heatflux/scale ${heat_scale} &
          rocket_eta 0.0 heating oml ion_mass_amu 2.0 material Li &
          model dustt2005
