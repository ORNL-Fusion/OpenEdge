# Plasma transport, powder injection, ablation, and Li chemistry.

compute cgeom nearest_surf/grid all wall dist surfid nx ny nz
global  pusher plasma pd mode gca gca_switch 2.5 &
        gca_integrator rk4 subcycles 1 &
        bad_dt_check yes bad_dt_limit 0.5 &
        gc_wall flux skip droplet &
        sheath boundary geom cgeom mD_amu 2.0

variable Dperp index 0.1
fix fcd   cross_field_diffusion 1 background pd D_perp ${Dperp}
fix fcoul coulomb/background 1 background pd 2.0 1.0
fix fth   force/thermal 1 background pd ion_thermal yes elec_thermal yes

include input/srate.deck

fix fdrop particulate/emit droplet dropper nevery ${nemit} n v_srate &
          nweight ${nweight} perspecies no normal no vmin 1.0 vmax 1.5 &
          cone 15 dir 0 -1 0 planar yes
fix fcharge particulate/charge 1 background pd model dustt2005 ion_mass_amu 2.0 &
            thermionic no mixture droplet material Li
fix fdrag particulate/drag 1 2.0 1 background pd gravity 0 -9.8 0 &
          model dustt2005 coulomb/self yes material Li mixture droplet
fix fevap particulate/thermal 1 droplet background pd model dustt2005 &
          heatflux/scale 1.0 &
          rocket_eta 0.0 emit_into livapor heating oml ion_mass_amu 2.0 &
          material Li
fix fchem volume/chem/adas 1 Li input/li.reactions output summary

# One simple policy is sufficient for this modest case. It also avoids the
# previous two-run balswitch/nrest control flow.
fix fbal balance 1000 1.1 rcb time
