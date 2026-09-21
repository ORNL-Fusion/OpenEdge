# Compact deterministic-cohort diagnostics (5 ms cadence at dt=2 us).

dump dgrain particle droplet ${ndump} ${outdir}/grain_state &
            id type x y z vx vy vz temp radius p_particulate_charge &
            p_droplet_heating_q p_droplet_a_over_lambdaD &
            p_droplet_oml_weight p_phi_unwrap
dump_modify dgrain first yes
dump_modify dgrain format float %.14g

fix fevapout ave/time ${ndump} 1 ${ndump} f_fevap[1] f_fevap[2] &
             file ${outdir}/evaporation_inventory

compute cnt count droplet/default
stats ${ndump}
stats_style step cpu np c_cnt
