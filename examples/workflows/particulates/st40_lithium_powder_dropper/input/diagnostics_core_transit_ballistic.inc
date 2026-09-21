# Ballistic-oracle diagnostics at the same 1 ms cadence as the physics cohort.

dump dgrain particle droplet ${ndump} ${outdir}/grain_state &
            id type x y z vx vy vz temp radius p_phi_unwrap
dump_modify dgrain first yes
dump_modify dgrain format float %.14g

compute cnt count droplet/default
stats ${ndump}
stats_style step cpu np c_cnt
