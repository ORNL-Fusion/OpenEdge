# Core-transit diagnostics. There is no numerical core surface to dump.

dump dpart particle all ${ndump} ${outdir}/state id type x y z vx vy vz &
           p_phi_unwrap
dump dgrain particle droplet ${ndump} ${outdir}/grain_state &
            id type x y z vx vy vz temp radius p_particulate_charge &
            p_phi_unwrap

fix     fpw particle/weight
# Species-resolved, global core sink. The fcore vector is flattened as
# [marker events, cumulative physical particles, mean physical rate (s^-1)]
# for Li, Li+, Li2+, and Li3+ in that order. The cumulative pweight ledger is
# the conserved quantity; finite-window rates are obtained by differencing it.
fix     fcoreflux ave/time ${ndump} 1 ${ndump} f_fcore mode vector &
        file ${outdir}/core_flux
compute cli grid/weighted all allLi nrho_w
fix     fli ave/grid all 10 1000 10000 &
        c_cli[1] c_cli[2] c_cli[3] c_cli[4] ave one
dump    dgrid grid all 10000 ${outdir}/li_dens id xc yc &
        f_fli[1] f_fli[2] f_fli[3] f_fli[4]

dump dwall surf wall 10000 ${outdir}/wall_adens id &
     s_adens[1] s_adens[2] s_adens[3] s_adens[4] s_adens[5] s_adens_dep

compute cnt count droplet/default allLi/Li allLi/Li+ allLi/Li2+ allLi/Li3+
stats ${ndump}
stats_style step cpu np nscoll nsreact &
            c_cnt[1] c_cnt[2] c_cnt[3] c_cnt[4] c_cnt[5]
