# Pusher verification

Pusher verification suite. Every test bakes PASS/FAIL + exit code into its
plot/check script.

| Test | Covers | Reference |
|---|---|---|
| `orbit/` | Boris vs GCA (rk2/rk4) drift physics, 3D / 2D Cartesian / 2D axi; MPI rank invariance (`check_mpi.sh`) | Khan et al. (2012) banana orbit |
| `hybrid/` | (planned) hybrid boris_near wall handoff: Boris vs hybrid vs GCA single-ion impact on a flat target | Boris-resolved impact state |

Single-particle runs: use `-np 1` for reproducible gate numbers. The MPI
rank-dependence bug (btf/rtf never broadcast for embedded /equilibrium →
B_phi = 0 on ranks != 0) is FIXED; `orbit/check_mpi.sh` gates the fix.
Multi-rank runs agree to dump precision for ~150k steps, after which
ulp-level migration-retrace roundoff is amplified by the marginally
trapped orbit — expected, not a bug.
