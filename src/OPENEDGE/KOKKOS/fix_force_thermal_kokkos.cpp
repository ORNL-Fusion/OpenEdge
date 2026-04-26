/* ----------------------------------------------------------------------
   OpenEdge: fix force/thermal — Kokkos backend (skeleton).
   See fix_force_thermal_kokkos.h for the porting plan.
------------------------------------------------------------------------- */

#include "fix_force_thermal_kokkos.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixForceThermalKokkos::FixForceThermalKokkos(SPARTA *sparta, int narg,
                                             char **arg) :
  FixForceThermal(sparta, narg, arg)
{
  kokkosable = 1;
  // The CPU base sets per-particle data masks; the Kokkos backend doesn't
  // currently push any work to device, so leave datamask_read /
  // datamask_modify as the parent already configured them.
}

/* ---------------------------------------------------------------------- */

FixForceThermalKokkos::~FixForceThermalKokkos() {}

/* ---------------------------------------------------------------------- */

void FixForceThermalKokkos::init()
{
  FixForceThermal::init();
  // TODO(kokkos kernel):
  //   pc_bx_     = particle->find_custom((char*)"pc_bx");
  //   pc_by_     = particle->find_custom((char*)"pc_by");
  //   pc_bz_     = particle->find_custom((char*)"pc_bz");
  //   pc_gter_   = particle->find_custom((char*)"pc_grad_te_r");
  //   pc_gtez_   = particle->find_custom((char*)"pc_grad_te_z");
  //   pc_gtir_   = particle->find_custom((char*)"pc_grad_ti_r");
  //   pc_gtiz_   = particle->find_custom((char*)"pc_grad_ti_z");
  // These indices feed ParticleKokkos::k_edvec.h_view[ewhich[idx]].k_view.d_view
  // (a t_float_1d) inside end_of_step() before launching the parallel_for.
}

/* ---------------------------------------------------------------------- */

void FixForceThermalKokkos::start_of_step()
{
  // Host fallback for now. The CPU kick_half is correctness-validated;
  // re-using it keeps `-sf kk` decks producing the right physics until
  // the device kernel lands.
  FixForceThermal::start_of_step();
}

/* ---------------------------------------------------------------------- */

void FixForceThermalKokkos::end_of_step()
{
  // TODO(kokkos kernel): replace the parent call with a Kokkos
  // parallel_for over particle->nlocal that:
  //   1. Reads p.x, p.v, p.ispecies from d_particles(ip).
  //   2. Looks up species mass/charge from d_species(isp).
  //   3. Reads B0,B1,B2 from d_pc_bx[ip], d_pc_by[ip], d_pc_bz[ip].
  //   4. Reads gTiR/gTiZ from d_pc_grad_ti_r[ip], d_pc_grad_ti_z[ip]
  //      (similar for grad_te if elec_thermal is on).
  //   5. Maps bhat to cylindrical via OpenEdge::sparta_v_to_RZphi
  //      (already KOKKOS_INLINE_FUNCTION-decorated).
  //   6. Accumulates a_par = Z^2 * QE * (beta * gradTi.bhat + alpha * gradTe.bhat) / m.
  //   7. Updates p.v[0..2] += a_par * bhat * 0.5 * dt   (twice — start_of_step + end_of_step).
  //
  // Falls back to host while the kernel is being authored. This keeps
  // `-sf kk` decks running with correct physics; benchmark numbers
  // won't show GPU speedup until the kernel lands.
  FixForceThermal::end_of_step();
}
