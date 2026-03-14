/* ----------------------------------------------------------------------
   OpenEdge: generic per-particle field fix — Kokkos version.
------------------------------------------------------------------------- */

#include "fix_field_particle_kokkos.h"
#include "particle.h"
#include "memory_kokkos.h"
#include "error.h"

using namespace SPARTA_NS;

FixFieldParticleKokkos::FixFieldParticleKokkos(SPARTA *sparta, int narg, char **arg) :
  FixFieldParticle(sparta, narg, arg)
{
  kokkos_flag = 1;
  maxpart_kk = 0;
}

FixFieldParticleKokkos::~FixFieldParticleKokkos()
{
  if (copymode) return;
}

void FixFieldParticleKokkos::init()
{
  FixFieldParticle::init();
  int np = particle->maxlocal;
  int nc = size_per_particle_cols;
  if (np > maxpart_kk || maxpart_kk == 0) {
    maxpart_kk = np;
    k_array_particle = DAT::tdual_float_2d_lr("field_part:array", maxpart_kk, nc);
    d_array_particle = k_array_particle.d_view;
  }
}

void FixFieldParticleKokkos::compute_field()
{
  FixFieldParticle::compute_field();
  if (!particle->nlocal) return;

  int np = particle->nlocal;
  int nc = size_per_particle_cols;
  if (np > maxpart_kk) {
    maxpart_kk = particle->maxlocal;
    k_array_particle = DAT::tdual_float_2d_lr("field_part:array", maxpart_kk, nc);
    d_array_particle = k_array_particle.d_view;
  }

  auto h_array = k_array_particle.h_view;
  for (int i = 0; i < np; i++)
    for (int j = 0; j < nc; j++)
      h_array(i, j) = array_particle[i][j];

  k_array_particle.modify_host();
  k_array_particle.sync_device();
  d_array_particle = k_array_particle.d_view;
}
