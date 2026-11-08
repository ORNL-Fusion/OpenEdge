/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.github.io
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#include <cstring>
#include "math.h"
#include "surf_collide_vanish_kokkos.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

SurfCollideVanishKokkos::SurfCollideVanishKokkos(SPARTA *sparta, int narg, char **arg) :
  SurfCollideVanish(sparta, narg, arg)
{
  kokkosable = 1;

  k_nsingle = DAT::tdual_int_scalar("SurfCollide:nsingle");
  d_nsingle = k_nsingle.view_device();
  h_nsingle = k_nsingle.view_host();
  p_nsingle = t_pinned_int_scalar("SurfCollide:nsingle_pinned");

  allowreact = 0;
}

SurfCollideVanishKokkos::SurfCollideVanishKokkos(SPARTA *sparta) :
  SurfCollideVanish(sparta)
{
  copy = 1;
}

/* ---------------------------------------------------------------------- */

void SurfCollideVanishKokkos::pre_collide()
{
  Kokkos::deep_copy(DeviceType(),d_nsingle,0);
}

/* ---------------------------------------------------------------------- */

void SurfCollideVanishKokkos::post_collide()
{
  // nothing: the readback is issued by post_collide_issue()
}

/* ---------------------------------------------------------------------- */

void SurfCollideVanishKokkos::post_collide_issue()
{
  Kokkos::deep_copy(DeviceType(),p_nsingle,d_nsingle);   // async into pinned staging; UpdateKokkos fences once
}

/* ---------------------------------------------------------------------- */

void SurfCollideVanishKokkos::post_collide_apply()
{
  memcpy(h_nsingle.data(),p_nsingle.data(),h_nsingle.span()*sizeof(int));   // host <-> pinned staging (plain memcpy)
  int m = surf->find_collide(id);
  auto sc = surf->sc[m];
  sc->nsingle += h_nsingle();
}
