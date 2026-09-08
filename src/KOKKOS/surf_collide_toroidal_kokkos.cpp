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

#include "math.h"
#include "surf_collide_toroidal_kokkos.h"
#include "surf.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

SurfCollideToroidalKokkos::SurfCollideToroidalKokkos(SPARTA *sparta, int narg, char **arg) :
  SurfCollideToroidal(sparta, narg, arg)
{
  kokkosable = 1;
  k_nsingle = DAT::tdual_int_scalar("SurfCollide:nsingle");
  d_nsingle = k_nsingle.view_device();
  h_nsingle = k_nsingle.view_host();
  allowreact = 0;
}

SurfCollideToroidalKokkos::SurfCollideToroidalKokkos(SPARTA *sparta) :
  SurfCollideToroidal(sparta)
{
  copy = 1;
}

/* ---------------------------------------------------------------------- */

void SurfCollideToroidalKokkos::pre_collide()
{
  Kokkos::deep_copy(d_nsingle,0);
}

/* ---------------------------------------------------------------------- */

void SurfCollideToroidalKokkos::post_collide()
{
  Kokkos::deep_copy(h_nsingle,d_nsingle);

  // can't modify the copy directly, use the original
  int m = surf->find_collide(id);
  auto sc = surf->sc[m];
  sc->nsingle += h_nsingle();
}
