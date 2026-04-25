/* ----------------------------------------------------------------------
   OpenEdge: fix coulomb/background -- kinetic vs. Maxwellian fluid bg.
------------------------------------------------------------------------- */

#include "fix_coulomb_background.h"

#include <cstring>

#include "error.h"
#include "input.h"
#include "update.h"

using namespace SPARTA_NS;

FixCoulombBackground::FixCoulombBackground(SPARTA *sparta, int narg, char **arg) :
  FixCoulombBase(sparta, narg, arg)
{
  do_binary_ = 0;
  have_background_ = 1;

  int iarg = iarg_after_common_;
  if (iarg + 2 > narg)
    error->all(FLERR,
      "fix coulomb/background: need A_bg Z_bg after plasma block");

  A_bg_ = input->numeric(FLERR, arg[iarg++]);
  Z_bg_ = input->numeric(FLERR, arg[iarg++]);

  if (!use_plasma_data_) {
    if (iarg + 6 > narg)
      error->all(FLERR,
        "fix coulomb/background: need TiSrc NiSrc VparSrc BxSrc BySrc BzSrc");
    parse_compute_src(arg[iarg++], srcTi_bg_,  "Ti_bg");
    parse_compute_src(arg[iarg++], srcNi_bg_,  "Ni_bg");
    parse_compute_src(arg[iarg++], srcVpar_bg_, "Vpar_bg");
    parse_compute_src(arg[iarg++], srcBx_,     "Bx");
    parse_compute_src(arg[iarg++], srcBy_,     "By");
    parse_compute_src(arg[iarg++], srcBz_,     "Bz");
  }

  if (iarg < narg)
    error->all(FLERR, "fix coulomb/background: extra arguments");

  // precompute background mass and charge
  const double amu = 1.66053906660e-27;  // kg
  m_bg_ = A_bg_ * amu;
  q_bg_ = Z_bg_ * update->echarge;
}
