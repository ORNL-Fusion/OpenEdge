/* ----------------------------------------------------------------------
    OpenEdge: fix reflect/psi
    See fix_reflect_psi.h for description and syntax.

    Marks particles for radial velocity reflection when psi_norm < threshold.
    The actual reflection is applied in update.cpp during the move step,
    ensuring proper surface collision checks.
------------------------------------------------------------------------- */

#include "fix_reflect_psi.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_background.h"
#include "memory.h"
#include "mixture.h"
#include "modify.h"
#include "particle.h"
#include "update.h"
#include "mpi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

using namespace SPARTA_NS;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};  // several files

/* ---------------------------------------------------------------------- */

FixReflectPsi::FixReflectPsi(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg < 4)
    error->all(FLERR, "Illegal fix reflect/psi command: "
               "fix ID reflect/psi {equ PATH | background FIXID} "
               "[psi_norm VALUE] [action ...] [mixture ID]");

  action_ = PSI_ACTION_REFLECT;
  imix_ = -1;
  nrows_ = 0;
  pweight_index_ = pweight_ewhich_ = -1;
  start_step_ = reduced_step_ = -1;
  nw_ = nh_ = 0;
  psi_axis_ = psib_ = 0.0;

  std::string equ_path;
  std::string plasma_fix_id;
  std::string mixture_id;
  int threshold_user_set = 0;
  psi_threshold_ = 0.926;  // default for equ/geqdsk mode

  int iarg = 2;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "equ") == 0 || strcmp(arg[iarg], "geqdsk") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix reflect/psi: missing equ path");
      equ_path = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "background") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix reflect/psi: missing background fix id");
      plasma_fix_id = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "psi_norm") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix reflect/psi: missing psi_norm value");
      psi_threshold_ = atof(arg[iarg + 1]);
      threshold_user_set = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "action") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix reflect/psi: missing action value");
      if (strcmp(arg[iarg + 1], "reflect") == 0)
        action_ = PSI_ACTION_REFLECT;
      else if (strcmp(arg[iarg + 1], "absorb") == 0)
        action_ = PSI_ACTION_ABSORB;
      else
        error->all(FLERR, "fix reflect/psi: action must be 'reflect' or 'absorb'");
      iarg += 2;
    } else if (strcmp(arg[iarg], "mixture") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix reflect/psi: missing mixture ID");
      mixture_id = arg[iarg + 1];
      iarg += 2;
    } else {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "fix reflect/psi: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  if (equ_path.empty() && plasma_fix_id.empty())
    error->all(FLERR, "fix reflect/psi: one of 'equ PATH' or 'background FIXID' is required");
  if (!equ_path.empty() && !plasma_fix_id.empty())
    error->all(FLERR, "fix reflect/psi: 'equ' and 'background' are mutually exclusive");
  if (!mixture_id.empty()) {
    imix_ = particle->find_mixture(const_cast<char *>(mixture_id.c_str()));
    if (imix_ < 0)
      error->all(FLERR, "fix reflect/psi: unknown mixture ID");
  }

  nrows_ = (imix_ >= 0) ? particle->mixture[imix_]->ngroup
                         : particle->nspecies;
  if (nrows_ <= 0)
    error->all(FLERR, "fix reflect/psi: no species available for tally");
  vector_flag = 1;
  size_vector = 3 * nrows_;
  global_freq = 1;
  absorbed_events_local_.assign(nrows_, 0.0);
  absorbed_physical_local_.assign(nrows_, 0.0);
  absorbed_events_global_.assign(nrows_, 0.0);
  absorbed_physical_global_.assign(nrows_, 0.0);

  if (!equ_path.empty()) {
    read_equ_file(equ_path);
  } else {
    // Default threshold in background mode: 0.0 = SOLEDGE /psicore surface.
    // Anything with psi_norm < 0 is on the core side of /psicore.
    if (!threshold_user_set) psi_threshold_ = 0.0;
    load_from_background(plasma_fix_id);
  }

  if (comm->me == 0) {
    if (!equ_path.empty())
      printf("fix reflect/psi: equ %s\n", equ_path.c_str());
    else
      printf("fix reflect/psi: background %s\n", plasma_fix_id.c_str());
    printf("  grid: %d x %d, R=[%.4f,%.4f], Z=[%.4f,%.4f]\n",
           nw_, nh_, r_grid_.front(), r_grid_.back(),
           z_grid_.front(), z_grid_.back());
    printf("  psi_axis = %.6e  psib = %.6e\n", psi_axis_, psib_);
    printf("  psi_norm threshold = %.4f\n", psi_threshold_);
    printf("  action = %s\n", action_ == PSI_ACTION_ABSORB ? "absorb" : "reflect");
    printf("  species scope = %s\n",
           imix_ < 0 ? "all" : particle->mixture[imix_]->id);
  }
}

/* ---------------------------------------------------------------------- */

void FixReflectPsi::load_from_background(const std::string &fix_id)
{
  int ifix = modify->find_fix(fix_id.c_str());
  if (ifix < 0) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "fix reflect/psi: cannot find fix background '%s'",
             fix_id.c_str());
    error->all(FLERR, msg);
  }
  FixBackground *pd = dynamic_cast<FixBackground *>(modify->fix[ifix]);
  if (!pd)
    error->all(FLERR, "fix reflect/psi: referenced fix is not background");

  // Ensure the plasma fix has loaded its data (init() may not have fired yet).
  if (!pd->has_equ) pd->init();

  if (!pd->has_equ || pd->psirz.empty())
    error->all(FLERR,
      "fix reflect/psi: background fix does not expose psi. "
      "Make sure plasma.h5 contains /psi + /psicore + /psisep");

  nw_ = pd->equ_jm;
  nh_ = pd->equ_km;
  r_grid_ = pd->equ_r;
  z_grid_ = pd->equ_z;
  psirz_  = pd->psirz;
  psi_axis_ = pd->psi_axis;
  psib_     = pd->psib;
}

/* ---------------------------------------------------------------------- */

FixReflectPsi::~FixReflectPsi()
{
  // init() hands Update raw pointers into fix-owned vectors; clear them
  // so an unfix followed by another run cannot reflect through dangling
  // storage
  if (update && update->psi_reflect_flag) {
    update->psi_reflect_flag = 0;
    update->psi_r_grid = nullptr;
    update->psi_z_grid = nullptr;
    update->psi_rz = nullptr;
  }
}

/* ---------------------------------------------------------------------- */

int FixReflectPsi::setmask()
{
  return 0;  // no per-step callbacks needed; data is set in init()
}

/* ---------------------------------------------------------------------- */

void FixReflectPsi::init()
{
  // Pass equilibrium data pointers to update for use in mover
  update->psi_reflect_flag = 1;
  update->psi_reflect_action = action_;
  update->psi_reflect_threshold = psi_threshold_;
  update->psi_reflect_imix = imix_;
  update->psi_reflect_fix = this;
  update->psi_nw = nw_;
  update->psi_nh = nh_;
  update->psi_axis = psi_axis_;
  update->psi_bry = psib_;
  update->psi_r_grid = r_grid_.data();
  update->psi_z_grid = z_grid_.data();
  update->psi_rz = psirz_.data();

  // fix particle/weight may appear later than this fix in the input deck, but
  // all fix constructors have run before init(). Use it when available;
  // otherwise fall back to the standard fnum marker weight.
  pweight_index_ = particle->find_custom((char *) "pweight");
  pweight_ewhich_ = (pweight_index_ >= 0) ? particle->ewhich[pweight_index_] : -1;
  if (start_step_ < 0) start_step_ = update->ntimestep;
}

/* ----------------------------------------------------------------------
   Map a particle species to a row in the public tally vector.
------------------------------------------------------------------------- */

int FixReflectPsi::row_for_species(int ispecies) const
{
  if (ispecies < 0 || ispecies >= particle->nspecies) return -1;
  if (imix_ < 0) return ispecies;
  int *species2group = particle->mixture[imix_]->species2group;
  return species2group ? species2group[ispecies] : -1;
}

/* ----------------------------------------------------------------------
   Record one absorbed simulation particle.

   pweight is the complete physical population carried by an OpenEdge marker.
   Do not multiply it by SPARTA's cell/radial sampling weight: that weight is
   part of legacy sampling-density normalization, not this global inventory.
   With no pweight custom, the standard fnum convention is used.
------------------------------------------------------------------------- */

void FixReflectPsi::tally_absorb(int ispecies, int iparticle)
{
  const int row = row_for_species(ispecies);
  if (row < 0 || row >= nrows_ || iparticle < 0 ||
      iparticle >= particle->nlocal) return;

  double marker_weight = update->fnum;
  if (pweight_index_ >= 0) {
    pweight_ewhich_ = particle->ewhich[pweight_index_];
    const double candidate = particle->edvec[pweight_ewhich_][iparticle];
    if (candidate > 0.0 && std::isfinite(candidate)) marker_weight = candidate;
  }
  absorbed_events_local_[row] += 1.0;
  absorbed_physical_local_[row] += marker_weight;
  reduced_step_ = -1;
}

/* ----------------------------------------------------------------------
   Fold a batch of absorptions tallied on the device for one species
   (nevents simulation particles carrying `weight` physical particles).
------------------------------------------------------------------------- */

void FixReflectPsi::tally_absorb_bulk(int ispecies, double nevents,
                                      double weight)
{
  const int row = row_for_species(ispecies);
  if (row < 0 || row >= nrows_ || !(nevents > 0.0)) return;
  absorbed_events_local_[row] += nevents;
  absorbed_physical_local_[row] += weight;
  reduced_step_ = -1;
}

/* ---------------------------------------------------------------------- */

void FixReflectPsi::reduce_tallies()
{
  if (reduced_step_ == update->ntimestep) return;
  MPI_Allreduce(absorbed_events_local_.data(), absorbed_events_global_.data(),
                nrows_, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(absorbed_physical_local_.data(),
                absorbed_physical_global_.data(), nrows_, MPI_DOUBLE,
                MPI_SUM, world);
  reduced_step_ = update->ntimestep;
}

/* ----------------------------------------------------------------------
   Flattened vector, three entries per selected mixture group/species:
     3*i+1 simulation absorption events (cumulative)
     3*i+2 physical particles absorbed (cumulative)
     3*i+3 physical removal rate averaged since fix initialization [s^-1]
------------------------------------------------------------------------- */

double FixReflectPsi::compute_vector(int index)
{
  if (index < 0 || index >= size_vector) return 0.0;
  reduce_tallies();
  const int row = index / 3;
  const int field = index % 3;
  if (field == 0) return absorbed_events_global_[row];
  if (field == 1) return absorbed_physical_global_[row];
  const double elapsed = (update->ntimestep - start_step_) * update->dt;
  return elapsed > 0.0 ? absorbed_physical_global_[row] / elapsed : 0.0;
}

/* ---------------------------------------------------------------------- */

void FixReflectPsi::read_equ_file(const std::string &path)
{
  std::ifstream ifs(path);
  if (!ifs.good()) {
    char msg[512];
    snprintf(msg, sizeof(msg),
             "fix reflect/psi: cannot open file '%s'", path.c_str());
    error->all(FLERR, msg);
  }
  std::string text((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
  ifs.close();

  auto parse_int = [&](const char *name) -> int {
    size_t pos = text.find(std::string(name));
    while (pos != std::string::npos) {
      size_t eq = text.find('=', pos);
      if (eq != std::string::npos && eq - pos < 20) {
        int val = atoi(text.c_str() + eq + 1);
        if (val > 0) return val;
      }
      pos = text.find(std::string(name), pos + 1);
    }
    return 0;
  };

  nw_ = parse_int("jm");
  nh_ = parse_int("km");

  if (nw_ <= 0 || nh_ <= 0) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "fix reflect/psi: cannot parse jm/km from '%s' (got %d, %d)",
             path.c_str(), nw_, nh_);
    error->all(FLERR, msg);
  }

  {
    size_t pos = text.find("psib");
    if (pos != std::string::npos) {
      size_t eq = text.find('=', pos);
      if (eq != std::string::npos) psib_ = atof(text.c_str() + eq + 1);
    }
  }

  auto read_floats_after = [&](const std::string &marker, int n,
                               std::vector<double> &out) {
    size_t pos = text.find(marker);
    if (pos == std::string::npos) {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "fix reflect/psi: cannot find '%s' in %s",
               marker.c_str(), path.c_str());
      error->all(FLERR, msg);
    }
    pos += marker.size();

    out.clear();
    out.reserve(n);

    const char *c = text.c_str() + pos;
    const char *end = text.c_str() + text.size();
    while ((int)out.size() < n && c < end) {
      while (c < end && !std::isdigit(*c) && *c != '+' && *c != '-' && *c != '.') c++;
      if (c >= end) break;
      char *endp;
      double val = strtod(c, &endp);
      if (endp > c) {
        out.push_back(val);
        c = endp;
      } else {
        c++;
      }
    }

    if ((int)out.size() < n) {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "fix reflect/psi: only read %d of %d values after '%s'",
               (int)out.size(), n, marker.c_str());
      error->all(FLERR, msg);
    }
  };

  read_floats_after("r(1:jm);", nw_, r_grid_);
  read_floats_after("z(1:km);", nh_, z_grid_);

  if (nw_ < 2 || nh_ < 2)
    error->all(FLERR, "fix reflect/psi: equilibrium grid must be at least 2x2");
  if (!std::is_sorted(r_grid_.begin(), r_grid_.end()) ||
      !std::is_sorted(z_grid_.begin(), z_grid_.end()))
    error->all(FLERR, "fix reflect/psi: equilibrium R/Z grids must be monotonic increasing");

  std::vector<double> psi_minus_psib;
  read_floats_after("((psi(j,k)-psib,j=1,jm),k=1,km)", nw_ * nh_, psi_minus_psib);

  psirz_.resize(nw_ * nh_);
  for (int i = 0; i < nw_ * nh_; i++)
    psirz_[i] = psi_minus_psib[i] + psib_;

  psi_axis_ = 1e30;
  int j0 = nh_ / 4, j1 = 3 * nh_ / 4;
  int i0 = nw_ / 4, i1 = 3 * nw_ / 4;
  for (int j = j0; j < j1; j++)
    for (int i = i0; i < i1; i++) {
      double p = psirz_[j * nw_ + i];
      if (p < psi_axis_) psi_axis_ = p;
    }
}

/* ---------------------------------------------------------------------- */

double FixReflectPsi::psi_norm_at_point(double R, double Z) const
{
  if (r_grid_.empty() || z_grid_.empty() || psirz_.empty()) return 1.0;

  double Rc = std::min(std::max(R, r_grid_.front()), r_grid_.back());
  double Zc = std::min(std::max(Z, z_grid_.front()), z_grid_.back());

  auto bracket_index = [](const std::vector<double> &grid, double x) {
    if (x <= grid.front()) return 0;
    if (x >= grid.back()) return static_cast<int>(grid.size()) - 2;
    auto it = std::upper_bound(grid.begin(), grid.end(), x);
    int idx = static_cast<int>(it - grid.begin()) - 1;
    if (idx < 0) idx = 0;
    if (idx > static_cast<int>(grid.size()) - 2)
      idx = static_cast<int>(grid.size()) - 2;
    return idx;
  };

  int i = bracket_index(r_grid_, Rc);
  int j = bracket_index(z_grid_, Zc);

  double dr = r_grid_[i+1] - r_grid_[i];
  double dz = z_grid_[j+1] - z_grid_[j];
  if (std::abs(dr) < 1e-30 || std::abs(dz) < 1e-30) return 1.0;

  double t = (Rc - r_grid_[i]) / dr;
  double u = (Zc - z_grid_[j]) / dz;
  t = std::min(std::max(t, 0.0), 1.0);
  u = std::min(std::max(u, 0.0), 1.0);

  double psi = (1-t)*(1-u)*psirz_[j*nw_+i]     + t*(1-u)*psirz_[j*nw_+i+1]
             + (1-t)*u*psirz_[(j+1)*nw_+i] + t*u*psirz_[(j+1)*nw_+i+1];

  double dpsi = psib_ - psi_axis_;
  if (std::abs(dpsi) < 1e-30) return 1.0;
  return (psi - psi_axis_) / dpsi;
}

/* ----------------------------------------------------------------------
   Bilinear psi_N and its local (R,Z) gradient.
------------------------------------------------------------------------- */

double FixReflectPsi::psi_norm_gradient(double R, double Z,
                                        double &grad_r,
                                        double &grad_z) const
{
  grad_r = grad_z = 0.0;
  if (r_grid_.size() < 2 || z_grid_.size() < 2 || psirz_.empty()) return 1.0;

  const double Rc = std::min(std::max(R, r_grid_.front()), r_grid_.back());
  const double Zc = std::min(std::max(Z, z_grid_.front()), z_grid_.back());
  auto bracket = [](const std::vector<double> &grid, double value) {
    if (value <= grid.front()) return 0;
    if (value >= grid.back()) return static_cast<int>(grid.size()) - 2;
    const auto it = std::upper_bound(grid.begin(), grid.end(), value);
    return std::max(0, std::min(static_cast<int>(grid.size()) - 2,
                               static_cast<int>(it - grid.begin()) - 1));
  };

  const int i = bracket(r_grid_, Rc);
  const int j = bracket(z_grid_, Zc);
  const double dr = r_grid_[i+1] - r_grid_[i];
  const double dz = z_grid_[j+1] - z_grid_[j];
  const double dpsi = psib_ - psi_axis_;
  if (std::abs(dr) < 1.0e-30 || std::abs(dz) < 1.0e-30 ||
      std::abs(dpsi) < 1.0e-30) return 1.0;

  const double t = std::min(std::max((Rc-r_grid_[i])/dr, 0.0), 1.0);
  const double u = std::min(std::max((Zc-z_grid_[j])/dz, 0.0), 1.0);
  const double p00 = psirz_[j*nw_+i];
  const double p10 = psirz_[j*nw_+i+1];
  const double p01 = psirz_[(j+1)*nw_+i];
  const double p11 = psirz_[(j+1)*nw_+i+1];
  const double psi = (1.0-t)*(1.0-u)*p00 + t*(1.0-u)*p10 +
                     (1.0-t)*u*p01 + t*u*p11;
  grad_r = ((1.0-u)*(p10-p00) + u*(p11-p01)) / (dr*dpsi);
  grad_z = ((1.0-t)*(p01-p00) + t*(p11-p10)) / (dz*dpsi);
  return (psi-psi_axis_) / dpsi;
}

/* ---------------------------------------------------------------------- */

double FixReflectPsi::psi_norm_at_sparta(const double xyz[3]) const
{
  double R, Z;
  if (domain->dimension == 3) {
    R = std::hypot(xyz[0], xyz[1]);
    Z = xyz[2];
  } else if (domain->axisymmetric) {
    Z = xyz[0];
    R = xyz[1];
  } else {
    R = xyz[0];
    Z = xyz[1];
  }
  return psi_norm_at_point(R, Z);
}

/* ----------------------------------------------------------------------
   Locate an outside-to-inside crossing and return the outward local normal.
   The root is solved on the actual Cartesian/SPARTA chord, so R(phi) is
   handled correctly in 3-D rather than linearly interpolating cylindrical R.
------------------------------------------------------------------------- */

bool FixReflectPsi::segment_crossing(const double x0[3], const double x1[3],
                                     double &fraction, double normal[3]) const
{
  const double p0 = psi_norm_at_sparta(x0);
  const double p1 = psi_norm_at_sparta(x1);
  if (p0 < psi_threshold_ || p1 >= psi_threshold_) return false;

  double lo = 0.0, hi = 1.0;
  double xc[3] = {0.0, 0.0, 0.0};
  for (int iter = 0; iter < 60; iter++) {
    const double mid = 0.5*(lo+hi);
    for (int k = 0; k < 3; k++) xc[k] = x0[k] + mid*(x1[k]-x0[k]);
    if (psi_norm_at_sparta(xc) >= psi_threshold_) lo = mid;
    else hi = mid;
  }
  fraction = 0.5*(lo+hi);
  for (int k = 0; k < 3; k++)
    xc[k] = x0[k] + fraction*(x1[k]-x0[k]);

  double R, Z;
  if (domain->dimension == 3) {
    R = std::hypot(xc[0], xc[1]);
    Z = xc[2];
  } else if (domain->axisymmetric) {
    Z = xc[0];
    R = xc[1];
  } else {
    R = xc[0];
    Z = xc[1];
  }
  double gR, gZ;
  psi_norm_gradient(R, Z, gR, gZ);

  if (domain->dimension == 3) {
    if (R <= 1.0e-30) return false;
    normal[0] = gR*xc[0]/R;
    normal[1] = gR*xc[1]/R;
    normal[2] = gZ;
  } else if (domain->axisymmetric) {
    normal[0] = gZ;
    normal[1] = gR;
    normal[2] = 0.0;
  } else {
    normal[0] = gR;
    normal[1] = gZ;
    normal[2] = 0.0;
  }
  const double nmag = std::sqrt(normal[0]*normal[0] +
                                normal[1]*normal[1] +
                                normal[2]*normal[2]);
  if (!(nmag > 1.0e-20) || !std::isfinite(nmag)) return false;
  normal[0] /= nmag;
  normal[1] /= nmag;
  normal[2] /= nmag;
  return true;
}
