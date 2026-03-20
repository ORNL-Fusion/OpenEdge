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
#include "memory.h"
#include "particle.h"
#include "update.h"

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
  if (narg < 7)
    error->all(FLERR, "Illegal fix reflect/psi command: "
               "fix ID reflect/psi Nevery equ PATH psi_norm VALUE");

  nevery_ = atoi(arg[2]);
  if (nevery_ <= 0)
    error->all(FLERR, "fix reflect/psi: Nevery must be > 0");

  psi_threshold_ = 0.926;
  nw_ = nh_ = 0;
  psi_axis_ = psib_ = 0.0;

  std::string equ_path;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "equ") == 0 || strcmp(arg[iarg], "geqdsk") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix reflect/psi: missing equ path");
      equ_path = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "psi_norm") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix reflect/psi: missing psi_norm value");
      psi_threshold_ = atof(arg[iarg + 1]);
      iarg += 2;
    } else {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "fix reflect/psi: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  if (equ_path.empty())
    error->all(FLERR, "fix reflect/psi: equ keyword is required");

  read_equ_file(equ_path);

  if (comm->me == 0) {
    printf("fix reflect/psi: equ %s\n", equ_path.c_str());
    printf("  grid: %d x %d, R=[%.4f,%.4f], Z=[%.4f,%.4f]\n",
           nw_, nh_, r_grid_.front(), r_grid_.back(),
           z_grid_.front(), z_grid_.back());
    printf("  psi_axis = %.6e  psib = %.6e\n", psi_axis_, psib_);
    printf("  psi_norm threshold = %.4f\n", psi_threshold_);
  }
}

/* ---------------------------------------------------------------------- */

FixReflectPsi::~FixReflectPsi() {}

/* ---------------------------------------------------------------------- */

int FixReflectPsi::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixReflectPsi::init() {}

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

  double dr = r_grid_[1] - r_grid_[0];
  double dz = z_grid_[1] - z_grid_[0];

  int i = (int)((Rc - r_grid_[0]) / dr);
  int j = (int)((Zc - z_grid_[0]) / dz);
  i = std::min(std::max(i, 0), nw_ - 2);
  j = std::min(std::max(j, 0), nh_ - 2);

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

/* ---------------------------------------------------------------------- */

void FixReflectPsi::start_of_step()
{
  if ((update->ntimestep % nevery_) != 0) return;

  int nlocal = particle->nlocal;
  int dim = domain->dimension;

  // Grow buffer if needed
  if (nlocal > update->psi_reflect_nmax) {
    memory->destroy(update->psi_do_reflect);
    update->psi_reflect_nmax = nlocal + nlocal / 10 + 1;
    memory->create(update->psi_do_reflect, update->psi_reflect_nmax,
                   "update:psi_do_reflect");
  }

  update->psi_reflect_flag = 1;
  memset(update->psi_do_reflect, 0, sizeof(int) * nlocal);

  Particle::OnePart *particles = particle->particles;

  for (int ip = 0; ip < nlocal; ip++) {
    Particle::OnePart &p = particles[ip];

    double R, Z;
    if (dim == 3) {
      R = sqrt(p.x[0] * p.x[0] + p.x[1] * p.x[1]);
      Z = p.x[2];
    } else {
      R = p.x[0];
      Z = p.x[1];
    }

    double psi_n = psi_norm_at_point(R, Z);

    if (psi_n < psi_threshold_) {
      update->psi_do_reflect[ip] = 1;
    }
  }
}
