/* ----------------------------------------------------------------------
    OpenEdge: fix reflect/psi
    See fix_reflect_psi.h for description and syntax.
------------------------------------------------------------------------- */

#include "fix_reflect_psi.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "particle.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace SPARTA_NS;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};  // several files
enum { REFLECT = 0, DELETE = 1 };

/* ---------------------------------------------------------------------- */

FixReflectPsi::FixReflectPsi(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg < 7)
    error->all(FLERR, "Illegal fix reflect/psi command: "
               "fix ID reflect/psi Nevery geqdsk PATH psi_norm VALUE "
               "[action reflect|delete]");

  // arg[0] = fix ID
  // arg[1] = reflect/psi
  // arg[2] = Nevery
  nevery_ = atoi(arg[2]);
  if (nevery_ <= 0)
    error->all(FLERR, "fix reflect/psi: Nevery must be > 0");

  // Parse keyword arguments
  psi_threshold_ = 0.05;
  action_ = REFLECT;
  nmax_prev_ = 0;
  x_prev_ = NULL;

  std::string geqdsk_path;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "geqdsk") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix reflect/psi: missing geqdsk path");
      geqdsk_path = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "psi_norm") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix reflect/psi: missing psi_norm value");
      psi_threshold_ = atof(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "action") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix reflect/psi: missing action value");
      if (strcmp(arg[iarg + 1], "reflect") == 0) action_ = REFLECT;
      else if (strcmp(arg[iarg + 1], "delete") == 0) action_ = DELETE;
      else error->all(FLERR, "fix reflect/psi: action must be reflect or delete");
      iarg += 2;
    } else {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "fix reflect/psi: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  if (geqdsk_path.empty())
    error->all(FLERR, "fix reflect/psi: geqdsk keyword is required");

  // Read equilibrium
  int ret = geqdsk_.read(geqdsk_path);
  if (ret < 0) {
    char msg[512];
    snprintf(msg, sizeof(msg),
             "fix reflect/psi: failed to read geqdsk '%s': %s",
             geqdsk_path.c_str(), geqdsk_.error_msg().c_str());
    error->all(FLERR, msg);
  }

  const GEQDSKData &d = geqdsk_.data();
  if (comm->me == 0) {
    printf("fix reflect/psi: geqdsk %s\n", geqdsk_path.c_str());
    printf("  simag = %.6e  sibry = %.6e\n", d.simag, d.sibry);
    printf("  psi_norm threshold = %.4f\n", psi_threshold_);
    printf("  action = %s\n", action_ == REFLECT ? "reflect" : "delete");
  }
}

/* ---------------------------------------------------------------------- */

FixReflectPsi::~FixReflectPsi()
{
  memory->destroy(x_prev_);
}

/* ---------------------------------------------------------------------- */

int FixReflectPsi::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixReflectPsi::init() {}

/* ---------------------------------------------------------------------- */

void FixReflectPsi::start_of_step()
{
  // Save current positions before the move step
  if ((update->ntimestep % nevery_) != 0) return;

  int nlocal = particle->nlocal;

  // Grow buffer if needed
  if (nlocal > nmax_prev_) {
    memory->destroy(x_prev_);
    nmax_prev_ = nlocal + nlocal / 10 + 1;
    memory->create(x_prev_, nmax_prev_, 3, "reflect_psi:x_prev");
  }

  Particle::OnePart *particles = particle->particles;
  for (int ip = 0; ip < nlocal; ip++) {
    x_prev_[ip][0] = particles[ip].x[0];
    x_prev_[ip][1] = particles[ip].x[1];
    x_prev_[ip][2] = particles[ip].x[2];
  }
}

/* ---------------------------------------------------------------------- */

void FixReflectPsi::end_of_step()
{
  if ((update->ntimestep % nevery_) != 0) return;

  Particle::OnePart *particles = particle->particles;
  int nlocal = particle->nlocal;
  int dim = domain->dimension;

  int nreflect = 0;
  int ndelete = 0;

  for (int ip = 0; ip < nlocal; ip++) {
    Particle::OnePart &p = particles[ip];

    // Convert Cartesian (x,y,z) to cylindrical (R,Z)
    double R, Z;
    if (dim == 3) {
      R = sqrt(p.x[0] * p.x[0] + p.x[1] * p.x[1]);
      Z = p.x[2];
    } else {
      // 2D: x[0]=R, x[1]=Z
      R = p.x[0];
      Z = p.x[1];
    }

    double psi_n = geqdsk_.psi_norm_at_point(R, Z);

    if (psi_n < psi_threshold_) {
      if (action_ == DELETE) {
        // Mark particle for deletion
        p.flag = PDISCARD;
        ndelete++;
      } else {
        // Reflect: restore previous position and reverse velocity
        if (nmax_prev_ > ip) {
          p.x[0] = x_prev_[ip][0];
          p.x[1] = x_prev_[ip][1];
          p.x[2] = x_prev_[ip][2];
        }

        // Reverse radial velocity component
        if (dim == 3) {
          double phi = atan2(p.v[1], p.v[0]);
          double vr = p.v[0] * cos(phi) + p.v[1] * sin(phi);
          double vphi = -p.v[0] * sin(phi) + p.v[1] * cos(phi);
          // Flip radial component
          vr = -vr;
          p.v[0] = vr * cos(phi) - vphi * sin(phi);
          p.v[1] = vr * sin(phi) + vphi * cos(phi);
          // Keep vz unchanged
        } else {
          // 2D: flip v[0] (radial)
          p.v[0] = -p.v[0];
        }
        nreflect++;
      }
    }
  }

  // Compress particle list to remove deleted particles
  if (action_ == DELETE && ndelete > 0) {
    particle->compress_rebalance();
  }

  // Optional periodic reporting
  int allreflect, alldelete;
  MPI_Allreduce(&nreflect, &allreflect, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&ndelete, &alldelete, 1, MPI_INT, MPI_SUM, world);

  if ((allreflect > 0 || alldelete > 0) && comm->me == 0 &&
      update->ntimestep % (100 * nevery_) == 0) {
    if (action_ == REFLECT)
      printf("fix reflect/psi: step %ld, reflected %d particles\n",
             update->ntimestep, allreflect);
    else
      printf("fix reflect/psi: step %ld, deleted %d particles\n",
             update->ntimestep, alldelete);
  }
}
