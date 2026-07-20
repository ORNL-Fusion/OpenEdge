/* ----------------------------------------------------------------------
    OpenEdge: compute surf/weighted — pweight-aware per-surf flux tally.
    See compute_surf_weighted.h. Subclasses ComputeSurf and reuses its
    framework (init_normflux, clear, tallyinfo, post_process_surf,
    reallocate); overrides only surf_tally to weight by the pweight custom.
------------------------------------------------------------------------- */

#include "compute_surf_weighted.h"

#include <cstring>

#include "error.h"
#include "mixture.h"
#include "particle.h"
#include "surf.h"
#include "update.h"

using namespace SPARTA_NS;

// must match the (file-local) enum order in compute_surf.cpp
enum{NUM,NUMWT,NFLUX,NFLUXIN,MFLUX,MFLUXIN,FX,FY,FZ,TX,TY,TZ,
     PRESS,XPRESS,YPRESS,ZPRESS,XSHEAR,YSHEAR,ZSHEAR,KE,EROT,EVIB,ECHEM,ETOT};

/* ---------------------------------------------------------------------- */

ComputeSurfWeighted::ComputeSurfWeighted(SPARTA *sparta, int narg, char **arg) :
  ComputeSurf(sparta, narg, arg)
{
  // ComputeSurf has parsed group, mixture, and the value keywords into
  // which[]. Restrict to the count/flux keywords — the ones for which a
  // pweight-weighted surface flux is meaningful.
  for (int i = 0; i < nvalue; i++)
    if (which[i] > MFLUXIN)
      error->all(FLERR,
        "compute surf/weighted supports only num, numwt, nflux, "
        "nflux_incident, mflux, mflux_incident");

  pweight_index = -1;
  pweight_ewhich = -1;
}

/* ---------------------------------------------------------------------- */

void ComputeSurfWeighted::init()
{
  ComputeSurf::init();

  pweight_index = particle->find_custom((char *) "pweight");
  if (pweight_index < 0)
    error->all(FLERR,
      "compute surf/weighted requires the pweight custom "
      "(add fix particle/weight)");
  pweight_ewhich = particle->ewhich[pweight_index];
}

/* ----------------------------------------------------------------------
   tally values for a single particle-surf collision, weighted by pweight
   iorig = incident particle (pre-collision copy; carries no custom data)
   ip,jp = particles after collision (ip = &particles[i], jp = new particle)
   The pweight is a per-particle custom keyed by particle index, so read it
   from ip (same index as the incident particle) rather than the iorig copy.
------------------------------------------------------------------------- */

void ComputeSurfWeighted::surf_tally(double /*dtremain*/, int isurf,
                                     int /*icell*/, int reaction,
                                     Particle::OnePart *iorig,
                                     Particle::OnePart *ip,
                                     Particle::OnePart *jp)
{
  // incidence-only: emission (iorig==NULL) is measured by erosion_flux
  if (!iorig) return;

  // skip if isurf not in surface group
  if (dim == 2) { if (!(lines[isurf].mask & groupbit)) return; }
  else          { if (!(tris[isurf].mask & groupbit)) return; }

  // skip if incident/emitting species not in mixture group
  int origspecies = -1, igroup;
  if (iorig) {
    origspecies = iorig->ispecies;
    igroup = particle->mixture[imix]->species2group[origspecies];
  } else {
    igroup = particle->mixture[imix]->species2group[ip->ispecies];
  }
  if (igroup < 0) return;

  // w_in = incident pweight (mover stamps it, so absorbed W with ip=NULL is
  // still weighted right); pw_ip/pw_jp = outgoing pweight by particle index.
  const double w_in = update->tally_pweight;
  double *pweight = particle->edvec[pweight_ewhich];
  const double pw_ip = ip ? pweight[ip - particle->particles] : 0.0;
  const double pw_jp = jp ? pweight[jp - particle->particles] : 0.0;

  // itally = tally index of isurf (add to hash on first hit)
  int itally;
  double *vec;
  surfint surfID = (dim == 2) ? lines[isurf].id : tris[isurf].id;
  int transparent = (dim == 2) ? lines[isurf].transparent
                               : tris[isurf].transparent;
  if (hash->find(surfID) != hash->end()) itally = (*hash)[surfID];
  else {
    if (ntally == maxtally) grow_tally();
    itally = ntally;
    (*hash)[surfID] = itally;
    tally2surf[itally] = surfID;
    vec = array_surf_tally[itally];
    for (int i = 0; i < ntotal; i++) vec[i] = 0.0;
    ntally++;
  }

  // normflux embeds fnum (nfactor = dt/fnum); divide it back out because the
  // pweights already ARE real-particle counts (flux = weight/(line_size*dt)).
  const double fluxscale = normflux[isurf] / update->fnum;
  const double m_in = (origspecies >= 0)
    ? particle->species[origspecies].mass * w_in : 0.0;
  const double m_ip = ip ? particle->species[ip->ispecies].mass * pw_ip : 0.0;
  const double m_jp = jp ? particle->species[jp->ispecies].mass * pw_jp : 0.0;

  vec = array_surf_tally[itally];
  int k = igroup * nvalue;

  for (int m = 0; m < nvalue; m++) {
    switch (which[m]) {
    case NUM:                              // count of incidence events
      if (iorig) vec[k] += 1.0;
      k++;
      break;
    case NUMWT:                            // incident real-particle count
      vec[k++] += w_in;
      break;
    case NFLUX:                            // DEPOSITION = incident - reflected
      vec[k] += w_in * fluxscale;
      if (!transparent) {
        if (ip) vec[k] -= pw_ip * fluxscale;
        if (jp) vec[k] -= pw_jp * fluxscale;
      }
      k++;
      break;
    case NFLUXIN:                          // GROSS wall load (every incidence)
      vec[k] += w_in * fluxscale;
      k++;
      break;
    case MFLUX:                            // mass DEPOSITION
      vec[k] += m_in * fluxscale;
      if (!transparent) {
        if (ip) vec[k] -= m_ip * fluxscale;
        if (jp) vec[k] -= m_jp * fluxscale;
      }
      k++;
      break;
    case MFLUXIN:                          // gross mass load
      vec[k] += m_in * fluxscale;
      k++;
      break;
    }
  }
}
