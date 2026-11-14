/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
------------------------------------------------------------------------- */

#include "mpi.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "fix_dt_monitor.h"
#include "sparta_masks.h"
#include "fix_volume_chem_adas.h"
#include "pusher.h"
#include "update.h"
#include "particle.h"
#include "modify.h"
#include "surf.h"
#include "surf_react.h"
#include "comm.h"
#include "error.h"

using namespace SPARTA_NS;

enum{INT,DOUBLE};                      // several files
enum{GWH,GBAD,GMAX,GGCA,CW,CBAD,CMAX,NCOL};   // columns of the per-species tallies

/* ---------------------------------------------------------------------- */

FixDtMonitor::FixDtMonitor(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal fix dt/monitor command");
  nevery = atoi(arg[2]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix dt/monitor command");

  vector_flag = 1;
  size_vector = 4;
  global_freq = nevery;

  // This fix only switches the tallies on and reports them; it never reads
  // or writes particle data. Without empty masks the Kokkos build copies
  // all particles to the host and back at every step (measured 2026-11-14:
  // 2.4 -> 5.8 ms per step on 4 A100).
  datamask_read = datamask_modify = EMPTY_MASK;

  for (int m = 0; m < modify->nfix; m++)
    if (strcmp(modify->fix[m]->style,"dt/monitor") == 0)
      error->all(FLERR,"Only one fix dt/monitor can be defined");

  MPI_Comm_rank(world,&me);
  nsp = 0;
  for (int i = 0; i < 4; i++) vec[i] = -1.0;
}

/* ---------------------------------------------------------------------- */

FixDtMonitor::~FixDtMonitor()
{
  // the tallies belong to the pusher and the chemistry fixes, which may
  // already be gone: nothing is touched here. They are armed one step at a
  // time (start_of_step) and stop counting once this fix no longer does it.
}

/* ---------------------------------------------------------------------- */

int FixDtMonitor::setmask()
{
  return START_OF_STEP | END_OF_STEP;
}

/* ----------------------------------------------------------------------
   (re)size and zero the tallies kept by the pusher and the chemistry fixes
------------------------------------------------------------------------- */

void FixDtMonitor::allocate()
{
  nsp = particle->nspecies;
  Pusher *p = update->pusher;
  if (p) {
    delete [] p->dtm_wh; delete [] p->dtm_whbad;
    delete [] p->dtm_max; delete [] p->dtm_whgca;
    p->dtm_wh = new double[nsp]();    p->dtm_whbad = new double[nsp]();
    p->dtm_max = new double[nsp]();   p->dtm_whgca = new double[nsp]();
    p->dtm_nonfinite = 0.0;
    p->dtm_last = -1;
    p->dtm_nsp = nsp;
  }
  for (FixVolumeChemAdas *c : chem) {
    delete [] c->dtm_w; delete [] c->dtm_wbad; delete [] c->dtm_max;
    c->dtm_w = new double[nsp]();     c->dtm_wbad = new double[nsp]();
    c->dtm_max = new double[nsp]();
    c->dtm_nsat = c->dtm_nonfinite = 0.0;
    c->dtm_nsp = nsp;
  }
}

/* ---------------------------------------------------------------------- */

void FixDtMonitor::init()
{
  // a report must follow the chemistry of its step: fixes run in input order
  chem.clear();
  int seen = 0;
  for (int m = 0; m < modify->nfix; m++) {
    if (modify->fix[m] == this) seen = 1;
    FixVolumeChemAdas *c = dynamic_cast<FixVolumeChemAdas *>(modify->fix[m]);
    if (!c) continue;
    if (seen)
      error->all(FLERR,"Fix dt/monitor must be defined after fix volume/chem/adas");
    chem.push_back(c);
  }

  // the tallies of a run start empty, on the host and on the device
  update->dtm_sync();
  for (FixVolumeChemAdas *c : chem) c->dtm_sync();
  allocate();
  Pusher *p = update->pusher;
  if (p) {
    const int ipw = particle->find_custom((char *) "pweight");
    p->dtm_pw = (ipw >= 0) ? particle->ewhich[ipw] : -1;
    int iat = particle->find_custom((char *) "dtm_push");
    if (iat < 0) iat = particle->add_custom((char *) "dtm_push",DOUBLE,0);
    p->dtm_attr = particle->ewhich[iat];
  }

  run.assign(NCOL*nsp,0.0);
  for (int i = 0; i < 4; i++) run_bad[i] = 0.0;
  for (int i = 0; i < 4; i++) vec[i] = -1.0;
  win_first = run_first = update->ntimestep + 1;
  last_report = update->ntimestep;
  warned_gyro = warned_chem = 0;
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   arm the tallies for this step
------------------------------------------------------------------------- */

void FixDtMonitor::start_of_step()
{
  Pusher *p = update->pusher;
  if (p) { p->dtm_on = 1; p->dtm_until = update->ntimestep; }
  for (FixVolumeChemAdas *c : chem) { c->dtm_on = 1; c->dtm_until = update->ntimestep; }
}

/* ---------------------------------------------------------------------- */

void FixDtMonitor::end_of_step()
{
  if (update->ntimestep % nevery == 0) report();
}

/* ----------------------------------------------------------------------
   close the reporting window: reduce the tallies over ranks, fold them
   into the run totals, set the global vector, warn the first time a
   criterion is exceeded
------------------------------------------------------------------------- */

void FixDtMonitor::report()
{
  Pusher *p = update->pusher;
  update->dtm_sync();
  for (FixVolumeChemAdas *c : chem) c->dtm_sync();

  // sums and maxima are reduced separately; fractions are formed afterwards
  const int n = NCOL*nsp;
  std::vector<double> sum(n+3,0.0),mx(n,0.0),gsum(n+3,0.0),gmx(n,0.0);
  for (int s = 0; s < nsp; s++) {
    if (p && p->dtm_wh) {
      sum[NCOL*s+GWH] = p->dtm_wh[s];   sum[NCOL*s+GBAD] = p->dtm_whbad[s];
      sum[NCOL*s+GGCA] = p->dtm_whgca[s];
      mx[NCOL*s+GMAX] = p->dtm_max[s];
      p->dtm_wh[s] = p->dtm_whbad[s] = p->dtm_whgca[s] = p->dtm_max[s] = 0.0;
    }
    for (FixVolumeChemAdas *c : chem) {
      sum[NCOL*s+CW] += c->dtm_w[s];    sum[NCOL*s+CBAD] += c->dtm_wbad[s];
      if (c->dtm_max[s] > mx[NCOL*s+CMAX]) mx[NCOL*s+CMAX] = c->dtm_max[s];
      c->dtm_w[s] = c->dtm_wbad[s] = c->dtm_max[s] = 0.0;
    }
  }
  if (p) { sum[n] = p->dtm_nonfinite; p->dtm_nonfinite = 0.0; p->dtm_last = -1; }
  for (FixVolumeChemAdas *c : chem) {
    sum[n+1] += c->dtm_nonfinite; sum[n+2] += c->dtm_nsat;
    c->dtm_nonfinite = c->dtm_nsat = 0.0;
  }
  MPI_Allreduce(sum.data(),gsum.data(),n+3,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(mx.data(),gmx.data(),n,MPI_DOUBLE,MPI_MAX,world);

  double g[NCOL];
  for (int k = 0; k < NCOL; k++) g[k] = 0.0;
  int sgyro = -1, schem = -1;
  for (int s = 0; s < nsp; s++) {
    for (int k = 0; k < NCOL; k++) {
      const double v = (k == GMAX || k == CMAX) ? gmx[NCOL*s+k] : gsum[NCOL*s+k];
      if (k == GMAX || k == CMAX) {
        if (v > run[NCOL*s+k]) run[NCOL*s+k] = v;
        if (v > g[k]) { g[k] = v; if (k == GMAX) sgyro = s; else schem = s; }
      } else {
        run[NCOL*s+k] += v;
        g[k] += v;
      }
    }
  }
  for (int i = 0; i < 3; i++) run_bad[i] += gsum[n+i];

  // -1 = not assessed in this window (nothing pushed / no update, or
  // non-finite values): never a satisfied zero
  vec[0] = (g[GWH] > 0.0 && gsum[n] == 0.0) ? g[GMAX] : -1.0;
  vec[1] = (g[GWH] > 0.0 && gsum[n] == 0.0) ? g[GBAD]/g[GWH] : -1.0;
  vec[2] = (g[CW] > 0.0 && gsum[n+1] == 0.0) ? g[CMAX] : -1.0;
  vec[3] = (g[CW] > 0.0 && gsum[n+1] == 0.0) ? g[CBAD]/g[CW] : -1.0;

  if (me == 0) {
    char str[512];
    if (gsum[n] > 0.0 || gsum[n+1] > 0.0)
      error->warning(FLERR,"Fix dt/monitor: non-finite gyro or reaction "
                     "values in this window, not assessed");
    if (g[GBAD] > 0.0 && !warned_gyro && p) {
      warned_gyro = 1;
      const double lim = p->pusher_bad_dt_limit;
      snprintf(str,512,"Fix dt/monitor: orbit resolution limit exceeded: "
               "|q/m||B|dt_sub = %.3g (species %s), advisory limit %g, for "
               "%.3g of the Boris exposure in steps " BIGINT_FORMAT "-"
               BIGINT_FORMAT ". Increase global pusher subcycles (now %d; %d "
               "subcycles would meet it for the fields seen so far) or reduce "
               "the timestep; this addresses gyromotion only",
               g[GMAX],particle->species[sgyro].id,lim,g[GBAD]/g[GWH],
               win_first,update->ntimestep,p->pusher_subcycles,
               static_cast<int>(ceil(p->pusher_subcycles*g[GMAX]/lim)));
      error->warning(FLERR,str);
    }
    if (g[CBAD] > 0.0 && !warned_chem) {
      warned_chem = 1;
      snprintf(str,512,"Fix dt/monitor: reaction hazard limit exceeded: "
               "rate x interval = %.3g (species %s), advisory limit %g, for "
               "%.3g of the updated weight in steps " BIGINT_FORMAT "-"
               BIGINT_FORMAT ". Reduce the timestep or the interval of fix "
               "volume/chem/adas (Boris subcycles do not change it)",
               g[CMAX],particle->species[schem].id,chem[0]->dtm_limit,
               g[CBAD]/g[CW],win_first,update->ntimestep);
      error->warning(FLERR,str);
    }
  }

  last_report = update->ntimestep;
  win_first = update->ntimestep + 1;
}

/* ----------------------------------------------------------------------
   end of run: close a partial window, print the run table
------------------------------------------------------------------------- */

void FixDtMonitor::post_run()
{
  if (update->ntimestep > last_report) report();
  if (me || update->ntimestep < run_first) return;   // nothing for a run of 0 steps

  Pusher *p = update->pusher;
  FILE *out[2] = {screen,logfile};
  for (int io = 0; io < 2; io++) {
    FILE *fp = out[io];
    if (!fp) continue;
    fprintf(fp,"\nTimestep monitor (fix %s), steps " BIGINT_FORMAT "-"
            BIGINT_FORMAT ":\n",id,run_first,update->ntimestep);

    double den = 0.0, bad = 0.0, gca = 0.0, mx = 0.0;
    fprintf(fp,"  Gyromotion: |q/m||B|dt_sub in the Boris push, advisory level %g, "
            "%d subcycle(s)\n",p ? p->pusher_bad_dt_limit : 0.0,
            p ? p->pusher_subcycles : 0);
    fprintf(fp,"    %-10s %12s %14s %18s %18s\n","species","max",
            "above (approx)","assessed (atom s)","gc, not assessed");
    for (int s = 0; s < nsp; s++) {
      const double *r = &run[NCOL*s];
      if (r[GWH] <= 0.0 && r[GGCA] <= 0.0) continue;
      if (r[GWH] > 0.0)
        fprintf(fp,"    %-10s %12.6g %14.6g %18.8g %18.8g\n",particle->species[s].id,
                r[GMAX],r[GBAD]/r[GWH],r[GWH],r[GGCA]);
      else
        fprintf(fp,"    %-10s %12s %14s %18.8g %18.8g\n",particle->species[s].id,
                "-","-",0.0,r[GGCA]);
      den += r[GWH]; bad += r[GBAD]; gca += r[GGCA];
      if (r[GMAX] > mx) mx = r[GMAX];
    }
    if (run_bad[0] > 0.0)
      fprintf(fp,"    status: INVALID, %g non-finite value(s)\n",run_bad[0]);
    else if (den <= 0.0)
      fprintf(fp,"    status: not assessed (no Boris push of a charged particle)\n");
    else if (bad > 0.0)
      fprintf(fp,"    status: EXCEEDED for %.4g of the exposure, max %.4g\n",bad/den,mx);
    else
      fprintf(fp,"    status: below the advisory level everywhere assessed, max %.4g\n",mx);
    if (gca > 0.0)
      fprintf(fp,"    %.4g of the charged exposure was pushed as guiding centres "
              "and is not assessed\n",gca/(gca+den));
    if (den > 0.0)
      fprintf(fp,"    the exposure and its fraction above are approximate: a wall "
              "hit removes the time left in the step in proportion to the whole "
              "push, an exit through an open boundary removes nothing\n");

    den = bad = mx = 0.0;
    fprintf(fp,"  Reactions: rate x interval in fix volume/chem/adas, advisory "
            "level %g, interval %g s\n",chem.size() ? chem[0]->dtm_limit : 0.0,
            chem.size() ? chem[0]->nevery*update->dt : 0.0);
    fprintf(fp,"    %-10s %12s %14s %18s\n","species","max","above (frac)",
            "assessed (atoms)");
    for (int s = 0; s < nsp; s++) {
      const double *r = &run[NCOL*s];
      if (r[CW] <= 0.0) continue;
      fprintf(fp,"    %-10s %12.6g %14.6g %18.8g\n",particle->species[s].id,
              r[CMAX],r[CBAD]/r[CW],r[CW]);
      den += r[CW]; bad += r[CBAD];
      if (r[CMAX] > mx) mx = r[CMAX];
    }
    if (run_bad[1] > 0.0)
      fprintf(fp,"    status: INVALID, %g non-finite value(s)\n",run_bad[1]);
    else if (den <= 0.0)
      fprintf(fp,"    status: not assessed (no chemistry update)\n");
    else if (bad > 0.0)
      fprintf(fp,"    status: EXCEEDED for %.4g of the updated weight, max %.4g\n",bad/den,mx);
    else
      fprintf(fp,"    status: below the advisory level everywhere assessed, max %.4g\n",mx);
    if (run_bad[2] > 0.0)
      fprintf(fp,"    %g update(s) had a channel at the cap of 50: the max is a "
              "lower bound\n",run_bad[2]);
    // active processes without a check yet: said so, never implied resolved
    int ncoul = 0, ndiff = 0, npwi = 0;
    for (int m = 0; m < modify->nfix; m++) {
      if (strncmp(modify->fix[m]->style,"coulomb/",8) == 0) ncoul++;
      if (strncmp(modify->fix[m]->style,"cross_field_diffusion",21) == 0) ndiff++;
    }
    for (int m = 0; m < surf->nsr; m++)
      if (strncmp(surf->sr[m]->style,"surface/pwi",11) == 0) npwi++;
    if (ncoul) fprintf(fp,"  Coulomb collisions: not assessed (no check yet)\n");
    if (ndiff) fprintf(fp,"  Cross-field diffusion near walls: not assessed (no check yet)\n");
    if (update->sheath_flag)
      fprintf(fp,"  Sheath traversal (%s model): not assessed (no check yet)\n",
              update->sheath_boundary ? "boundary" : (update->sheath_kick ? "kick" : "spatial"));
    if (npwi) fprintf(fp,"  Surface evolution: not assessed (no check yet)\n");
    fprintf(fp,"  The advisory levels flag a clearly coarse step; they are not "
            "an accuracy guarantee.\n\n");
  }
}

/* ----------------------------------------------------------------------
   last reporting window: 1 max |q/m||B|dt_sub, 2 fraction of the Boris
   exposure above the level, 3 max rate x interval, 4 fraction of the
   updated weight above the level; -1 = not assessed
------------------------------------------------------------------------- */

double FixDtMonitor::compute_vector(int i)
{
  return vec[i];
}
