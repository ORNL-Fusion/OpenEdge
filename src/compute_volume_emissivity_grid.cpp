/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.

    compute volume/emissivity/grid — per-grid volumetric line emissivity
      emissivity = ne * nz * PEC(Te, ne)   [photons/m^3/s/sr]

    Syntax:
      compute ID volume/emissivity/grid group-ID mixture-ID \
              pec <elem> <pec_id> <line_key> plasma_compute CID \
              [pec_units cm3s|m3s]

    Reads `/volume/pec/<elem>/<pec_id>/<line_key>/{coefficient,temperature,
    density}` from `database/processes.h5` via the ProcessLibrary loader.
    The coefficient dataset is log10(cm^3/s) or log10(m^3/s) per pec_units.
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_volume_emissivity_grid.h"
#include "compute_plasma_fields.h"
#include "particle.h"
#include "mixture.h"
#include "grid.h"
#include "domain.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "process_library.h"
#include "database_paths.h"

#include <cmath>
#include <algorithm>

using namespace SPARTA_NS;

#define MAXACCUMULATE 1

/* ---------------------------------------------------------------------- */

ComputeVolumeEmissivityGrid::ComputeVolumeEmissivityGrid(
    SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 9)
    error->all(FLERR,
      "Illegal compute volume/emissivity/grid command\n"
      "Usage: compute ID volume/emissivity/grid group mix "
      "pec <elem> <pec_id> <line> plasma_compute CID");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0)
    error->all(FLERR,
      "Compute volume/emissivity/grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  imix = particle->find_mixture(arg[3]);
  if (imix < 0)
    error->all(FLERR,
      "Compute volume/emissivity/grid mixture ID does not exist");
  ngroup = particle->mixture[imix]->ngroup;

  // parse keyword pairs
  std::string pec_elem, pec_id, pec_line;
  plasma_compute_id = NULL;
  pec_unit_conv = 1.0e-6;  // default: ADAS cm^3/s -> m^3/s

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"pec") == 0) {
      if (iarg+3 >= narg)
        error->all(FLERR,"compute volume/emissivity/grid: "
                   "pec needs <elem> <pec_id> <line>");
      pec_elem = arg[iarg+1];
      pec_id   = arg[iarg+2];
      pec_line = arg[iarg+3];
      iarg += 4;
    } else if (strcmp(arg[iarg],"plasma_compute") == 0) {
      if (iarg+1 >= narg)
        error->all(FLERR,"compute volume/emissivity/grid: "
                   "missing value after plasma_compute");
      int n = strlen(arg[iarg+1]) + 1;
      plasma_compute_id = new char[n];
      strcpy(plasma_compute_id, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"pec_units") == 0) {
      if (iarg+1 >= narg)
        error->all(FLERR,"compute volume/emissivity/grid: "
                   "missing value after pec_units");
      if (strcmp(arg[iarg+1],"cm3s") == 0)
        pec_unit_conv = 1.0e-6;       // cm^3/s -> m^3/s
      else if (strcmp(arg[iarg+1],"m3s") == 0)
        pec_unit_conv = 1.0;           // already m^3/s
      else
        error->all(FLERR,"compute volume/emissivity/grid: "
                   "pec_units must be cm3s or m3s");
      iarg += 2;
    } else {
      error->all(FLERR,"compute volume/emissivity/grid: "
                 "unknown keyword");
    }
  }

  if (pec_elem.empty() || pec_id.empty() || pec_line.empty())
    error->all(FLERR,
      "compute volume/emissivity/grid requires 'pec <elem> <pec_id> <line>'");
  if (!plasma_compute_id)
    error->all(FLERR,
      "compute volume/emissivity/grid requires plasma_compute");

  // tally setup: one accumulator (WCOUNT) per group
  npergroup = 1;
  ntotal = ngroup * npergroup;

  nmap = new int[ngroup];
  memory->create(map, ngroup, MAXACCUMULATE,
                 "volume_emissivity:map");
  for (int ig = 0; ig < ngroup; ig++) {
    nmap[ig] = 1;
    map[ig][0] = ig * npergroup;
  }

  per_grid_flag = 1;
  size_per_grid_cols = ngroup;
  post_process_grid_flag = 1;

  nglocal = 0;
  vector_grid = NULL;
  tally = NULL;

  pweight_index = -1;
  pweight_ewhich = -1;
  cp_plasma = NULL;

  // Load PEC table from database/processes.h5 via ProcessLibrary.
  // The loader does rank-0-read + MPI_Bcast internally.
  pec_nte = pec_nne = 0;
  std::string processes_path = resolve_processes_file();
  if (processes_path.empty())
    error->all(FLERR,
      "compute volume/emissivity/grid: cannot locate processes.h5 "
      "(set OPENEDGE_ROOT or build with -DOPENEDGE_DATABASE_DIR)");
  ProcessLibrary plib;
  plib.open(processes_path, world, error);
  ProcessLibrary::PecTable pec;
  if (!plib.load_pec_line(pec_elem, pec_id, pec_line, pec)) {
    char msg[512];
    snprintf(msg, sizeof(msg),
      "compute volume/emissivity/grid: /volume/pec/%s/%s/%s not found in %s",
      pec_elem.c_str(), pec_id.c_str(), pec_line.c_str(),
      processes_path.c_str());
    error->all(FLERR, msg);
  }
  // ProcessLibrary returns log10(temperature) + log10(density) grids
  // and coefficient already in log10 form.  Unit conversion shifts the
  // additive constant: log10(x * conv) = log10(x) + log10(conv).
  pec_nte = pec.nT;
  pec_nne = pec.nNe;
  pec_log_te = std::move(pec.logT);
  pec_log_ne = std::move(pec.logN);
  pec_log_val.assign(pec.coef.size(), 0.0);
  const double log_conv = std::log10(pec_unit_conv);
  for (size_t i = 0; i < pec.coef.size(); i++)
    pec_log_val[i] = pec.coef[i] + log_conv;

  if (comm->me == 0 && screen) {
    fprintf(screen,
      "  PEC table: %s/%s/%s  %d Te x %d ne  "
      "log10Te=[%.2f,%.2f]  log10ne=[%.2f,%.2f]\n",
      pec_elem.c_str(), pec_id.c_str(), pec_line.c_str(),
      pec_nte, pec_nne,
      pec_log_te.front(), pec_log_te.back(),
      pec_log_ne.front(), pec_log_ne.back());
  }
}

/* ---------------------------------------------------------------------- */

ComputeVolumeEmissivityGrid::~ComputeVolumeEmissivityGrid()
{
  if (copymode) return;
  delete [] plasma_compute_id;
  delete [] nmap;
  memory->destroy(map);
  memory->destroy(vector_grid);
  memory->destroy(tally);
}

/* ---------------------------------------------------------------------- */

void ComputeVolumeEmissivityGrid::init()
{
  // verify pweight custom attribute
  pweight_index = particle->find_custom((char *) "pweight");
  if (pweight_index < 0)
    error->all(FLERR,
      "Compute volume/emissivity/grid requires fix particle/weight");
  pweight_ewhich = particle->ewhich[pweight_index];

  // verify mixture ngroup unchanged
  if (ngroup != particle->mixture[imix]->ngroup)
    error->all(FLERR,
      "Number of groups in volume/emissivity/grid mixture has changed");

  // find plasma compute
  int ic = modify->find_compute(plasma_compute_id);
  if (ic < 0)
    error->all(FLERR,
      "Compute volume/emissivity/grid: plasma_compute ID not found");
  cp_plasma = dynamic_cast<ComputePlasmaFields *>(modify->compute[ic]);
  if (!cp_plasma)
    error->all(FLERR,
      "Compute volume/emissivity/grid: plasma_compute is not "
      "a plasma/fields compute");

  reallocate();
}

/* ----------------------------------------------------------------------
   tally pweight per cell per species group
------------------------------------------------------------------------- */

void ComputeVolumeEmissivityGrid::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;

  Grid::ChildInfo *cinfo = grid->cinfo;
  Particle::OnePart *particles = particle->particles;
  int *s2g = particle->mixture[imix]->species2group;
  int nlocal = particle->nlocal;

  pweight_ewhich = particle->ewhich[pweight_index];
  double *pweight_dvec = particle->edvec[pweight_ewhich];

  int i,ispecies,igroup,icell;

  for (i = 0; i < nglocal; i++)
    for (int j = 0; j < ntotal; j++)
      tally[i][j] = 0.0;

  for (i = 0; i < nlocal; i++) {
    ispecies = particles[i].ispecies;
    igroup = s2g[ispecies];
    if (igroup < 0) continue;
    icell = particles[i].icell;
    if (!(cinfo[icell].mask & groupbit)) continue;

    tally[icell][igroup * npergroup] += pweight_dvec[i];
  }
}

/* ---------------------------------------------------------------------- */

int ComputeVolumeEmissivityGrid::query_tally_grid(
    int index, double **&array, int *&cols)
{
  index--;
  array = tally;
  cols = map[index];
  return nmap[index];
}

/* ----------------------------------------------------------------------
   post-process: emissivity = ne * nz * PEC(Te,ne)
   nz = sum(pweight) * cellweight / volume
------------------------------------------------------------------------- */

void ComputeVolumeEmissivityGrid::post_process_grid(
    int index, int nsample,
    double **etally, int *emap, double *vec, int nstride)
{
  index--;

  int lo = 0;
  int hi = nglocal;
  int k = 0;

  if (!etally) {
    nsample = 1;
    etally = tally;
    emap = map[index];
    vec = vector_grid;
    nstride = 1;
  }

  Grid::ChildInfo *cinfo = grid->cinfo;
  Grid::ChildCell *cells = grid->cells;
  int wcount_col = emap[0];

  for (int icell = lo; icell < hi; icell++) {
    double vol = cinfo[icell].volume;
    double wsum = etally[icell][wcount_col];

    if (vol == 0.0 || wsum == 0.0) {
      vec[k] = 0.0;
      k += nstride;
      continue;
    }

    // cell center for plasma query
    double xc[3];
    xc[0] = 0.5 * (cells[icell].lo[0] + cells[icell].hi[0]);
    xc[1] = 0.5 * (cells[icell].lo[1] + cells[icell].hi[1]);
    xc[2] = 0.5 * (cells[icell].lo[2] + cells[icell].hi[2]);

    PlasmaFileParams pf = cp_plasma->query_plasma_at_point(xc);
    double Te = pf.temp_e;
    double ne = pf.dens_e;

    if (Te <= 0.0 || ne <= 0.0) {
      vec[k] = 0.0;
      k += nstride;
      continue;
    }

    // PEC interpolation in log-log space
    double log_pec = interpolatePEC(std::log10(Te), std::log10(ne));
    double pec_val = std::pow(10.0, log_pec);   // m^3/s

    // weighted number density
    double wt = cinfo[icell].weight / vol;
    double nz = wt * wsum / nsample;

    // emissivity [photons/m^3/s/sr]
    vec[k] = ne * nz * pec_val;
    k += nstride;
  }
}

/* ----------------------------------------------------------------------
   bilinear interpolation in log10(Te)-log10(ne) space
   returns log10(PEC)
------------------------------------------------------------------------- */

double ComputeVolumeEmissivityGrid::interpolatePEC(
    double log_te, double log_ne) const
{
  // clamp to table bounds
  double te_lo = pec_log_te.front();
  double te_hi = pec_log_te.back();
  double ne_lo = pec_log_ne.front();
  double ne_hi = pec_log_ne.back();

  if (log_te <= te_lo) log_te = te_lo;
  if (log_te >= te_hi) log_te = te_hi;
  if (log_ne <= ne_lo) log_ne = ne_lo;
  if (log_ne >= ne_hi) log_ne = ne_hi;

  // find bracketing indices for Te
  int ite = 0;
  for (int i = 0; i < pec_nte - 1; i++) {
    if (pec_log_te[i+1] >= log_te) { ite = i; break; }
  }
  if (log_te >= te_hi) ite = pec_nte - 2;

  // find bracketing indices for ne
  int ine = 0;
  for (int i = 0; i < pec_nne - 1; i++) {
    if (pec_log_ne[i+1] >= log_ne) { ine = i; break; }
  }
  if (log_ne >= ne_hi) ine = pec_nne - 2;

  double t = 0.0, u = 0.0;
  double dt = pec_log_te[ite+1] - pec_log_te[ite];
  double dn = pec_log_ne[ine+1] - pec_log_ne[ine];
  if (dt > 0.0) t = (log_te - pec_log_te[ite]) / dt;
  if (dn > 0.0) u = (log_ne - pec_log_ne[ine]) / dn;

  // four corners: pec_log_val is [nte x nne] row-major
  double q00 = pec_log_val[ite     * pec_nne + ine    ];
  double q10 = pec_log_val[(ite+1) * pec_nne + ine    ];
  double q01 = pec_log_val[ite     * pec_nne + (ine+1)];
  double q11 = pec_log_val[(ite+1) * pec_nne + (ine+1)];

  return (1-t)*(1-u)*q00 + t*(1-u)*q10 + (1-t)*u*q01 + t*u*q11;
}

/* ---------------------------------------------------------------------- */

void ComputeVolumeEmissivityGrid::reallocate()
{
  if (grid->nlocal == nglocal) return;

  memory->destroy(vector_grid);
  memory->destroy(tally);
  nglocal = grid->nlocal;
  memory->create(vector_grid, nglocal, "volume_emissivity:vector_grid");
  memory->create(tally, nglocal, ntotal, "volume_emissivity:tally");
}

/* ---------------------------------------------------------------------- */

bigint ComputeVolumeEmissivityGrid::memory_usage()
{
  bigint bytes = 0;
  bytes += nglocal * sizeof(double);
  bytes += (bigint)ntotal * nglocal * sizeof(double);
  bytes += (pec_nte + pec_nne + pec_nte * pec_nne) * sizeof(double);
  return bytes;
}
