/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.

    compute photon_emissivity/grid — per-grid volumetric photon emissivity
      emissivity = ne * nz * PEC(Te, ne)   [photons/m^3/s/sr]

    Syntax:
      compute ID photon_emissivity/grid group-ID mixture-ID \
              pec_file PATH plasma_compute CID

    PEC HDF5 file layout (flexible dataset names):
      te or te_grid   (N_te,)       Te in eV
      ne or ne_grid   (N_ne,)       ne in m^-3
      <anything else> (N_te, N_ne)  PEC values (the 2D dataset)

    Optional keywords:
      pec_units cm3s   — PEC in cm^3/s (default, ADAS convention)
      pec_units m3s    — PEC already in m^3/s
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_photon_emissivity_grid.h"
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

#include <cmath>
#include <algorithm>
#include <H5Cpp.h>

using namespace SPARTA_NS;

#define MAXACCUMULATE 1

/* ---------------------------------------------------------------------- */

ComputePhotonEmissivityGrid::ComputePhotonEmissivityGrid(
    SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 8)
    error->all(FLERR,
      "Illegal compute photon_emissivity/grid command\n"
      "Usage: compute ID photon_emissivity/grid group mix "
      "pec_file PATH plasma_compute CID");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0)
    error->all(FLERR,
      "Compute photon_emissivity/grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  imix = particle->find_mixture(arg[3]);
  if (imix < 0)
    error->all(FLERR,
      "Compute photon_emissivity/grid mixture ID does not exist");
  ngroup = particle->mixture[imix]->ngroup;

  // parse keyword pairs
  std::string pec_path;
  plasma_compute_id = NULL;
  pec_unit_conv = 1.0e-6;  // default: ADAS cm^3/s -> m^3/s

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"pec_file") == 0) {
      if (iarg+1 >= narg)
        error->all(FLERR,"compute photon_emissivity/grid: "
                   "missing value after pec_file");
      pec_path = arg[iarg+1];
      iarg += 2;
    } else if (strcmp(arg[iarg],"plasma_compute") == 0) {
      if (iarg+1 >= narg)
        error->all(FLERR,"compute photon_emissivity/grid: "
                   "missing value after plasma_compute");
      int n = strlen(arg[iarg+1]) + 1;
      plasma_compute_id = new char[n];
      strcpy(plasma_compute_id, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"pec_units") == 0) {
      if (iarg+1 >= narg)
        error->all(FLERR,"compute photon_emissivity/grid: "
                   "missing value after pec_units");
      if (strcmp(arg[iarg+1],"cm3s") == 0)
        pec_unit_conv = 1.0e-6;       // cm^3/s -> m^3/s
      else if (strcmp(arg[iarg+1],"m3s") == 0)
        pec_unit_conv = 1.0;           // already m^3/s
      else
        error->all(FLERR,"compute photon_emissivity/grid: "
                   "pec_units must be cm3s or m3s");
      iarg += 2;
    } else {
      error->all(FLERR,"compute photon_emissivity/grid: "
                 "unknown keyword");
    }
  }

  if (pec_path.empty())
    error->all(FLERR,"compute photon_emissivity/grid requires pec_file");
  if (!plasma_compute_id)
    error->all(FLERR,
      "compute photon_emissivity/grid requires plasma_compute");

  // tally setup: one accumulator (WCOUNT) per group
  npergroup = 1;
  ntotal = ngroup * npergroup;

  nmap = new int[ngroup];
  memory->create(map, ngroup, MAXACCUMULATE,
                 "photon_emissivity:map");
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

  // read PEC table (rank 0 reads, broadcasts)
  pec_nte = pec_nne = 0;
  readPECFile(pec_path);
  broadcastPECData();
}

/* ---------------------------------------------------------------------- */

ComputePhotonEmissivityGrid::~ComputePhotonEmissivityGrid()
{
  if (copymode) return;
  delete [] plasma_compute_id;
  delete [] nmap;
  memory->destroy(map);
  memory->destroy(vector_grid);
  memory->destroy(tally);
}

/* ---------------------------------------------------------------------- */

void ComputePhotonEmissivityGrid::init()
{
  // verify pweight custom attribute
  pweight_index = particle->find_custom((char *) "pweight");
  if (pweight_index < 0)
    error->all(FLERR,
      "Compute photon_emissivity/grid requires fix particle/weight");
  pweight_ewhich = particle->ewhich[pweight_index];

  // verify mixture ngroup unchanged
  if (ngroup != particle->mixture[imix]->ngroup)
    error->all(FLERR,
      "Number of groups in photon_emissivity/grid mixture has changed");

  // find plasma compute
  int ic = modify->find_compute(plasma_compute_id);
  if (ic < 0)
    error->all(FLERR,
      "Compute photon_emissivity/grid: plasma_compute ID not found");
  cp_plasma = dynamic_cast<ComputePlasmaFields *>(modify->compute[ic]);
  if (!cp_plasma)
    error->all(FLERR,
      "Compute photon_emissivity/grid: plasma_compute is not "
      "a plasma/fields compute");

  reallocate();
}

/* ----------------------------------------------------------------------
   tally pweight per cell per species group
------------------------------------------------------------------------- */

void ComputePhotonEmissivityGrid::compute_per_grid()
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

int ComputePhotonEmissivityGrid::query_tally_grid(
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

void ComputePhotonEmissivityGrid::post_process_grid(
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

double ComputePhotonEmissivityGrid::interpolatePEC(
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

/* ----------------------------------------------------------------------
   read PEC HDF5 file (rank 0 only)
------------------------------------------------------------------------- */

void ComputePhotonEmissivityGrid::readPECFile(const std::string &path)
{
  if (comm->me != 0) return;

  try {
    H5::Exception::dontPrint();
    H5::H5File file(path, H5F_ACC_RDONLY);

    // read 1D helper
    auto read1D = [&](const std::string &name) -> std::vector<double> {
      H5::DataSet ds = file.openDataSet(name);
      H5::DataSpace space = ds.getSpace();
      hsize_t dim;
      space.getSimpleExtentDims(&dim);
      std::vector<double> vec(dim);
      ds.read(vec.data(), H5::PredType::NATIVE_DOUBLE);
      return vec;
    };

    // Auto-detect dataset names:
    //   Te: "te" or "te_grid"
    //   ne: "ne" or "ne_grid"
    //   PEC: the remaining 2D dataset
    auto tryOpen = [&](const std::string &a, const std::string &b) -> std::string {
      htri_t ex = 0;
      H5E_BEGIN_TRY { ex = H5Lexists(file.getId(), a.c_str(), H5P_DEFAULT); } H5E_END_TRY;
      if (ex > 0) return a;
      H5E_BEGIN_TRY { ex = H5Lexists(file.getId(), b.c_str(), H5P_DEFAULT); } H5E_END_TRY;
      if (ex > 0) return b;
      return "";
    };

    std::string te_name = tryOpen("te_grid", "te");
    std::string ne_name = tryOpen("ne_grid", "ne");
    if (te_name.empty())
      error->one(FLERR, "PEC file: no 'te' or 'te_grid' dataset found");
    if (ne_name.empty())
      error->one(FLERR, "PEC file: no 'ne' or 'ne_grid' dataset found");

    std::vector<double> te = read1D(te_name);
    std::vector<double> ne = read1D(ne_name);
    pec_nte = (int) te.size();
    pec_nne = (int) ne.size();

    // Find the 2D PEC dataset (any dataset that is not te or ne)
    std::string pec_name;
    hsize_t nobj = file.getNumObjs();
    for (hsize_t i = 0; i < nobj; i++) {
      std::string name = file.getObjnameByIdx(i);
      if (name == te_name || name == ne_name) continue;
      if (file.getObjTypeByIdx(i) != H5G_DATASET) continue;
      H5::DataSet ds = file.openDataSet(name);
      H5::DataSpace space = ds.getSpace();
      if (space.getSimpleExtentNdims() == 2) {
        pec_name = name;
        break;
      }
    }
    if (pec_name.empty())
      error->one(FLERR, "PEC file: no 2D PEC dataset found");

    // read 2D PEC table [nte x nne]
    H5::DataSet ds = file.openDataSet(pec_name);
    H5::DataSpace space = ds.getSpace();
    hsize_t dims[2];
    space.getSimpleExtentDims(dims);
    if ((int)dims[0] != pec_nte || (int)dims[1] != pec_nne)
      error->one(FLERR,
        "PEC file: 2D dataset shape does not match te x ne grids");

    std::vector<double> raw(pec_nte * pec_nne);
    ds.read(raw.data(), H5::PredType::NATIVE_DOUBLE);

    // apply unit conversion (default: cm^3/s -> m^3/s)
    for (int i = 0; i < pec_nte * pec_nne; i++)
      raw[i] *= pec_unit_conv;

    // convert to log10
    pec_log_te.resize(pec_nte);
    for (int i = 0; i < pec_nte; i++)
      pec_log_te[i] = std::log10(std::max(te[i], 1.0e-30));

    pec_log_ne.resize(pec_nne);
    for (int i = 0; i < pec_nne; i++)
      pec_log_ne[i] = std::log10(std::max(ne[i], 1.0e-30));

    pec_log_val.resize(pec_nte * pec_nne);
    for (int i = 0; i < pec_nte * pec_nne; i++)
      pec_log_val[i] = std::log10(std::max(raw[i], 1.0e-99));

    file.close();

    printf("  PEC table: dataset '%s', %d Te x %d ne, "
           "Te=[%.2f,%.2f] eV, ne=[%.2e,%.2e] m^-3\n",
           pec_name.c_str(), pec_nte, pec_nne,
           te.front(), te.back(), ne.front(), ne.back());

  } catch (H5::Exception &e) {
    char msg[512];
    snprintf(msg, sizeof(msg),
      "Compute photon_emissivity/grid: cannot read PEC file '%s': %s",
      path.c_str(), e.getDetailMsg().c_str());
    error->one(FLERR, msg);
  }
}

/* ----------------------------------------------------------------------
   broadcast PEC table from rank 0 to all ranks
------------------------------------------------------------------------- */

void ComputePhotonEmissivityGrid::broadcastPECData()
{
  MPI_Bcast(&pec_nte, 1, MPI_INT, 0, world);
  MPI_Bcast(&pec_nne, 1, MPI_INT, 0, world);

  if (comm->me != 0) {
    pec_log_te.resize(pec_nte);
    pec_log_ne.resize(pec_nne);
    pec_log_val.resize(pec_nte * pec_nne);
  }

  MPI_Bcast(pec_log_te.data(), pec_nte, MPI_DOUBLE, 0, world);
  MPI_Bcast(pec_log_ne.data(), pec_nne, MPI_DOUBLE, 0, world);
  MPI_Bcast(pec_log_val.data(), pec_nte * pec_nne, MPI_DOUBLE, 0, world);
}

/* ---------------------------------------------------------------------- */

void ComputePhotonEmissivityGrid::reallocate()
{
  if (grid->nlocal == nglocal) return;

  memory->destroy(vector_grid);
  memory->destroy(tally);
  nglocal = grid->nlocal;
  memory->create(vector_grid, nglocal, "photon_emissivity:vector_grid");
  memory->create(tally, nglocal, ntotal, "photon_emissivity:tally");
}

/* ---------------------------------------------------------------------- */

bigint ComputePhotonEmissivityGrid::memory_usage()
{
  bigint bytes = 0;
  bytes += nglocal * sizeof(double);
  bytes += (bigint)ntotal * nglocal * sizeof(double);
  bytes += (pec_nte + pec_nne + pec_nte * pec_nne) * sizeof(double);
  return bytes;
}
