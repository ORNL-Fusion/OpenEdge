/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_evap.h"
#include "update.h"
#include "grid.h"
#include "particle.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "math.h"
#include "input.h"
#include "collide.h"
#include "modify.h"
#include "fix.h"
#include "math_const.h"
#include "math_extra.h"
#include <cmath>
#include <domain.h>
#include "random_knuth.h"
#include "mixture.h"

enum HeatfluxMode { HF_NONE=0, HF_FILE, HF_CONST };
HeatfluxMode heatflux_mode = HF_NONE;

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixEvap::FixEvap(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  // Required: fix ID style nevery mix-ID ...
  if (narg < 4) error->all(FLERR,"Illegal fix evap command (need: nevery mix-ID)");

  // required positional
  nevery = atoi(arg[2]);
  imix   = particle->find_mixture(arg[3]);
  if (imix < 0) error->all(FLERR,"Fix evap: unknown mixture ID");

  // defaults for optionals
  set_mass   = NAN;      // or a physical default
  set_temp   = NAN;
  set_radius = NAN;
  heatflux_mode = HF_NONE;   // your enum
  Qs_const   = 0.0;

  // parse optional keywords starting at arg[4]
  int i = 4;
  while (i < narg) {
    if (strcmp(arg[i],"mass") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evap: missing value for 'mass'");
      set_mass = atof(arg[i+1]); i += 2;

    } else if (strcmp(arg[i],"temp") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evap: missing value for 'temp'");
      set_temp = atof(arg[i+1]); i += 2;

    } else if (strcmp(arg[i],"radius") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evap: missing value for 'radius'");
      set_radius = atof(arg[i+1]); i += 2;

    } else if (strcmp(arg[i],"heatflux/file") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evap: missing value for 'heatflux/file'");
      heatflux_mode = HF_FILE;
      heatfluxFilename = std::string(arg[i+1]);
      i += 2;

    } else if (strcmp(arg[i],"heatflux/constant") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evap: missing value for 'heatflux/constant'");
      heatflux_mode = HF_CONST;
      Qs_const = atof(arg[i+1]);
      i += 2;

    } else {
      char msg[256];
      snprintf(msg,sizeof(msg),"Fix evap: unknown keyword '%s'",arg[i]);
      error->all(FLERR,msg);
    }
  }

  // Optional: validate required optionals depending on mode
  if (heatflux_mode == HF_FILE && heatfluxFilename.empty())
    error->all(FLERR,"Fix evap: empty filename for heatflux/file");

  // per-grid memory initialization

  size_per_grid_cols = 3;
  maxgrid = 0;
  array_grid = NULL;

  }



/* ---------------------------------------------------------------------- */

FixEvap::~FixEvap()
{
  if (copymode) return;
  memory->destroy(array_grid);
}

/* ---------------------------------------------------------------------- */
int FixEvap::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;   // pre-Boris half "evap"
  mask |= END_OF_STEP;     // post-Boris half "evap"
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixEvap::init() {
  if (domain->dimension != 2)
    error->all(FLERR,"Fix evap: only 2D geometry supported");

  if (heatflux_mode == HF_FILE) {
    if (heatfluxFilename.empty())
      error->all(FLERR,"Fix evap: heatflux/file given but filename is empty");
    initializeHeatFluxData();               // reads + broadcasts
  } else if (heatflux_mode == HF_CONST) {
    Qs_const = Qs_const;   // already set
  } else {
    error->all(FLERR,"Fix evap: must provide heatflux/constant <W/m^2> or heatflux/file <h5>");
  }

    if (grid->nlocal > maxgrid) {
    maxgrid = grid->maxlocal;
    memory->destroy(array_grid);
    memory->create(array_grid,maxgrid,size_per_grid_cols,"array_grid");
  }

  // bigint nbytes = (bigint) grid->nlocal * size_per_grid_cols;
  // if (nbytes) memset(&array_grid[0][0],0,nbytes*sizeof(double));

  if (grid->nlocal) {
    memset(&array_grid[0][0], 0, grid->nlocal * size_per_grid_cols * sizeof(double));
  }

}

/* ---------------------------------------------------------------------- */

void FixEvap::start_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  evap_half(0.5 * update->dt);
}

void FixEvap::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  evap_half(0.5 * update->dt);
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double FixEvap::memory_usage() {
  double bytes = 0.0;
  bytes += maxgrid * size_per_grid_cols * sizeof(double);
  return bytes;
}


// advance (r_d, T_d, m_d) by dt_half using current Q_s and model
void FixEvap::evap_half(double dt_half)
{
  if ((update->ntimestep % nevery) != 0) return;
  if (!particle->sorted) particle->sort();

  // (Re)alloc per-grid arrays if needed
  if (grid->nlocal > maxgrid) {
    maxgrid = grid->maxlocal;
    memory->destroy(array_grid);
    memory->create(array_grid, maxgrid, size_per_grid_cols, "array_grid");
  }
  if (grid->nlocal) {
    memset(&array_grid[0][0], 0, grid->nlocal * size_per_grid_cols * sizeof(double));
  }

  Particle::OnePart *parts = particle->particles;
  int *next = particle->next;
  Grid::ChildInfo *cinfo = grid->cinfo;
  const int nglocal = grid->nlocal;

  int *s2g = particle->mixture[imix]->species2group;

  for (int icell = 0; icell < nglocal; icell++) {
    if (cinfo[icell].count == 0) continue;
    int ip = cinfo[icell].first;
    while (ip >= 0) {
      const int is  = parts[ip].ispecies;
      const int ig  = s2g[is];
      if (ig < 0) { ip = next[ip]; continue; }

      // one-time seeding
      if (set_mass   > 0.0 && parts[ip].mass   <= 0.0) parts[ip].mass   = set_mass;
      if (set_radius > 0.0 && parts[ip].radius <= 0.0) parts[ip].radius = set_radius;
      if (set_temp   > 0.0 && parts[ip].temp   <= 0.0) parts[ip].temp   = set_temp;

      droplet_evaporation_model(&parts[ip], dt_half, icell);

      ip = next[ip];
    }
  }
}


/*----------------------------------------------------------------------
Sergey's Evaporation Model
----------------------------------------------------------------------*/
void FixEvap::droplet_evaporation_model(Particle::OnePart *ip,
                                        const double dt_half,
                                        const int icell)
{
  // --- constants ---
  const double AM   = 1.53e-26;      // Li atom mass [kg/atom]
  const double Rho  = 534.0;         // kg/m^3
  const double Cp   = 4200.0;        // J/kg-K
  const double DHm  = 3.158e+03;     // J/mol  (your Python uses this; consider ~1e5 J/mol physically)
  const double AN   = 6.022e+23;     // 1/mol
  const double DT   = dt_half;

    // microscopic cutoffs shared with viscous
  // constexpr double R_STOP = 5e-9;   // m
  // constexpr double R_ATOM = 5e-9;   // m, same as R_STOP
  // constexpr double AM_LI  = 1.53e-26; // kg

      // microscopic cutoffs shared with viscous
  constexpr double R_STOP = 5e-8;   // m
  constexpr double R_ATOM = 5e-8;   // m, same as R_STOP
  constexpr double AM_LI  = 1.53e-25; // kg
  // --- current state (Kelvin in OpenEdge) ---
  const double mass   = (ip->mass   > 0.0) ? ip->mass   : particle->species[ip->ispecies].mass;
  const double radius = (ip->radius > 0.0) ? ip->radius : pow((3.0*mass)/(4.0*M_PI*Rho), 1.0/3.0);
  const double TK     = (ip->temp   > 0.0) ? ip->temp   : 300.0;

    // If already microscopic: freeze and skip evap math
  if (radius <= R_STOP) {
    ip->radius =  5e-9;
    if (ip->mass > 0.0 && ip->mass != AM_LI) ip->mass = 1.53e-26;
    ip->temp   = 0.0;
    return;
  }

  const double rpos = ip->x[0];
  const double zpos = ip->x[1];

  // --- heat flux Qs (W/m^2) ---
  double Qs = 0.0;
  if (heatflux_mode == HF_CONST) {
    Qs = Qs_const;
  } else if (heatflux_mode == HF_FILE) {
    if (domain->dimension != 2)
      error->all(FLERR,"Fix evap: only 2D geometry is supported for heat flux interpolation");
    HeatFluxParams hp = interpHeatFluxAtPos(rpos, zpos, heat_flux_data);
    Qs = hp.q_mag;
    if (!std::isfinite(Qs) || Qs < 0.0) Qs = 0.0;
  } else {
    error->all(FLERR,"Fix evap: heatflux mode not set properly");
  }

  if (Qs <= 0.0) 
  {
        // --- write back ---
    ip->radius = radius;
    ip->temp   = TK;
    ip->mass   = mass;
    return;   // no evaporation if no heat flux
  }
 
  // --- Antoine vapor pressure (your Python fit) ---
  const double a1 = 5.055;
  const double b1 = -8023.0;
  const double xm1 = 6.939;             // molar mass used in your fit
  const double vpres1 = 760.0 * pow(10.0, (a1 + b1 / TK));  // mmHg

  // Keep that here so the math matches exactly.
  const double Gevap_atoms = 1.0e4 * 3.513e22 * vpres1 / sqrt(xm1 * TK);  // atoms/(m^2 s)

  // --- dR/dt and dT/dt (mirror Python) ---
  const double dRdt = -AM * Gevap_atoms / Rho;                        // m/s
  const double HF   = Qs - Gevap_atoms * (DHm / AN);                  // W/m^2 (DHm/AN = J/atom)
  const double dTdt = (3.0 / (Rho * Cp)) * HF;                        // K/s, lumped sphere

  // --- advance state ---
  const double R_new = std::max(0.0, radius + dRdt * DT);
  const double T_new = TK + dTdt * DT;                                // Kelvin
  // mass derived from radius (no fnum anywhere)
  const double mass_new = (R_new > 0.0) ? (Rho * (4.0/3.0) * M_PI * R_new*R_new*R_new) : 0.0;

  // --- diagnostics: N_emit this step (atoms), for droplet surface area only ---
  const double Adrop = 4.0 * M_PI * std::max(R_new, 0.0) * std::max(R_new, 0.0);  // m^2
  const double N_emit_diag = Gevap_atoms * Adrop * DT;  // atoms emitted this step (no fnum)

  // --- guards ---
  if (T_new < 0.0 || R_new < 0.0 || mass_new < 0.0)
    error->all(FLERR,"Fix evap: particle temperature dropped below 0 K");

  // --- write back ---
  ip->radius = R_new;
  ip->temp   = T_new;
  ip->mass   = mass_new;

  if (icell >= 0 && icell < grid->nlocal && array_grid) {
    array_grid[icell][0] = N_emit_diag;
    array_grid[icell][1]  = icell;
    array_grid[icell][2]  = Qs;
  }

}
/* ----------------------------------------------------------------------
   Read plasma data from HDF5 file
------------------------------------------------------------------------- */
HeatFluxData FixEvap::readHeatFlux(const std::string& filePath) {
    printf("Reading heat flux data from file: %s\n", filePath.c_str());
    HeatFluxData data;

    try {
        H5::H5File file(filePath, H5F_ACC_RDONLY);

        // Utility to read 1D dataset
        auto read1D = [&](const std::string& name) -> std::vector<double> {
            H5::DataSet ds = file.openDataSet(name);
            H5::DataSpace space = ds.getSpace();
            hsize_t dim;
            space.getSimpleExtentDims(&dim);
            std::vector<double> vec(dim);
            ds.read(vec.data(), H5::PredType::NATIVE_DOUBLE);
            return vec;
        };

                // First read coordinates
        data.r = read1D("grid/Rc");
        data.z = read1D("grid/Zc");
        size_t nr = data.r.size();
        size_t nz = data.z.size();

                // Utility to read 2D dataset with shape validation
        auto read2D = [&](const std::string& name) -> std::vector<std::vector<double>> {
            H5::DataSet ds = file.openDataSet(name);
            H5::DataSpace space = ds.getSpace();
            hsize_t dims[2];
            space.getSimpleExtentDims(dims);

            if (dims[0] != nz || dims[1] != nr) {
                throw std::runtime_error("Dataset '" + name + "' shape mismatch: expected " +
                                         std::to_string(nz) + " x " + std::to_string(nr) +
                                         ", got " + std::to_string(dims[0]) + " x " + std::to_string(dims[1]));
            }

            std::vector<double> raw(dims[0] * dims[1]);
            ds.read(raw.data(), H5::PredType::NATIVE_DOUBLE);

            std::vector<std::vector<double>> grid(nz, std::vector<double>(nr));
            for (size_t i = 0; i < nz; ++i) {
                for (size_t j = 0; j < nr; ++j) {
                    grid[i][j] = raw[i * nr + j];
                }
            }
            return grid;
        };

        data.q_mag = read2D("fields/q_mag");

    } catch (const H5::Exception& e) {
        fprintf(stderr, "HDF5 error: %s\n", e.getCDetailMsg());
        throw;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        throw;
    }

    printf("Finished reading heat flux data from file: %s\n", filePath.c_str());
    return data;
}


/*----------------------------------------------------------------------
   broadcast heat flux data
------------------------------------------------------------------------- */

void FixEvap::broadcastHeatFluxData(HeatFluxData& data) {
  int me = comm->me;

  // Broadcast sizes of 1D vectors (e.g., r and z for the magnetic field)
  int r_size = data.r.size();
  int z_size = data.z.size();
  MPI_Bcast(&r_size, 1, MPI_INT, 0, world);
  MPI_Bcast(&z_size, 1, MPI_INT, 0, world);

  // Resize vectors on non-root processes
  if (me != 0) {
      data.r.resize(r_size);
      data.z.resize(z_size);
  }

  // Broadcast 1D vector data (r and z)
  MPI_Bcast(data.r.data(), r_size, MPI_DOUBLE, 0, world);
  MPI_Bcast(data.z.data(), z_size, MPI_DOUBLE, 0, world);

  // Broadcast 2D vectors (e.g., br, bt, bz)
  auto broadcast2DVector = [&](std::vector<std::vector<double>>& vec) {
      int dim1 = vec.size();
      int dim2 = dim1 ? vec[0].size() : 0;

      MPI_Bcast(&dim1, 1, MPI_INT, 0, world);
      MPI_Bcast(&dim2, 1, MPI_INT, 0, world);

      // Resize the outer vector and each inner vector on non-root processes
      if (me != 0) {
          vec.resize(dim1, std::vector<double>(dim2));
      }

      for (int i = 0; i < dim1; ++i) {
          MPI_Bcast(vec[i].data(), dim2, MPI_DOUBLE, 0, world);
      }
  };

  // Broadcast the magnetic field components
  broadcast2DVector(data.q_mag);
}


/*---------------------------------
  initialize heat flux data
-----------------------------------*/
void FixEvap::initializeHeatFluxData() {
  int me = comm->me;

  // Load heat flux data only on the root process
  if (me == 0) {
      heat_flux_data = readHeatFlux(heatfluxFilename);
  }

  // Broadcast the heat flux data to all processes
  broadcastHeatFluxData(heat_flux_data);
}
// Utilities
inline static double safe_val(double v) {
  return std::isfinite(v) ? v : 0.0;
}

HeatFluxParams FixEvap::interpHeatFluxAtPos(double r, double z,
                                            const HeatFluxData& data) const
{
  HeatFluxParams res{}; // {r=0,z=0,q_mag=0} by default

  if (data.r.empty() || data.z.empty()) return res;

  const auto& r_vals = data.r;   // assumed sorted ascending
  const auto& z_vals = data.z;

  // out of bounds -> zero
  if (r < r_vals.front() || r > r_vals.back() ||
      z < z_vals.front() || z > z_vals.back()) {
    return res;
  }

  // locate bracketing indices
  auto r_it = std::lower_bound(r_vals.begin(), r_vals.end(), r);
  auto z_it = std::lower_bound(z_vals.begin(), z_vals.end(), z);

  int r1 = std::max(0, int(r_it - r_vals.begin()) - 1);
  int r2 = std::min(int(r_vals.size()) - 1, r1 + 1);
  int z1 = std::max(0, int(z_it - z_vals.begin()) - 1);
  int z2 = std::min(int(z_vals.size()) - 1, z1 + 1);

  const double R1 = r_vals[r1], R2 = r_vals[r2];
  const double Z1 = z_vals[z1], Z2 = z_vals[z2];
  const double denom = (R2 - R1) * (Z2 - Z1);

  auto interp = [&](const std::vector<std::vector<double>>& field)->double {
    if (field.size() <= size_t(z2) || field[0].size() <= size_t(r2)) return 0.0;

    // load 4 corners and sanitize NaNs/Infs
    const double Q11 = safe_val(field[z1][r1]);
    const double Q21 = safe_val(field[z1][r2]);
    const double Q12 = safe_val(field[z2][r1]);
    const double Q22 = safe_val(field[z2][r2]);

    if (denom == 0.0) return 0.25*(Q11 + Q21 + Q12 + Q22);

    double q = (Q11*(R2-r)*(Z2-z) + Q21*(r-R1)*(Z2-z)
              + Q12*(R2-r)*(z-Z1) + Q22*(r-R1)*(z-Z1)) / denom;

    if (!std::isfinite(q)) q = 0.0; // guard
    if (q < 0.0) q = 0.0;           // no negative heat flux
    return q;
  };

  res.r = r;
  res.z = z;
  res.q_mag = interp(data.q_mag);

  // final guard
  if (!std::isfinite(res.q_mag) || res.q_mag < 0.0) res.q_mag = 0.0;
  return res;
}
