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
#include "fix_evaporation.h"
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
#include "domain.h"
#include "random_knuth.h"
#include "mixture.h"
#include <stdexcept>


using namespace SPARTA_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

FixEvap::FixEvap(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  // Required: fix ID style nevery mix-ID ...
  if (narg < 4) error->all(FLERR,"Illegal fix evaporation command (need: nevery mix-ID)");

  // required positional
  nevery = atoi(arg[2]);
  imix   = particle->find_mixture(arg[3]);
  if (imix < 0) error->all(FLERR,"Fix evaporation: unknown mixture ID");

  // defaults for optionals
  set_mass   = NAN;
  set_temp   = NAN;
  set_radius = NAN;
  heatflux_mode = HF_NONE;
  Qs_const   = 0.0;

  // parse optional keywords starting at arg[4]
  int i = 4;
  while (i < narg) {
    if (strcmp(arg[i],"mass") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'mass'");
      set_mass = atof(arg[i+1]); i += 2;

    } else if (strcmp(arg[i],"temp") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'temp'");
      set_temp = atof(arg[i+1]); i += 2;

    } else if (strcmp(arg[i],"radius") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'radius'");
      set_radius = atof(arg[i+1]); i += 2;

    } else if (strcmp(arg[i],"heatflux/file") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'heatflux/file'");
      heatflux_mode = HF_FILE;
      heatfluxFilename = std::string(arg[i+1]);
      i += 2;

    } else if (strcmp(arg[i],"heatflux/constant") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'heatflux/constant'");
      heatflux_mode = HF_CONST;
      Qs_const = atof(arg[i+1]);
      i += 2;

    } else {
      char msg[256];
      snprintf(msg,sizeof(msg),"Fix evaporation: unknown keyword '%s'",arg[i]);
      error->all(FLERR,msg);
    }
  }

  // Optional: validate required optionals depending on mode
  if (heatflux_mode == HF_FILE && heatfluxFilename.empty())
    error->all(FLERR,"Fix evaporation: empty filename for heatflux/file");

  per_grid_flag = 1;
  per_grid_freq = nevery;
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
    error->all(FLERR,"Fix evaporation: only 2D geometry supported");

  if (heatflux_mode == HF_FILE) {
    if (heatfluxFilename.empty())
      error->all(FLERR,"Fix evaporation: heatflux/file given but filename is empty");
    initializeHeatFluxData();               // reads + broadcasts
  } else if (heatflux_mode != HF_CONST) {
    error->all(FLERR,"Fix evaporation: must provide heatflux/constant <W/m^2> or heatflux/file <h5>");
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
  // Zero per-cell accumulators once per full step, before the first half-kick.
  // end_of_step calls evap_half again; it must NOT re-zero the arrays there.
  if (grid->nlocal > maxgrid) {
    maxgrid = grid->maxlocal;
    memory->destroy(array_grid);
    memory->create(array_grid, maxgrid, size_per_grid_cols, "array_grid");
  }
  if (grid->nlocal)
    memset(&array_grid[0][0], 0, grid->nlocal * size_per_grid_cols * sizeof(double));
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

  // (Re)alloc per-grid arrays if the grid grew since start_of_step.
  // Zeroing is done once per step in start_of_step() so both half-kicks accumulate.
  if (grid->nlocal > maxgrid) {
    maxgrid = grid->maxlocal;
    memory->destroy(array_grid);
    memory->create(array_grid, maxgrid, size_per_grid_cols, "array_grid");
    // Grid grew mid-step: zero the newly allocated block so no garbage leaks in.
    if (grid->nlocal)
      memset(&array_grid[0][0], 0, grid->nlocal * size_per_grid_cols * sizeof(double));
  }

  Particle::OnePart *parts = particle->particles;
  const int nlocal = particle->nlocal;

  int *s2g = particle->mixture[imix]->species2group;
  int ndeleted = 0;

  // Iterate particles directly using each particle's icell. This avoids
  // forcing particle->sort() every half-step, which is expensive for long runs.
  for (int ip = 0; ip < nlocal; ip++) {
    const int is  = parts[ip].ispecies;
    const int ig  = s2g[is];
    if (ig < 0) continue;

    // one-time seeding
    if (set_mass   > 0.0 && parts[ip].mass   <= 0.0) parts[ip].mass   = set_mass;
    if (set_radius > 0.0 && parts[ip].radius <= 0.0) parts[ip].radius = set_radius;
    if (set_temp   > 0.0 && parts[ip].temp   <= 0.0) parts[ip].temp   = set_temp;

    const int icell = parts[ip].icell;
    droplet_evaporation_model(&parts[ip], dt_half, icell);

    // Remove droplet after 90% mass loss relative to its initial reference mass.
    const double m0_ref = (set_mass > 0.0) ? set_mass : particle->species[is].mass;
    const double m_cut  = 0.1 * m0_ref;
    if (parts[ip].mass > 0.0 && m0_ref > 0.0 && parts[ip].mass <= m_cut) {
      parts[ip].mass = 0.0;
      parts[ip].radius = 0.0;
      parts[ip].temp = 0.0;
      parts[ip].icell = -1;   // particle->compress_rebalance() deletes marked particles
      ndeleted++;
    }
  }

  // Physically remove evaporated droplets from the particle list.
  if (ndeleted > 0) particle->compress_rebalance();
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

  // --- current state (Kelvin in OpenEdge) ---
  const double mass   = (ip->mass   > 0.0) ? ip->mass   : particle->species[ip->ispecies].mass;
  const double radius = (ip->radius > 0.0) ? ip->radius : pow((3.0*mass)/(4.0*MY_PI*Rho), 1.0/3.0);
  const double TK     = (ip->temp   > 0.0) ? ip->temp   : 300.0;

  // Plasma backgrounds are 2D (r,z): map particle position for any SPARTA geometry.
  double rpos = 0.0, zpos = 0.0;
  if (domain->axisymmetric || domain->dimension == 2) {
    rpos = ip->x[0];
    zpos = ip->x[1];
  } else {
    rpos = std::sqrt(ip->x[0]*ip->x[0] + ip->x[1]*ip->x[1]);
    zpos = ip->x[2];
  }

  // --- heat flux Qs (W/m^2) ---
  double Qs = 0.0;
  if (heatflux_mode == HF_CONST) {
    Qs = Qs_const;
  } else if (heatflux_mode == HF_FILE) {
    HeatFluxParams hp = interpHeatFluxAtPos(rpos, zpos, heat_flux_data);
    Qs = hp.q_mag;
    if (!std::isfinite(Qs) || Qs < 0.0) Qs = 0.0;
  } else {
    error->one(FLERR,"Fix evaporation: heatflux mode not set properly");
  }

  if (Qs <= 0.0) 
  {
        // --- write back ---
    ip->radius = radius;
    ip->temp   = TK;
    ip->mass   = mass;
    return;   // no evaporation if no heat flux
  }
    Qs = Qs * 3;
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
  // Spherical droplet energy balance: dT/dt = (3/(rho*Cp*r)) * (q'' - m''*L).
  const double radius_safe = (radius > 1.0e-20) ? radius : 1.0e-20;
  const double dTdt = (3.0 / (Rho * Cp * radius_safe)) * HF;          // K/s

  // --- advance state ---
  const double R_new = std::max(0.0, radius + dRdt * DT);
  const double T_new = TK + dTdt * DT;                                // Kelvin
  // mass derived from radius (no fnum anywhere)
  const double mass_new = (R_new > 0.0) ? (Rho * (4.0/3.0) * MY_PI * R_new*R_new*R_new) : 0.0;

  // --- guards ---
  // R_new and mass_new are guarded by std::max(0,.) above so can't go negative.
  // Only T_new can genuinely go negative (excessive cooling).
  if (T_new < 0.0)
    error->one(FLERR,"Fix evaporation: particle temperature dropped below 0 K");

  // --- write back ---
  ip->radius = R_new;
  ip->temp   = T_new;
  ip->mass   = mass_new;

  // NOTE: do NOT write particle->species[].mass here — that is a global
  // table shared across all ranks and particles.  Per-particle mass lives
  // in ip->mass (already set above).  Writing the species table from a
  // per-particle loop is a race condition under MPI and Kokkos.

  // Per-cell accumulators: mass lost (kg), atoms lost, heat absorbed (J).
  // Uses += so multiple droplets per cell and both half-kicks accumulate correctly.
  if (icell >= 0 && icell < grid->nlocal && array_grid) {
    const double dm = Rho * (4.0/3.0) * MY_PI *
                      (radius*radius*radius - R_new*R_new*R_new);  // kg, >= 0
    array_grid[icell][0] += dm;                                    // kg
    array_grid[icell][1] += dm / AM;                               // atoms
    array_grid[icell][2] += Qs * 4.0*MY_PI * R_new*R_new * DT;    // J
    
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

        auto exists = [&](const std::string& name) -> bool {
            try {
                file.openDataSet(name);
                return true;
            } catch (const H5::Exception&) {
                return false;
            }
        };

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

        // Utility to read 2D dataset (raw)
        auto read2DRaw = [&](const std::string& name,
                             std::vector<double>& raw,
                             hsize_t dims[2]) {
            H5::DataSet ds = file.openDataSet(name);
            H5::DataSpace space = ds.getSpace();
            space.getSimpleExtentDims(dims);
            if (space.getSimpleExtentNdims() != 2)
                throw std::runtime_error("Dataset '" + name + "' is not 2D");
            raw.assign(dims[0] * dims[1], 0.0);
            ds.read(raw.data(), H5::PredType::NATIVE_DOUBLE);
        };

        // Utility to convert flat raw -> vector<vector> with shape [nz][nr]
        auto to2D = [&](const std::vector<double>& raw, size_t nz, size_t nr) {
            std::vector<std::vector<double>> grid(nz, std::vector<double>(nr));
            for (size_t iz = 0; iz < nz; ++iz) {
                for (size_t ir = 0; ir < nr; ++ir) {
                    grid[iz][ir] = raw[iz * nr + ir];
                }
            }
            return grid;
        };

        const bool has_legacy_1d = exists("grid/Rc") && exists("grid/Zc");
        const bool has_grid_2d   = exists("grid/R") && exists("grid/Z");
        const bool has_root_2d   = exists("R") && exists("Z");

        if (has_legacy_1d) {
            data.r = read1D("grid/Rc");
            data.z = read1D("grid/Zc");
        } else if (has_grid_2d || has_root_2d) {
            const std::string rname = has_grid_2d ? "grid/R" : "R";
            const std::string zname = has_grid_2d ? "grid/Z" : "Z";

            std::vector<double> rraw, zraw;
            hsize_t rdims[2] = {0, 0}, zdims[2] = {0, 0};
            read2DRaw(rname, rraw, rdims);
            read2DRaw(zname, zraw, zdims);
            if (rdims[0] != zdims[0] || rdims[1] != zdims[1]) {
                throw std::runtime_error("Heatflux grid R/Z shape mismatch");
            }
            const size_t nz = static_cast<size_t>(rdims[0]);
            const size_t nr = static_cast<size_t>(rdims[1]);
            data.r.resize(nr);
            data.z.resize(nz);
            // For 2D coordinate datasets, collapse to separable 1D axes by averaging.
            // This matches SPARTA's rectilinear bilinear interpolation assumptions.
            for (size_t ir = 0; ir < nr; ++ir) {
                double acc = 0.0;
                for (size_t iz = 0; iz < nz; ++iz) acc += rraw[iz * nr + ir];
                data.r[ir] = acc / static_cast<double>(nz);
            }
            for (size_t iz = 0; iz < nz; ++iz) {
                double acc = 0.0;
                for (size_t ir = 0; ir < nr; ++ir) acc += zraw[iz * nr + ir];
                data.z[iz] = acc / static_cast<double>(nr);
            }
        } else {
            throw std::runtime_error(
                "Heatflux grid coordinates not found. Expected either "
                "'grid/Rc'+'grid/Zc' or 2D 'grid/R'+'grid/Z' (or root 'R'+'Z')."
            );
        }

        const size_t nr = data.r.size();
        const size_t nz = data.z.size();
        if (nr < 2 || nz < 2) {
            throw std::runtime_error("Heatflux grid must have at least 2 points in both R and Z");
        }

        auto read2DValidated = [&](const std::string& name) -> std::vector<std::vector<double>> {
            std::vector<double> raw;
            hsize_t dims[2] = {0, 0};
            read2DRaw(name, raw, dims);
            if (dims[0] == nz && dims[1] == nr) {
                return to2D(raw, nz, nr);
            }
            if (dims[0] == nr && dims[1] == nz) {
                // Accept transposed storage and convert once at read time.
                std::vector<std::vector<double>> transposed(nz, std::vector<double>(nr, 0.0));
                for (size_t ir = 0; ir < nr; ++ir) {
                    for (size_t iz = 0; iz < nz; ++iz) {
                        transposed[iz][ir] = raw[ir * nz + iz];
                    }
                }
                return transposed;
            }
            throw std::runtime_error("Dataset '" + name + "' shape mismatch: expected " +
                                     std::to_string(nz) + " x " + std::to_string(nr) +
                                     " (or transposed " + std::to_string(nr) + " x " +
                                     std::to_string(nz) + "), got " +
                                     std::to_string(dims[0]) + " x " + std::to_string(dims[1]));
        };

        if (exists("fields/q_mag")) {
            data.q_mag = read2DValidated("fields/q_mag");
        } else if (exists("q_mag")) {
            data.q_mag = read2DValidated("q_mag");
        } else {
            throw std::runtime_error("Missing heatflux dataset: expected 'fields/q_mag' or 'q_mag'");
        }

        for (size_t iz = 0; iz < nz; ++iz) {
            for (size_t ir = 0; ir < nr; ++ir) {
                double &q = data.q_mag[iz][ir];
                if (!std::isfinite(q) || q < 0.0) q = 0.0;
            }
        }

        if (!std::is_sorted(data.r.begin(), data.r.end()) ||
            !std::is_sorted(data.z.begin(), data.z.end())) {
            throw std::runtime_error("Heatflux coordinate axes must be monotonic increasing");
        }

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

  if (data.r.size() < 2 || data.z.size() < 2 || data.q_mag.empty()) return res;

  const auto& r_vals = data.r;
  const auto& z_vals = data.z;
  const int nr = static_cast<int>(r_vals.size());
  const int nz = static_cast<int>(z_vals.size());

  // Clamp, matching compute/plasma/fields behavior.
  const double r_clamp = std::min(std::max(r, r_vals.front()), r_vals.back());
  const double z_clamp = std::min(std::max(z, z_vals.front()), z_vals.back());

  auto r_it = std::lower_bound(r_vals.begin(), r_vals.end(), r_clamp);
  auto z_it = std::lower_bound(z_vals.begin(), z_vals.end(), z_clamp);

  int ir2 = static_cast<int>(r_it - r_vals.begin());
  int iz2 = static_cast<int>(z_it - z_vals.begin());
  if (ir2 <= 0) ir2 = 1;
  if (ir2 >= nr) ir2 = nr - 1;
  if (iz2 <= 0) iz2 = 1;
  if (iz2 >= nz) iz2 = nz - 1;
  const int ir1 = ir2 - 1;
  const int iz1 = iz2 - 1;

  const double R1 = r_vals[ir1], R2 = r_vals[ir2];
  const double Z1 = z_vals[iz1], Z2 = z_vals[iz2];
  const double dR = R2 - R1;
  const double dZ = Z2 - Z1;
  if (dR <= 0.0 || dZ <= 0.0) return res;

  const double t = (r_clamp - R1) / dR;
  const double u = (z_clamp - Z1) / dZ;
  const double w11 = (1.0 - t) * (1.0 - u);
  const double w21 = t * (1.0 - u);
  const double w12 = (1.0 - t) * u;
  const double w22 = t * u;

  if (iz1 < 0 || iz2 >= static_cast<int>(data.q_mag.size())) return res;
  if (ir1 < 0 || ir2 >= static_cast<int>(data.q_mag[iz1].size())) return res;
  if (ir2 >= static_cast<int>(data.q_mag[iz2].size())) return res;

  const double q11 = safe_val(data.q_mag[iz1][ir1]);
  const double q21 = safe_val(data.q_mag[iz1][ir2]);
  const double q12 = safe_val(data.q_mag[iz2][ir1]);
  const double q22 = safe_val(data.q_mag[iz2][ir2]);

  res.r = r_clamp;
  res.z = z_clamp;
  double q = w11 * q11 + w21 * q21 + w12 * q12 + w22 * q22;
  if (!std::isfinite(q) || q < 0.0) q = 0.0;
  res.q_mag = q;
  return res;
}
