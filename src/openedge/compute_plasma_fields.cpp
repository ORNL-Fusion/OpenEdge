/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_plasma_fields.h"
#include "update.h"
#include "grid.h"
#include "domain.h"
#include "input.h"
#include "memory.h"
#include "error.h"
#include "comm.h"

#include <tuple>
#include <string>
#include <vector> 
#include <H5Cpp.h>



using namespace SPARTA_NS;

// user keywords
enum {
  BR, BT, BZ, BX, BY,
  ER, ET, EZ, EX, EY,
  VR, VT, VZ, VX, VY,
  TI, TE, NI, NE, PARRFLOW, EPAR,
  GRAD_TE_R, GRAD_TE_T, GRAD_TE_Z, GRAD_TI_R, GRAD_TI_T, GRAD_TI_Z
};

/* ---------------------------------------------------------------------- */

ComputePlasmaFields::
ComputePlasmaFields(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 5)
    error->all(FLERR,"Illegal compute plasma/fields command");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute plasma/fields grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  // defaults
  bconst[0] = bconst[1] = bconst[2] = 0.0;
  econst[0] = econst[1] = econst[2] = 0.0;
  teconst = ticonst = niconst = neconst = parrflowconst = 0.0;

  // parse:
  // compute ... plasma/fields ggroup file plasma.h5 bfield.h5 ...
  // compute ... plasma/fields ggroup constant [const args] ...
  int iarg = 3;
  if (iarg >= narg)
    error->all(FLERR,"compute plasma/fields requires mode: file or constant");
  if (strcmp(arg[iarg],"file") == 0) {
    input_mode = MODE_FILE;
    iarg++;
    if (iarg + 1 >= narg)
      error->all(FLERR,"compute plasma/fields file mode needs: plasma.h5 bfield.h5");
    plasmaStatePath = std::string(arg[iarg++]);
    magneticFieldsPath = std::string(arg[iarg++]);
  } else if (strcmp(arg[iarg],"constant") == 0) {
    input_mode = MODE_CONSTANT;
    iarg++;
  } else {
    error->all(FLERR,"compute plasma/fields mode must be 'file' or 'constant'");
  }

  // constant mode options
  if (input_mode == MODE_CONSTANT) {
    while (iarg < narg) {
      if (iarg + 3 < narg && strcmp(arg[iarg],"magnetic_field")==0) {
        bconst[0] = input->numeric(FLERR,arg[iarg+1]);
        bconst[1] = input->numeric(FLERR,arg[iarg+2]);
        bconst[2] = input->numeric(FLERR,arg[iarg+3]);
        iarg += 4;
      } else if (iarg + 3 < narg && strcmp(arg[iarg],"electric_field")==0) {
        econst[0] = input->numeric(FLERR,arg[iarg+1]);
        econst[1] = input->numeric(FLERR,arg[iarg+2]);
        econst[2] = input->numeric(FLERR,arg[iarg+3]);
        iarg += 4;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"temp_e")==0) {
        teconst = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"temp_i")==0) {
        ticonst = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"dens_e")==0) {
        neconst = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"dens_i")==0) {
        niconst = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"parrflow")==0) {
        parrflowconst = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else break;
    }
  }

  if (iarg >= narg)
    error->all(FLERR,"plasma/fields needs values (br/bt/bz, er/et/ez, vr/vt/vz, ...)");

  // collect value keywords
  nvalue = narg - iarg;
  value = new int[nvalue];
  for (int iv = 0; iv < nvalue; ++iv, ++iarg) {
    if      (strcmp(arg[iarg],"br")==0)      value[iv] = BR;
    else if (strcmp(arg[iarg],"bt")==0)      value[iv] = BT;
    else if (strcmp(arg[iarg],"bz")==0)      value[iv] = BZ;
    else if (strcmp(arg[iarg],"bx")==0)      value[iv] = BX; // cartesian alias
    else if (strcmp(arg[iarg],"by")==0)      value[iv] = BY; // cartesian alias
    else if (strcmp(arg[iarg],"er")==0)      value[iv] = ER;
    else if (strcmp(arg[iarg],"et")==0)      value[iv] = ET;
    else if (strcmp(arg[iarg],"ez")==0)      value[iv] = EZ;
    else if (strcmp(arg[iarg],"ex")==0)      value[iv] = EX; // cartesian alias
    else if (strcmp(arg[iarg],"ey")==0)      value[iv] = EY; // cartesian alias
    else if (strcmp(arg[iarg],"vr")==0)      value[iv] = VR;
    else if (strcmp(arg[iarg],"vt")==0)      value[iv] = VT;
    else if (strcmp(arg[iarg],"vz")==0)      value[iv] = VZ;
    else if (strcmp(arg[iarg],"vx")==0)      value[iv] = VX; // cartesian alias
    else if (strcmp(arg[iarg],"vy")==0)      value[iv] = VY; // cartesian alias
    else if (strcmp(arg[iarg],"temp_i")==0)  value[iv] = TI;
    else if (strcmp(arg[iarg],"temp_e")==0)  value[iv] = TE;
    else if (strcmp(arg[iarg],"dens_i")==0)  value[iv] = NI;
    else if (strcmp(arg[iarg],"dens_e")==0)  value[iv] = NE;
    else if (strcmp(arg[iarg],"parrflow")==0) value[iv] = PARRFLOW;
    else if (strcmp(arg[iarg],"epar")==0)    value[iv] = EPAR;
    else if (strcmp(arg[iarg],"grad_te_r")==0) value[iv] = GRAD_TE_R;
    else if (strcmp(arg[iarg],"grad_te_t")==0) value[iv] = GRAD_TE_T;
    else if (strcmp(arg[iarg],"grad_te_z")==0) value[iv] = GRAD_TE_Z;
    else if (strcmp(arg[iarg],"grad_ti_r")==0) value[iv] = GRAD_TI_R;
    else if (strcmp(arg[iarg],"grad_ti_t")==0) value[iv] = GRAD_TI_T;
    else if (strcmp(arg[iarg],"grad_ti_z")==0) value[iv] = GRAD_TI_Z;
    else error->all(FLERR,"Illegal plasma/fields value");
  }

  per_grid_flag = 1;
  if (nvalue == 1) size_per_grid_cols = 0;
  else size_per_grid_cols = nvalue;
  post_process_grid_flag = 0;

  nglocal = 0;
  vector_grid = NULL;
  array_grid = NULL;

  plasma_arr = NULL;
  mag_arr    = NULL;


}

/* ---------------------------------------------------------------------- */

ComputePlasmaFields::~ComputePlasmaFields()
{
  if (copymode) return;
  delete [] value;
  memory->destroy(vector_grid);
  memory->destroy(array_grid);

  if (plasma_arr) {
    memory->destroy(plasma_arr);
    plasma_arr = NULL;
  }
  if (mag_arr) {
    memory->destroy(mag_arr);
    mag_arr = NULL;
  } 
  
}

/* ---------------------------------------------------------------------- */

void ComputePlasmaFields::init()
{
  reallocate();

  const int me     = comm->me;
  const int ncells = grid->nlocal;

  // clean any old pre-sampled field arrays
  if (plasma_arr) {
    memory->destroy(plasma_arr);
    plasma_arr = NULL;
  }
  if (mag_arr) {
    memory->destroy(mag_arr);
    mag_arr = NULL;
  }


  // --- 1) load field source (file or constants) ------------------------------
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  memory->create(plasma_arr, ncells, "plasma/fields:plasma_arr");
  memory->create(mag_arr,    ncells, "plasma/fields:mag_arr");

  if (input_mode == MODE_FILE) {
    if (me == 0) {
      plasma_data   = readPlasmaFileData(plasmaStatePath);
      magnetic_data = readMagneticFieldFileData(magneticFieldsPath);
    }
    broadcastPlasmaData(plasma_data);
    broadcastMagneticData(magnetic_data);
    precomputeStencils(plasma_data.r, plasma_data.z, plasma_stencil);
    precomputeStencils(magnetic_data.r, magnetic_data.z, magnetic_stencil);

    for (int icell = 0; icell < ncells; ++icell) {
      if (!(cinfo[icell].mask & groupbit)) continue;
      if (cells[icell].nsplit < 1)         continue;
      plasma_arr[icell] = bilinearInterpolationPlasma(icell, plasma_data);
      mag_arr[icell]    = bilinearInterpolationMagneticField(icell, magnetic_data);
    }
  } else {
    plasma_stencil.clear();
    magnetic_stencil.clear();
    for (int icell = 0; icell < ncells; ++icell) {
      PlasmaFileParams p{};
      p.dens_e = neconst;
      p.temp_e = teconst;
      p.dens_i = niconst;
      p.temp_i = ticonst;
      p.grad_dens_e_r = 0.0;
      p.grad_dens_e_t = 0.0;
      p.grad_dens_e_z = 0.0;
      p.parr_flow = parrflowconst;
      p.parr_flow_r = 0.0;
      p.parr_flow_t = 0.0;
      p.parr_flow_z = 0.0;
      p.grad_temp_e_r = 0.0;
      p.grad_temp_e_t = 0.0;
      p.grad_temp_e_z = 0.0;
      p.grad_temp_i_r = 0.0;
      p.grad_temp_i_t = 0.0;
      p.grad_temp_i_z = 0.0;
      plasma_arr[icell] = p;

      MagneticFieldFileDataParams b{};
      b.br = bconst[0];
      b.bt = bconst[1];
      b.bz = bconst[2];
      b.r = 0.0;
      b.z = 0.0;
      mag_arr[icell] = b;
    }
  }
}


void ComputePlasmaFields::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;
  constexpr double eQ = 1.602176634e-19;
  constexpr double tiny = 1.0e-30;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  const int dim = domain->dimension;

  for (int icell = 0; icell < nglocal; icell++) {
    double *row = (nvalue == 1) ? NULL : array_grid[icell];
    if (!(cinfo[icell].mask & groupbit) || cells[icell].nsplit < 1) {
      if (nvalue == 1) vector_grid[icell] = 0.0;
      else for (int iv = 0; iv < nvalue; ++iv) row[iv] = 0.0;
      continue;
    }

    const PlasmaFileParams &P = plasma_arr[icell];
    const MagneticFieldFileDataParams &B = mag_arr[icell];

    // Base components are cylindrical (r, t, z).
    const double Br = B.br;
    const double Bt = B.bt;
    const double Bzv = B.bz;
    const double Er = (input_mode == MODE_CONSTANT) ? econst[0] : 0.0;
    const double Et = (input_mode == MODE_CONSTANT) ? econst[1] : 0.0;
    const double Ezv = (input_mode == MODE_CONSTANT) ? econst[2] : 0.0;
    const double Vr = P.parr_flow_r;
    const double Vt = P.parr_flow_t;
    const double Vzv = P.parr_flow_z;

    // Cylindrical -> Cartesian transform in 3D Cartesian geometry.
    // In 2D (r,z), cartesian aliases map directly to cylindrical components.
    double cphi = 1.0, sphi = 0.0;
    if (dim == 3) {
      const Grid::ChildCell &cell = cells[icell];
      const double x = 0.5 * (cell.lo[0] + cell.hi[0]);
      const double y = 0.5 * (cell.lo[1] + cell.hi[1]);
      const double rxy = std::sqrt(x*x + y*y);
      if (rxy > 0.0) {
        cphi = x / rxy;
        sphi = y / rxy;
      }
    }

    const double Bx = Br*cphi - Bt*sphi;
    const double By = Br*sphi + Bt*cphi;
    const double Ex = Er*cphi - Et*sphi;
    const double Ey = Er*sphi + Et*cphi;
    const double Vx = Vr*cphi - Vt*sphi;
    const double Vy = Vr*sphi + Vt*cphi;
    const double ne = std::max(P.dens_e, tiny);
    // WEST/SOLPS convention used here:
    //   Te in eV, ne in 1/m^3, grad(Te) in eV/m, grad(ne) in 1/m^4.
    // Then electron pressure is:
    //   pe [Pa] = e * ne * Te[eV]
    // and
    //   grad(pe) [Pa/m] = e * (Te*grad(ne) + ne*grad(Te)).
    const double gradPe_r = eQ * (P.temp_e * P.grad_dens_e_r + ne * P.grad_temp_e_r);
    const double gradPe_t = eQ * (P.temp_e * P.grad_dens_e_t + ne * P.grad_temp_e_t);
    const double gradPe_z = eQ * (P.temp_e * P.grad_dens_e_z + ne * P.grad_temp_e_z);
    const double Bmag = std::sqrt(Br*Br + Bt*Bt + Bzv*Bzv);
    const double invBmag = (Bmag > tiny) ? 1.0 / Bmag : 0.0;
    const double bhat_r = Br * invBmag;
    const double bhat_t = Bt * invBmag;
    const double bhat_z = Bzv * invBmag;
    const double epar =
      (Bmag > tiny) ? (-(gradPe_r*bhat_r + gradPe_t*bhat_t + gradPe_z*bhat_z) / (ne * eQ)) : 0.0;

    for (int iv = 0; iv < nvalue; ++iv) {
      double vout = 0.0;
      switch (value[iv]) {
        case BR:        vout = Br; break;
        case BT:        vout = Bt; break;
        case BZ:        vout = Bzv; break;
        case BX:        vout = Bx; break;
        case BY:        vout = By; break;
        case ER:        vout = Er; break;
        case ET:        vout = Et; break;
        case EZ:        vout = Ezv; break;
        case EX:        vout = Ex; break;
        case EY:        vout = Ey; break;
        case VR:        vout = Vr; break;
        case VT:        vout = Vt; break;
        case VZ:        vout = Vzv; break;
        case VX:        vout = Vx; break;
        case VY:        vout = Vy; break;
        case TI:        vout = P.temp_i; break;
        case TE:        vout = P.temp_e; break;
        case NI:        vout = P.dens_i; break;
        case NE:        vout = P.dens_e; break;
        case PARRFLOW:  vout = P.parr_flow; break;
        case EPAR:      vout = epar; break;
        case GRAD_TE_R: vout = P.grad_temp_e_r; break;
        case GRAD_TE_T: vout = P.grad_temp_e_t; break;
        case GRAD_TE_Z: vout = P.grad_temp_e_z; break;
        case GRAD_TI_R: vout = P.grad_temp_i_r; break;
        case GRAD_TI_T: vout = P.grad_temp_i_t; break;
        case GRAD_TI_Z: vout = P.grad_temp_i_z; break;
        default:        vout = 0.0; break;
      }
      if (nvalue == 1) vector_grid[icell] = vout;
      else row[iv] = vout;
    }
  }
}


/* ----------------------------------------------------------------------
   reallocate vector if nglocal has changed
   called by init() and load balancer
------------------------------------------------------------------------- */

void ComputePlasmaFields::reallocate() {
  if (grid->nlocal == nglocal) return;
  memory->destroy(vector_grid);
  memory->destroy(array_grid);
  nglocal = grid->nlocal;
  if (nvalue == 1) {
    memory->create(vector_grid, nglocal, "plasma/fields:vector_grid");
  } else {
    memory->create(array_grid, nglocal, nvalue, "plasma/fields:array_grid");
  }
}


/* ----------------------------------------------------------------------
   memory usage of local grid-based array
------------------------------------------------------------------------- */

bigint ComputePlasmaFields::memory_usage()
{
  bigint bytes = (bigint) nglocal * nvalue * sizeof(double);
  return bytes;
}


/* ----------------------------------------------------------------------
  read magnetic field data from file
------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
   Read plasma data from HDF5 file
------------------------------------------------------------------------- */
PlasmaFileData ComputePlasmaFields::readPlasmaFileData(const std::string& filePath) {
    printf("Reading plasma data from file: %s\n", filePath.c_str());
    PlasmaFileData data;

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
        data.r = read1D("r");
        data.z = read1D("z");
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

        // Load 2D fields with strict shape check
        data.dens_e        = read2D("dens_e");
        data.temp_e        = read2D("temp_e");
        data.dens_i        = read2D("dens_i");
        data.temp_i        = read2D("temp_i");

        data.parr_flow     = read2D("parr_flow");
        data.parr_flow_r   = read2D("parr_flow_r");
        data.parr_flow_t   = read2D("parr_flow_t");
        data.parr_flow_z   = read2D("parr_flow_z");

        data.grad_temp_e_r = read2D("grad_te_r");
        data.grad_temp_e_t = read2D("grad_te_t");
        data.grad_temp_e_z = read2D("grad_te_z");

        data.grad_temp_i_r = read2D("grad_ti_r");
        data.grad_temp_i_t = read2D("grad_ti_t");
        data.grad_temp_i_z = read2D("grad_ti_z");

    } catch (const H5::Exception& e) {
        fprintf(stderr, "HDF5 error: %s\n", e.getCDetailMsg());
        throw;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        throw;
    }

    return data;
}



/*----------------------------------------------------------------------
   broadcast plasma data
------------------------------------------------------------------------- */
void ComputePlasmaFields::broadcastPlasmaData(PlasmaFileData& data) {
    int me = comm->me;

    // Broadcast sizes of 1D vectors (r and z)
    int r_size = data.r.size();
    int z_size = data.z.size();
    MPI_Bcast(&r_size, 1, MPI_INT, 0, world);
    MPI_Bcast(&z_size, 1, MPI_INT, 0, world);

    // Resize vectors on non-root processes
    if (me != 0) {
        data.r.resize(r_size);
        data.z.resize(z_size);
    }

    // Broadcast 1D vector data
    MPI_Bcast(data.r.data(), r_size, MPI_DOUBLE, 0, world);
    MPI_Bcast(data.z.data(), z_size, MPI_DOUBLE, 0, world);

    // Broadcast 2D vectors (dens_e, temp_e, dens_i, temp_i, parr_flow, grad_temp_e, grad_temp_i)
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

    broadcast2DVector(data.dens_e);
    broadcast2DVector(data.temp_e);
    broadcast2DVector(data.dens_i);
    broadcast2DVector(data.temp_i);

    broadcast2DVector(data.parr_flow);
    broadcast2DVector(data.parr_flow_r);
    broadcast2DVector(data.parr_flow_t);
    broadcast2DVector(data.parr_flow_z);

    broadcast2DVector(data.grad_temp_e_r);
    broadcast2DVector(data.grad_temp_e_t);
    broadcast2DVector(data.grad_temp_e_z);

    broadcast2DVector(data.grad_temp_i_r);
    broadcast2DVector(data.grad_temp_i_t);
    broadcast2DVector(data.grad_temp_i_z);
}


/*----------------------------------------------------------------------
   broadcast magnetic field data
------------------------------------------------------------------------- */

void ComputePlasmaFields::broadcastMagneticData(MagneticFieldFileData& data) {
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
  broadcast2DVector(data.br);
  broadcast2DVector(data.bt);
  broadcast2DVector(data.bz);
}

/*----------------------------------------------------------------------
   bilinear interpolation of plasma data at cell center
------------------------------------------------------------------------- */

PlasmaFileParams ComputePlasmaFields::bilinearInterpolationPlasma(
    int icell, const PlasmaFileData &data)
{
  PlasmaFileParams P{};  // default all zeros
  if (icell < 0 || icell >= static_cast<int>(plasma_stencil.size())) return P;
  const BilinearStencil &s = plasma_stencil[icell];
  if (!s.valid) return P;

  P.temp_e = interpField2D(data.temp_e, s);
  P.dens_e = interpField2D(data.dens_e, s);
  P.temp_i = interpField2D(data.temp_i, s);
  P.dens_i = interpField2D(data.dens_i, s);
  gradField2D(data.dens_e, s, P.grad_dens_e_r, P.grad_dens_e_z);
  P.grad_dens_e_t = 0.0;
  P.grad_temp_e_r = interpField2D(data.grad_temp_e_r, s);
  P.grad_temp_e_t = interpField2D(data.grad_temp_e_t, s);
  P.grad_temp_e_z = interpField2D(data.grad_temp_e_z, s);
  P.grad_temp_i_r = interpField2D(data.grad_temp_i_r, s);
  P.grad_temp_i_t = interpField2D(data.grad_temp_i_t, s);
  P.grad_temp_i_z = interpField2D(data.grad_temp_i_z, s);
  P.parr_flow_r   = interpField2D(data.parr_flow_r, s);
  P.parr_flow_t   = interpField2D(data.parr_flow_t, s);
  P.parr_flow_z   = interpField2D(data.parr_flow_z, s);
  P.parr_flow     = interpField2D(data.parr_flow, s);

  return P;
}


/*---------------------------------
  Bilinear interpolation plasma
-----------------------------------*/

MagneticFieldFileDataParams
ComputePlasmaFields::bilinearInterpolationMagneticField(
    int icell, const MagneticFieldFileData &data)
{
  MagneticFieldFileDataParams B{};  // all zeros
  if (icell < 0 || icell >= static_cast<int>(magnetic_stencil.size())) return B;
  const BilinearStencil &s = magnetic_stencil[icell];
  if (!s.valid) return B;

  B.br = interpField2D(data.br, s);
  B.bt = interpField2D(data.bt, s);
  B.bz = interpField2D(data.bz, s);
  return B;
}

void ComputePlasmaFields::precomputeStencils(
    const std::vector<double> &r_vals,
    const std::vector<double> &z_vals,
    std::vector<BilinearStencil> &stencil)
{
  const int ncells = grid->nlocal;
  stencil.clear();
  stencil.resize(ncells);
  if (r_vals.size() < 2 || z_vals.size() < 2) return;

  const int dim = domain->dimension;
  const int nr = static_cast<int>(r_vals.size());
  const int nz = static_cast<int>(z_vals.size());

  for (int icell = 0; icell < ncells; ++icell) {
    BilinearStencil s{};
    Grid::ChildCell *cell = &grid->cells[icell];

    const double x = 0.5 * (cell->lo[0] + cell->hi[0]);
    const double y = 0.5 * (cell->lo[1] + cell->hi[1]);
    const double zc = (dim == 3)
                      ? 0.5 * (cell->lo[2] + cell->hi[2])
                      : 0.5 * (cell->lo[1] + cell->hi[1]);

    double r = 0.0;
    double z = 0.0;
    if (dim == 2) {
      r = x;
      z = y;
    } else {
      r = std::sqrt(x * x + y * y);
      z = zc;
    }

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

    s.ir1 = ir2 - 1;
    s.ir2 = ir2;
    s.iz1 = iz2 - 1;
    s.iz2 = iz2;

    const double R1 = r_vals[s.ir1];
    const double R2 = r_vals[s.ir2];
    const double Z1 = z_vals[s.iz1];
    const double Z2 = z_vals[s.iz2];
    const double denomR = R2 - R1;
    const double denomZ = Z2 - Z1;

    if (denomR == 0.0 || denomZ == 0.0) {
      s.w11 = 1.0;
      s.w21 = s.w12 = s.w22 = 0.0;
      s.valid = 1;
      stencil[icell] = s;
      continue;
    }

    const double t = (r_clamp - R1) / denomR;
    const double u = (z_clamp - Z1) / denomZ;
    s.t = t;
    s.u = u;
    s.inv_dR = 1.0 / denomR;
    s.inv_dZ = 1.0 / denomZ;
    s.w11 = (1.0 - t) * (1.0 - u);
    s.w21 = t * (1.0 - u);
    s.w12 = (1.0 - t) * u;
    s.w22 = t * u;
    s.valid = 1;
    stencil[icell] = s;
  }
}

double ComputePlasmaFields::interpField2D(
    const std::vector<std::vector<double>> &field,
    const BilinearStencil &s) const
{
  if (!s.valid || field.empty()) return 0.0;
  if (s.iz1 < 0 || s.iz2 >= static_cast<int>(field.size())) return 0.0;
  if (field[s.iz1].empty() || field[s.iz2].empty()) return 0.0;
  if (s.ir1 < 0 || s.ir2 >= static_cast<int>(field[s.iz1].size())) return 0.0;
  if (s.ir2 >= static_cast<int>(field[s.iz2].size())) return 0.0;

  const double q11 = field[s.iz1][s.ir1];
  const double q21 = field[s.iz1][s.ir2];
  const double q12 = field[s.iz2][s.ir1];
  const double q22 = field[s.iz2][s.ir2];
  return s.w11 * q11 + s.w21 * q21 + s.w12 * q12 + s.w22 * q22;
}

void ComputePlasmaFields::gradField2D(
    const std::vector<std::vector<double>> &field,
    const BilinearStencil &s,
    double &grad_r,
    double &grad_z) const
{
  grad_r = 0.0;
  grad_z = 0.0;
  if (!s.valid || field.empty()) return;
  if (s.iz1 < 0 || s.iz2 >= static_cast<int>(field.size())) return;
  if (field[s.iz1].empty() || field[s.iz2].empty()) return;
  if (s.ir1 < 0 || s.ir2 >= static_cast<int>(field[s.iz1].size())) return;
  if (s.ir2 >= static_cast<int>(field[s.iz2].size())) return;
  if (s.inv_dR <= 0.0 || s.inv_dZ <= 0.0) return;

  const double q11 = field[s.iz1][s.ir1];
  const double q21 = field[s.iz1][s.ir2];
  const double q12 = field[s.iz2][s.ir1];
  const double q22 = field[s.iz2][s.ir2];

  grad_r = ((1.0 - s.u) * (q21 - q11) + s.u * (q22 - q12)) * s.inv_dR;
  grad_z = ((1.0 - s.t) * (q12 - q11) + s.t * (q22 - q21)) * s.inv_dZ;
}


/* ----------------------------------------------------------------------
   read magnetic field data from file
------------------------------------------------------------------------- */
MagneticFieldFileData ComputePlasmaFields::readMagneticFieldFileData(const std::string& filePath) {
  printf("Reading magnetic field data from file: %s\n", filePath.c_str());
    MagneticFieldFileData data; // Initialize an empty MagneticFieldFileData struct
    try {
        H5::H5File file(filePath, H5F_ACC_RDONLY);
        
        auto read1DDataSet = [&file](const std::string& datasetPath) {
            H5::DataSet ds = file.openDataSet(datasetPath);
            H5::DataSpace space = ds.getSpace();
            std::vector<hsize_t> dims(1);
            space.getSimpleExtentDims(dims.data(), NULL);
            
            std::vector<double> data(dims[0]);
            ds.read(data.data(), H5::PredType::NATIVE_DOUBLE);
            return data;
        };
        
        auto read2DDataSet = [&file](const std::string& datasetPath) {
            H5::DataSet ds = file.openDataSet(datasetPath);
            H5::DataSpace space = ds.getSpace();
            std::vector<hsize_t> dims(2);
            space.getSimpleExtentDims(dims.data(), NULL);
            
            std::vector<std::vector<double>> data(dims[0], std::vector<double>(dims[1]));
            std::vector<double> rawData(dims[0] * dims[1]);
            ds.read(rawData.data(), H5::PredType::NATIVE_DOUBLE);
            
            for (hsize_t i = 0; i < dims[0]; ++i) {
                for (hsize_t j = 0; j < dims[1]; ++j) {
                    data[i][j] = rawData[i * dims[1] + j];
                }
            }
            return data;
        };
        
        // Read the required datasets
        data.r = read1DDataSet("r");
        data.z = read1DDataSet("z");
        data.br = read2DDataSet("br");
        data.bz = read2DDataSet("bz");
        data.bt = read2DDataSet("bt");
        file.close();
    } catch (const H5::Exception& e) {
        printf("Error reading magnetic field file file: %s\n", e.getCDetailMsg());
        throw;  // Re-throw the exception to handle it outside
    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
        throw;
    }
    printf("Finished reading magnetic field data from file: %s\n", filePath.c_str());
    return data;
}
