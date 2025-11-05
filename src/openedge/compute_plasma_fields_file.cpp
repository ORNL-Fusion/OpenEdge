/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_plasma_fields_file.h"
#include "update.h"
#include "grid.h"
#include "surf.h"
#include "domain.h"
#include "input.h"
#include "geometry.h"
#include "math_extra.h"
#include "memory.h"
#include "error.h"
#include "sheath.h"
#include "comm.h"

#include <tuple>
#include <string>
#include <vector> 
#include <H5Cpp.h>



using namespace SPARTA_NS;

// user keywords
enum { MINDIST, SURFID, BX, BY, BZ, EX, EY, EZ, TI, TE, NI, NE, PARRFLOW, NESHEATH, GRAD_TE_R, GRAD_TE_T, GRAD_TE_Z, GRAD_TI_R, GRAD_TI_T, GRAD_TI_Z };

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

ComputePlasmaFieldsFile::
ComputePlasmaFieldsFile(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 6)
    error->all(FLERR,"Illegal compute plasma/fields/file (need grid_group surf_group file values...)");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute plasma/fields/file grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  igroup = surf->find_group(arg[3]);
  if (igroup < 0) error->all(FLERR,"Compute plasma/fields/file surface group ID does not exist");
  sgroupbit = surf->bitmask[igroup];

  // file path
  plasmaStatePath = std::string(arg[4]);  // make this a member in the header
  magneticFieldsPath = std::string(arg[5]);
  // defaults
  sdir[0] = sdir[1] = sdir[2] = 0.0;

  int iarg = 6; // first optional token after file path

  // optional: dir ax ay az
  if (iarg+3 < narg && strcmp(arg[iarg],"dir")==0) {
    sdir[0] = input->numeric(FLERR,arg[iarg+1]);
    sdir[1] = input->numeric(FLERR,arg[iarg+2]);
    sdir[2] = input->numeric(FLERR,arg[iarg+3]);
    if (domain->dimension == 2 && sdir[2] != 0.0)
      error->all(FLERR,"Illegal plasma/fields/file dir in 2d");
    iarg += 4;
  }

  if (iarg >= narg)
    error->all(FLERR,"plasma/fields/file needs values (mindist/surfid/te/ne/...)");

  // collect value keywords
  nvalue = narg - iarg;
  value = new int[nvalue];
  for (int iv = 0; iv < nvalue; ++iv, ++iarg) {
    if      (strcmp(arg[iarg],"mindist")==0) value[iv] = MINDIST;
    else if (strcmp(arg[iarg],"surfid")==0)  value[iv] = SURFID;
    else if (strcmp(arg[iarg],"bx")==0)      value[iv] = BX;
    else if (strcmp(arg[iarg],"by")==0)      value[iv] = BY;
    else if (strcmp(arg[iarg],"bz")==0)      value[iv] = BZ;
    else if (strcmp(arg[iarg],"ex")==0)      value[iv] = EX;
    else if (strcmp(arg[iarg],"ey")==0)      value[iv] = EY;
    else if (strcmp(arg[iarg],"ez")==0)      value[iv] = EZ;
    else if (strcmp(arg[iarg],"temp_i")==0)  value[iv] = TI;
    else if (strcmp(arg[iarg],"temp_e")==0)  value[iv] = TE;
    else if (strcmp(arg[iarg],"dens_i")==0)  value[iv] = NI;
    else if (strcmp(arg[iarg],"dens_e")==0)  value[iv] = NE;
    else if (strcmp(arg[iarg],"parrflow")==0) value[iv] = PARRFLOW;
    else if (strcmp(arg[iarg],"nesheath")==0) value[iv] = NESHEATH;
    else if (strcmp(arg[iarg],"grad_te_r")==0) value[iv] = GRAD_TE_R;
    else if (strcmp(arg[iarg],"grad_te_t")==0) value[iv] = GRAD_TE_T;
    else if (strcmp(arg[iarg],"grad_te_z")==0) value[iv] = GRAD_TE_Z;
    else if (strcmp(arg[iarg],"grad_ti_r")==0) value[iv] = GRAD_TI_R;
    else if (strcmp(arg[iarg],"grad_ti_t")==0) value[iv] = GRAD_TI_T;
    else if (strcmp(arg[iarg],"grad_ti_z")==0) value[iv] = GRAD_TI_Z;
    else error->all(FLERR,"Illegal plasma/fields/file value (mindist|surfid|bx|...|nesheath)");
  }

  per_grid_flag = 1;
  size_per_grid_cols = nvalue;
  post_process_grid_flag = 1;

  nmap = new int[nvalue];
  memory->create(map, nvalue, 1, "plasma/fields:file:map");
  for (int iv = 0; iv < nvalue; ++iv) { nmap[iv] = 1; map[iv][0] = iv; }

  nglocal = 0;
  vector_grid = NULL;
  vals = NULL;

}

/* ---------------------------------------------------------------------- */

ComputePlasmaFieldsFile::~ComputePlasmaFieldsFile()
{
  if (copymode) return;
  delete [] value;
  memory->destroy(vector_grid);
  memory->destroy(vals);
  delete [] nmap;
  memory->destroy(map);
  
}

/* ---------------------------------------------------------------------- */

void ComputePlasmaFieldsFile::init()
{
  reallocate();

  const int me     = comm->me;
  const int ncells = grid->nlocal;

  // 1) load on rank 0, broadcast to all ranks
  if (me == 0) {
      plasma_data = readPlasmaFileData(plasmaStatePath);
  }
  broadcastPlasmaData(plasma_data);

  // read magnetic field data
  if (me == 0) {
      magnetic_data = readMagneticFieldFileData(magneticFieldsPath);
  }
  broadcastMagneticData(magnetic_data);

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  const int dim = domain->dimension;

  for (int icell = 0; icell < ncells; ++icell) {
    if (!(cinfo[icell].mask & groupbit)) continue;   // respect the grid group
    if (cells[icell].nsplit < 1)         continue;   // skip empty

    // cell center (axisym note: use y for 2D, z for 3D)
    double *lo = cells[icell].lo;
    double *hi = cells[icell].hi;
    const double R = 0.5 * (lo[0] + hi[0]);
    const double Z = (dim == 2) ? 0.5 * (lo[1] + hi[1])
                                : 0.5 * (lo[2] + hi[2]);

    // sample all fields from the file at (R,Z)
    PlasmaFileParams P = bilinearInterpolationPlasma(icell, plasma_data);
    plasma_map[icell] = P;

    // sample magnetic field at (R,Z)
    MagneticFieldFileDataParams B = bilinearInterpolationMagneticField(icell, magnetic_data);
    magnetic_map[icell] = B;
  }

}

/* ---------------------------------------------------------------------- */

void ComputePlasmaFieldsFile::compute_per_grid()
{
  using std::abs;
  using std::max;
  using std::min;

  int i,m,n;
  int *csubs;
  surfint *csurfs;
  double dist,mindist;
  double *p1,*p2,*p3,*lo,*hi;
  double cctr[3],cell2surf[3];

  invoked_per_grid = update->ntimestep;

  const int dim = domain->dimension;
  Surf::Line *lines = surf->lines;
  Surf::Tri  *tris  = surf->tris;
  const int ntotal = surf->nsurf;

  // --- 0) Safe defaults so readers never see garbage -------------------------
  for (int ic = 0; ic < nglocal; ++ic)
    for (int iv = 0; iv < nvalue; ++iv)
      vals[ic][iv] = (value[iv] == MINDIST) ? BIG : -1.0;

  // --- 1) Build eligible-surface list (respect sgroupbit and visibility) -----
  int *eflag,*slist;
  int nsurf = 0;

  memory->create(eflag,ntotal,"plasma/fields/file:eflag");
  memory->create(slist,ntotal,"plasma/fields/file:slist");

  if (dim == 2) {
    for (i = 0; i < ntotal; i++) {
      eflag[i] = 0;
      if (!(lines[i].mask & sgroupbit)) continue;
      if (MathExtra::dot3(lines[i].norm,sdir) <= 0.0) { // “visible” side
        eflag[i] = 1;
        slist[nsurf++] = i;           // store ARRAY INDEX m
      }
    }
  } else {
    for (i = 0; i < ntotal; i++) {
      eflag[i] = 0;
      if (!(tris[i].mask & sgroupbit)) continue;
      if (MathExtra::dot3(tris[i].norm,sdir) <= 0.0) {
        eflag[i] = 1;
        slist[nsurf++] = i;           // store ARRAY INDEX m
      }
    }
  }

  // --- 2) Precompute eligible surface "centers" (for visibility prefilter) ---
  const double invthird = 1.0/3.0;
  double **sctr;
  memory->create(sctr,nsurf,3,"plasma/fields:sctr");
  for (i = 0; i < nsurf; i++) {
    m = slist[i];
    if (dim == 2) {
      p1 = lines[m].p1; p2 = lines[m].p2;
      sctr[i][0] = 0.5*(p1[0]+p2[0]);
      sctr[i][1] = 0.5*(p1[1]+p2[1]);
      sctr[i][2] = 0.0;
    } else {
      p1 = tris[m].p1; p2 = tris[m].p2; p3 = tris[m].p3;
      sctr[i][0] = invthird*(p1[0]+p2[0]+p3[0]);
      sctr[i][1] = invthird*(p1[1]+p2[1]+p3[1]);
      sctr[i][2] = invthird*(p1[2]+p2[2]+p3[2]);
    }
  }

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Grid::SplitInfo *sinfo = grid->sinfo;

  // --- 3) Loop over cells -----------------------------------------------------
  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit < 1) continue;

    // Base fields (global files if present)

    // --- Base fields from file (self-contained map) --------------------------
    PlasmaFileParams P = {};
    auto it = plasma_map.find(icell);
    if (it != plasma_map.end()) P = it->second;

    const double Ti = P.temp_i;                    // eV
    const double Te = P.temp_e;                    // eV
    const double ni = P.dens_i;                    // m^-3
    const double ne = P.dens_e;                    // m^-3
    const double parrflow = P.parr_flow;           // units: your file
    double grad_te_r = P.grad_temp_e_r;
    double grad_te_t = P.grad_temp_e_t;
    double grad_te_z = P.grad_temp_e_z;
    double grad_ti_r = P.grad_temp_i_r;
    double grad_ti_t = P.grad_temp_i_t;
    double grad_ti_z = P.grad_temp_i_z;


   // Magnetic field is NOT in the file (per your note) → hard-set to zero
   MagneticFieldFileDataParams B = {};
    auto itb = magnetic_map.find(icell);
    if (itb != magnetic_map.end()) B = itb->second;
    const double Bx = B.br;    // T
    const double By = B.bt;    // T
    const double Bz = B.bz;    // T

    // Electric field will be set by the sheath model; default to zero
    double Ex = 0.0, Ey = 0.0, Ez = 0.0;
    
    double nesheath = 0; // have_nesheathconst ? nesheathconst : 0.0;

    // --- 3a) Quick overlap branch: any eligible face exactly in this cell ----
    // If overlap: set dist=0, SURFID to some eligible touching face, ZERO E, propagate
    int found = -1; // ARRAY INDEX into lines[]/tris[]
    if (cells[icell].nsurf) {
      n = cells[icell].nsurf;
      csurfs = cells[icell].csurfs;
      for (i = 0; i < n; i++) {
        m = csurfs[i];                // ARRAY index
        if (eflag[m]) { found = m; break; }  // first eligible touching face is fine
      }
      if (found >= 0) {
        const int sid = (dim == 2) ? lines[found].id : tris[found].id;

        // outputs
        for (int iv = 0; iv < nvalue; iv++) {
          switch (value[iv]) {
            case MINDIST:   vals[icell][iv] = 0.0; break;
            case SURFID:    vals[icell][iv] = (double)sid; break;
            case BX:        vals[icell][iv] = Bx; break;
            case BY:        vals[icell][iv] = By; break;
            case BZ:        vals[icell][iv] = Bz; break;
            case EX:        vals[icell][iv] = Ex; break;
            case EY:        vals[icell][iv] = Ey; break;
            case EZ:        vals[icell][iv] = Ez; break;
            case TI:        vals[icell][iv] = Ti; break;
            case TE:        vals[icell][iv] = Te; break;
            case NI:        vals[icell][iv] = ni; break;
            case NE:        vals[icell][iv] = ne; break;
            case PARRFLOW:  vals[icell][iv] = parrflow; break;
            case NESHEATH:  vals[icell][iv] = nesheath; break;
            case GRAD_TE_R: vals[icell][iv] = grad_te_r; break;
            case GRAD_TE_T: vals[icell][iv] = grad_te_t; break;
            case GRAD_TE_Z: vals[icell][iv] = grad_te_z; break;
            case GRAD_TI_R: vals[icell][iv] = grad_ti_r; break;
            case GRAD_TI_T: vals[icell][iv] = grad_ti_t; break;
            case GRAD_TI_Z: vals[icell][iv] = grad_ti_z; break;
            default:        /* keep default */ break;
          }
        }

        // propagate to sub-cells if split
        if (cells[icell].nsplit > 1) {
          n = cells[icell].nsplit;
          csubs = sinfo[cells[icell].isplit].csubs;
          for (i = 0; i < n; i++) {
            const int sub = csubs[i];
            for (int iv = 0; iv < nvalue; iv++)
              vals[sub][iv] = vals[icell][iv];
          }
        }
        continue; // done with this cell
      }
    }

    // --- 3b) Compute cell center ---------------------------------------------
    lo = cells[icell].lo; hi = cells[icell].hi;
    cctr[0] = 0.5*(lo[0]+hi[0]);
    cctr[1] = 0.5*(lo[1]+hi[1]);
    cctr[2] = (dim==3) ? 0.5*(lo[2]+hi[2]) : 0.0;

    // --- 3c) Scan eligible surfaces, choose NEAREST (keep index + ID) --------
    mindist = BIG;
    int closest_surf_idx = -1; // ARRAY index → use for geometry
    int closest_surf_id  = -1; // user-visible ID → for output

    for (i = 0; i < nsurf; i++) {
      m = slist[i]; // ARRAY index
      // cheap back-side reject using center-to-center vector
      cell2surf[0] = sctr[i][0] - cctr[0];
      cell2surf[1] = sctr[i][1] - cctr[1];
      cell2surf[2] = sctr[i][2] - cctr[2];

      if (dim == 2) {
        if (MathExtra::dot3(cell2surf,lines[m].norm) > 0.0) continue;
        dist = Geometry::dist_line_quad(lines[m].p1,lines[m].p2,lo,hi);
      } else {
        if (MathExtra::dot3(cell2surf,tris[m].norm) > 0.0) continue;
        dist = Geometry::dist_tri_hex(tris[m].p1,tris[m].p2,tris[m].p3,
                                      tris[m].norm,lo,hi);
      }

      if (dist < mindist) {
        mindist = dist;
        closest_surf_idx = m;
        closest_surf_id  = (dim == 2) ? lines[m].id : tris[m].id;
      }
    }

    // --- 3d) Sheath model inputs & guards ------------------------------------
    // Physical files
    const double q_e  = update->echarge;     // 1.602e-19 C
    const double eps0 = update->epsilon_0;   // 8.854e-12 F/m

    constexpr double NE_FLOOR_FOR_LD = 1e10;                  // pick your floor
    const double ne_for_lambda = std::max(ne, NE_FLOOR_FOR_LD);
    double lambda_D_m = (Te > 0.0) ? std::sqrt( (eps0 * Te) / (ne_for_lambda * q_e) ) : 0.0;

    // write mindist & surfid regardless (already defaulted above)
    for (int iv = 0; iv < nvalue; ++iv) {
      if (value[iv] == MINDIST) vals[icell][iv] = mindist;
      else if (value[iv] == SURFID) vals[icell][iv] = (double)closest_surf_id;
    }

    // If no valid neighbor or invalid plasma inputs: zero E and finalize
    if (closest_surf_idx < 0 || lambda_D_m <= 0.0 || !(mindist < BIG)) {
      // Ex = Ey = Ez = 0.0;
      // write remaining fields
      for (int iv = 0; iv < nvalue; ++iv) {
        switch (value[iv]) {
          case BX: vals[icell][iv] = Bx; break;
          case BY: vals[icell][iv] = By; break;
          case BZ: vals[icell][iv] = Bz; break;
          case EX: vals[icell][iv] = Ex; break;
          case EY: vals[icell][iv] = Ey; break;
          case EZ: vals[icell][iv] = Ez; break;
          case TI: vals[icell][iv] = Ti; break;
          case TE: vals[icell][iv] = Te; break;
          case NI: vals[icell][iv] = ni; break;
          case NE: vals[icell][iv] = ne; break;
          case PARRFLOW: vals[icell][iv] = parrflow; break;
          case NESHEATH: vals[icell][iv] = nesheath; break;
          case GRAD_TE_R: vals[icell][iv] = grad_te_r; break;
          case GRAD_TE_T: vals[icell][iv] = grad_te_t; break;
          case GRAD_TE_Z: vals[icell][iv] = grad_te_z; break;
          case GRAD_TI_R: vals[icell][iv] = grad_ti_r; break;
          case GRAD_TI_T: vals[icell][iv] = grad_ti_t; break;
          case GRAD_TI_Z: vals[icell][iv] = grad_ti_z; break;
          default: break;
        }
      }
      continue;
    }

   // print icell and te 
  //  printf("icell=%d te=%g ne=%g mindist=%g\n", icell, Te, ne, mindist); 
    // distance in Debye lengths
    const double QE   = 1.602176634e-19;      // C
    const double KB   = 1.380649e-23;         // J/K
    const double MP   = 1.67262192369e-27;    // kg
    const double MI_D = 2.0 * MP;             // deuteron mass [kg]
    double normal[3];
    if (dim == 2) {
      normal[0] = lines[closest_surf_idx].norm[0];
      normal[1] = lines[closest_surf_idx].norm[1];
      normal[2] = 0.0;
    } else {
      normal[0] = tris[closest_surf_idx].norm[0];
      normal[1] = tris[closest_surf_idx].norm[1];
      normal[2] = tris[closest_surf_idx].norm[2];
    }

    // grazing angle α (deg): α = 90° - angle(B,n)
    const double B_mag = sqrt(Bx*Bx + By*By + Bz*Bz);
    double alpha_deg = 0.0;
    if (B_mag > 0.0) {
      double cos_theta = (Bx*normal[0] + By*normal[1] + Bz*normal[2]) / B_mag;
      cos_theta = max(-1.0, min(1.0, cos_theta));
      alpha_deg = 90.0 - std::acos(cos_theta) * 180.0/M_PI;
      if (alpha_deg < 0.0) alpha_deg = 0.0;
    }
    
  const double d_m      = std::max(0.0, mindist);
  const SheathParams S = eval_ds_mps(d_m, Te, Ti, ne, B_mag, alpha_deg, MI_D);
  // store sheath density for chemistry/output
  nesheath = S.ne;

  // project field along -normal so ions accelerate toward the wall
  // double Ex, Ey, Ez;
  field_vector_along_minus_n(S, normal, Ex, Ey, Ez);
  double Emag = S.E_mag;

    // --- 3e) Write all requested outputs for this cell -----------------------
    for (int iv = 0; iv < nvalue; ++iv) {
      switch (value[iv]) {
        case BX:        vals[icell][iv] = Bx; break;
        case BY:        vals[icell][iv] = By; break;
        case BZ:        vals[icell][iv] = Bz; break;
        case EX:        vals[icell][iv] = Ex; break;
        case EY:        vals[icell][iv] = Ey; break;
        case EZ:        vals[icell][iv] = Ez; break;
        case TI:        vals[icell][iv] = Ti; break;
        case TE:        vals[icell][iv] = Te; break;
        case NI:        vals[icell][iv] = ni; break;
        case NE:        vals[icell][iv] = ne; break;
        case PARRFLOW:  vals[icell][iv] = parrflow; break;
        case NESHEATH: vals[icell][iv] = nesheath; break;
        case GRAD_TE_R: vals[icell][iv] = grad_te_r; break;
        case GRAD_TE_T: vals[icell][iv] = grad_te_t; break;
        case GRAD_TE_Z: vals[icell][iv] = grad_te_z; break;
        case GRAD_TI_R: vals[icell][iv] = grad_ti_r; break;
        case GRAD_TI_T: vals[icell][iv] = grad_ti_t; break;
        case GRAD_TI_Z: vals[icell][iv] = grad_ti_z; break;
        // MINDIST and SURFID already set above
        default: break;
      }
    }
  } // end cell loop

  // --- 4) Clean up -----------------------------------------------------------
  memory->destroy(eflag);
  memory->destroy(slist);
  memory->destroy(sctr);
}

/* ----------------------------------------------------------------------
   reallocate vector if nglocal has changed
   called by init() and load balancer
------------------------------------------------------------------------- */

void ComputePlasmaFieldsFile::reallocate() {
  if (grid->nlocal == nglocal) return;
  memory->destroy(vector_grid);
  memory->destroy(vals);
  nglocal = grid->nlocal;
  memory->create(vector_grid, nglocal,                "plasma/fields:vector_grid");
  memory->create(vals,        nglocal, nvalue,        "plasma/fields:vals");
}


/* ----------------------------------------------------------------------
   memory usage of local grid-based array
------------------------------------------------------------------------- */

bigint ComputePlasmaFieldsFile::memory_usage()
{
  bigint bytes = 0;
  bytes += (bigint) nglocal * sizeof(double);          // vector_grid
  bytes += (bigint) nglocal * nvalue * sizeof(double); // vals
  return bytes;
}



int ComputePlasmaFieldsFile::query_tally_grid(int index, double **&array, int *&cols)
{
  index--;
  int ivalue = index % nvalue;

  array = vals;            // <-- IMPORTANT: give SPARTA the data source
  cols  = map[ivalue];     // pick the column for this output
  return nmap[ivalue];     // (# inputs per output col, here 1)
}


void ComputePlasmaFieldsFile::
post_process_grid(int index, int nsample,
                  double **etally, int *emap, double *vec, int nstride)
{
  index--;                 // SPARTA passes 1-based column index
  int iv = index % nvalue; // which keyword for this column

  if (!etally) {          // SPARTA asks us to fill our own buffer
    etally = vals;
    vec    = vector_grid;
    nstride = 1;
  }

  int src = emap[0];      // which column in vals to read (== iv)
  for (int ic = 0; ic < nglocal; ++ic) {
    vec[ic*nstride] = etally[ic][src];
  }
}





// / core evaluator
SheathParams ComputePlasmaFieldsFile::eval_ds_mps(double d_m,        // distance from wall (m), d >= 0
                       double Te_eV,      // electron temperature (eV)
                       double Ti_eV,      // ion temperature (eV)
                       double ne0_m3,     // upstream electron density (m^-3)
                       double B_T,        // magnetic field magnitude (T)
                       double alpha_deg,  // grazing angle in degrees
                       double mi_kg) // ion mass (kg)
{

   constexpr double QE   = 1.602176634e-19;     // C
   constexpr double EPS0 = 8.8541878128e-12;    // F/m
   constexpr double MP   = 1.67262192369e-27;   // kg

  double pot_mult = 3.0;
  SheathParams sheathParams{0.0, 0.0, ne0_m3};

  // basic guards
  if (d_m < 0.0) d_m = 0.0;
  if (Te_eV <= 0.0 || ne0_m3 <= 0.0 || mi_kg <= 0.0) return sheathParams;

  const double fd = fd_poly_deg(alpha_deg);
  const double pot = pot_mult * Te_eV; // volts (1 eV ≡ 1 V)

  // Debye length (Te in eV): lambdaD = sqrt( EPS0 * Te / (ne0 * QE) )
  const double lambdaD = std::sqrt(EPS0 * Te_eV / (ne0_m3 * QE));

  // ion-sound speed and ion gyro radius
  const double cs       = std::sqrt(std::max(Te_eV + Ti_eV, 0.0) * QE / mi_kg);
  const double omega_ci = (B_T > 0.0) ? (QE * B_T / mi_kg) : 0.0;
  const double rho_i    = (omega_ci > 0.0) ? (cs / omega_ci) : 1e300; // large if B=0
  const double L_MPS    = rho_i;

  // exponentials (avoid overflow)
  const double e_DS  = std::exp(- d_m / (2.0 * lambdaD));
  const double e_MPS = std::exp(- d_m / std::max(L_MPS, 1e-300));

  // potential (negative toward the wall)
  const double phi = - pot * ( fd * e_DS + (1.0 - fd) * e_MPS );

  // field magnitude (positive)
  const double E_DS  =  pot * ( fd / (2.0 * lambdaD) ) * e_DS;
  const double E_MPS =  pot * ( (1.0 - fd) / std::max(L_MPS, 1e-300) ) * e_MPS;
  const double E_mag =  std::abs(E_DS + E_MPS);

  // Boltzmann electrons (clamp exponent for safety)
  double x = phi / std::max(Te_eV, 1e-300);
  x = std::max(-100.0, std::min(50.0, x));
  const double ne = ne0_m3 * std::exp(x);

  sheathParams.phi   = phi;
  sheathParams.E_mag = E_mag;
  sheathParams.ne    = ne;
  return sheathParams;
}

// Project the field along -n so ions accelerate toward the wall
void ComputePlasmaFieldsFile::
field_vector_along_minus_n(const SheathParams& o,
                           const double n[3],
                           double& Ex, double& Ey, double& Ez)
{
  // normalize n
  const double nn = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  if (nn <= 0.0) { Ex = Ey = Ez = 0.0; return; }
  const double nx = n[0]/nn, ny = n[1]/nn, nz = n[2]/nn;

  // E points toward the surface: E = -|E| * n_hat
  Ex += - o.E_mag * nx;
  Ey += - o.E_mag * ny;
  Ez += - o.E_mag * nz;
}


/* ----------------------------------------------------------------------
  read magnetic field data from file
------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
   Read plasma data from HDF5 file
------------------------------------------------------------------------- */
PlasmaFileData ComputePlasmaFieldsFile::readPlasmaFileData(const std::string& filePath) {
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

    printf("Finished reading plasma data from file: %s\n", filePath.c_str());
    return data;
}



/*----------------------------------------------------------------------
   broadcast plasma data
------------------------------------------------------------------------- */
void ComputePlasmaFieldsFile::broadcastPlasmaData(PlasmaFileData& data) {
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

void ComputePlasmaFieldsFile::broadcastMagneticData(MagneticFieldFileData& data) {
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

// bilinearInterpolationMagneticField
/*--------------------------------- 
  Bilinear interpolation plasma
-----------------------------------*/

PlasmaFileParams ComputePlasmaFieldsFile::bilinearInterpolationPlasma(int icell, const PlasmaFileData& data) {
    // Check cache
    auto it = plasmaDataCache.find(icell);
    if (it != plasmaDataCache.end()) return it->second;

    // Validate coordinate arrays
    if (data.r.empty() || data.z.empty()) {
        throw std::runtime_error("Plasma data coordinate arrays are empty.");
    }

    const auto& r_vals = data.r;
    const auto& z_vals = data.z;

    // Get cell midpoint
    Grid::ChildCell* cell = &grid->cells[icell];
    double r = 0.5 * (cell->lo[0] + cell->hi[0]);
    double z = 0.5 * (cell->lo[1] + cell->hi[1]);

    // Out-of-bounds check
    if (r < r_vals.front() || r > r_vals.back() || z < z_vals.front() || z > z_vals.back()) {
        return PlasmaFileParams{}; // return zeroed struct
    }

    // Locate surrounding grid indices
    auto r_it = std::lower_bound(r_vals.begin(), r_vals.end(), r);
    auto z_it = std::lower_bound(z_vals.begin(), z_vals.end(), z);
    int r1 = std::max(0, static_cast<int>(r_it - r_vals.begin()) - 1);
    int r2 = std::min(static_cast<int>(r_vals.size()) - 1, r1 + 1);
    int z1 = std::max(0, static_cast<int>(z_it - z_vals.begin()) - 1);
    int z2 = std::min(static_cast<int>(z_vals.size()) - 1, z1 + 1);

    double R1 = r_vals[r1], R2 = r_vals[r2];
    double Z1 = z_vals[z1], Z2 = z_vals[z2];
    double denom = (R2 - R1) * (Z2 - Z1);

    // Bilinear interpolation lambda
    auto interp = [&](const std::vector<std::vector<double>>& field) -> double {
        if (field.size() <= z2 || field[0].size() <= r2) return 0.0;

        double Q11 = field[z1][r1];
        double Q21 = field[z1][r2];
        double Q12 = field[z2][r1];
        double Q22 = field[z2][r2];

        if (denom == 0.0) return (Q11 + Q21 + Q12 + Q22) / 4.0;

        return (
            Q11 * (R2 - r) * (Z2 - z) +
            Q21 * (r - R1) * (Z2 - z) +
            Q12 * (R2 - r) * (z - Z1) +
            Q22 * (r - R1) * (z - Z1)
        ) / denom;
    };

    // Fill and cache interpolated values
    PlasmaFileParams result;
    result.dens_e        = interp(data.dens_e);
    result.temp_e        = interp(data.temp_e);
    result.dens_i        = interp(data.dens_i);
    result.temp_i        = interp(data.temp_i);
    result.parr_flow     = interp(data.parr_flow);
    result.parr_flow_r   = interp(data.parr_flow_r);
    result.parr_flow_t   = interp(data.parr_flow_t);
    result.parr_flow_z   = interp(data.parr_flow_z);
    result.grad_temp_e_r = interp(data.grad_temp_e_r);
    result.grad_temp_e_t = interp(data.grad_temp_e_t);
    result.grad_temp_e_z = interp(data.grad_temp_e_z);
    result.grad_temp_i_r = interp(data.grad_temp_i_r);
    result.grad_temp_i_t = interp(data.grad_temp_i_t);
    result.grad_temp_i_z = interp(data.grad_temp_i_z);

    plasmaDataCache[icell] = result;
    return result;
}

/*---------------------------------
  Bilinear interpolation plasma
-----------------------------------*/
MagneticFieldFileDataParams ComputePlasmaFieldsFile::bilinearInterpolationMagneticField(int icell, const MagneticFieldFileData& data) {
   // Check if the result is already cached
   auto cache_it = magneticFieldDataCache.find(icell);
   if (cache_it != magneticFieldDataCache.end()) {
       return cache_it->second;
   }

   if (data.r.empty() || data.z.empty()) {
       printf("Data arrays are empty.\n");
       throw std::runtime_error("Data arrays are empty.");
   }

   const std::vector<double>& r_values = data.r;
   const std::vector<double>& z_values = data.z;

   // Access cell and calculate midpoints
   Grid::ChildCell* cell = &grid->cells[icell];
   double r_val = 0.5 * (cell->lo[0] + cell->hi[0]);
   double z_val = 0.5 * (cell->lo[1] + cell->hi[1]);

   // Ensure r and z are within the data bounds
   if (r_val < r_values.front() || r_val > r_values.back() ||
       z_val < z_values.front() || z_val > z_values.back()) {
       // printf("Interpolation point (r_val: %f, z_val: %f) is outside the bounds of the data grid.\n", r_val, z_val);
       MagneticFieldFileDataParams params = {};
       return params;
   }


    // Locate indices for surrounding grid points
    auto r_it = std::lower_bound(r_values.begin(), r_values.end(), r_val);
    auto z_it = std::lower_bound(z_values.begin(), z_values.end(), z_val);

    int r1_idx = std::max(0, int(r_it - r_values.begin()) - 1);
    int r2_idx = std::min(int(r_values.size()) - 1, r1_idx + 1);
    int z1_idx = std::max(0, int(z_it - z_values.begin()) - 1);
    int z2_idx = std::min(int(z_values.size()) - 1, z1_idx + 1);

    // Get surrounding grid values
    double r1 = r_values[r1_idx];
    double r2 = r_values[r2_idx];
    double z1 = z_values[z1_idx];
    double z2 = z_values[z2_idx];

    // Lambda for bilinear interpolation
    auto bilinearInterpolation = [&](const std::vector<std::vector<double>>& field) -> double {
        double Q11 = field[z1_idx][r1_idx];
        double Q12 = field[z2_idx][r1_idx];
        double Q21 = field[z1_idx][r2_idx];
        double Q22 = field[z2_idx][r2_idx];
        double denom = (r2 - r1) * (z2 - z1);

        // Check for zero denominators
        if (denom == 0.0) {
            return (Q11 + Q12 + Q21 + Q22) / 4.0;
        }

        return (Q11 * (r2 - r_val) * (z2 - z_val) +
                Q21 * (r_val - r1) * (z2 - z_val) +
                Q12 * (r2 - r_val) * (z_val - z1) +
                Q22 * (r_val - r1) * (z_val - z1)) / denom;
    };


   // pusher_Bororm bilinear interpolation for each field component
    MagneticFieldFileDataParams params;
    params.br = bilinearInterpolation(data.br);
    params.bt = bilinearInterpolation(data.bt);
    params.bz = bilinearInterpolation(data.bz);

  
 // Cache the result
    magneticFieldDataCache[icell] = params;

   return params;
}

/* ----------------------------------------------------------------------
   read magnetic field data from file
------------------------------------------------------------------------- */
MagneticFieldFileData ComputePlasmaFieldsFile::readMagneticFieldFileData(const std::string& filePath) {
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