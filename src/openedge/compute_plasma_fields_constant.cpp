/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_plasma_fields_constant.h"
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

using namespace SPARTA_NS;

// user keywords
enum { MINDIST, SURFID, BX, BY, BZ, EX, EY, EZ, TI, TE, NI, NE, PARRFLOW, NESHEATH};

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

ComputePlasmaFieldsConstant::
ComputePlasmaFieldsConstant(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal compute plasma/fields command");

  per_grid_flag = 1;
  size_per_grid_cols = 0;

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0)
    error->all(FLERR,"Compute plasma/fields/constant group ID does not exist");
  groupbit = grid->bitmask[igroup];

  igroup = surf->find_group(arg[3]);
  if (igroup < 0)
    error->all(FLERR,"Compute plasma/fields/constant command surface group "
               "does not exist");
  sgroupbit = surf->bitmask[igroup];

  // optional args

  // defaults
  sdir[0]=sdir[1]=sdir[2]=0.0;

  // iarg points to first optional
  int iarg = 4;

  // optional: dir ax ay az
  if (iarg+3 < narg && strcmp(arg[iarg],"dir")==0) {
    sdir[0] = input->numeric(FLERR,arg[iarg+1]);
    sdir[1] = input->numeric(FLERR,arg[iarg+2]);
    sdir[2] = input->numeric(FLERR,arg[iarg+3]);
    if (domain->dimension == 2 && sdir[2] != 0.0)
      error->all(FLERR,"Illegal plasma/fields dir in 2d");
    iarg += 4;
  }

  // magnetic_field Bx By Bz 
  if (iarg+3 < narg && strcmp(arg[iarg],"magnetic_field")==0) {
    bconst[0] = input->numeric(FLERR,arg[iarg+1]);
    bconst[1] = input->numeric(FLERR,arg[iarg+2]);
    bconst[2] = input->numeric(FLERR,arg[iarg+3]);
    have_bconst = 1;
    iarg += 4;
  }
  // electric_field Ex Ey Ez  (optional)
  if (iarg+3 < narg && strcmp(arg[iarg],"electric_field")==0) {
    econst[0] = input->numeric(FLERR,arg[iarg+1]);
    econst[1] = input->numeric(FLERR,arg[iarg+2]);
    econst[2] = input->numeric(FLERR,arg[iarg+3]);
    have_econst = 1;
    iarg += 4;
  }
  // temp_e constant Te  (optional)
  if (iarg+1 < narg && strcmp(arg[iarg],"temp_e")==0) {
    teconst = input->numeric(FLERR,arg[iarg+1]);
    have_teconst = 1;
    iarg += 2;
  }
  // temp_i constant Ti  (optional)
  if (iarg+1 < narg && strcmp(arg[iarg],"temp_i")==0) {
    ticonst = input->numeric(FLERR,arg[iarg+1]);
    have_ticonst = 1;
    iarg += 2;
  }
  // dens_e constant ne  (optional)
  if (iarg+1 < narg && strcmp(arg[iarg],"dens_e")==0) {
    neconst = input->numeric(FLERR,arg[iarg+1]);
    have_neconst = 1;
    iarg += 2;
  }
  // dense ion constant ni  (optional)
  if (iarg+1 < narg && strcmp(arg[iarg],"dens_i")==0) {
    niconst = input->numeric(FLERR,arg[iarg+1]);
    have_niconst = 1;
    iarg += 2;
  }
  // parrflow constant parrflow  (optional)
  if (iarg+1 < narg && strcmp(arg[iarg],"parrflow")==0) {
    parrflowconst = input->numeric(FLERR,arg[iarg+1]);
    have_parrflowconst = 1;
    iarg += 2;
  }
  // sheath density constant nesheath  (optional)
  if (iarg+1 < narg && strcmp(arg[iarg],"nesheath")==0) {
    nesheathconst = input->numeric(FLERR,arg[iarg+1]);
    have_nesheathconst = 1;
    iarg += 2;
  }
  
  // now: at least one value keyword required
  if (iarg >= narg) error->all(FLERR,"plasma/fields needs values (mindist/surfid)");

   // collect value keywords
  nvalue = narg - iarg;
  value = new int[nvalue];
  bool want_bx=false, want_by=false, want_bz=false;

  for (int iv = 0; iv < nvalue; ++iv, ++iarg) {
    if      (strcmp(arg[iarg],"mindist")==0) value[iv] = MINDIST;
    else if (strcmp(arg[iarg],"surfid")==0)  value[iv] = SURFID;
    else if (strcmp(arg[iarg],"bx")==0)      { value[iv] = BX; want_bx=true; }
    else if (strcmp(arg[iarg],"by")==0)      { value[iv] = BY; want_by=true; }
    else if (strcmp(arg[iarg],"bz")==0)      { value[iv] = BZ; want_bz=true; }
    else if (strcmp(arg[iarg],"ex")==0)      { value[iv] = EX; }
    else if (strcmp(arg[iarg],"ey")==0)      { value[iv] = EY; }
    else if (strcmp(arg[iarg],"ez")==0)      { value[iv] = EZ; }
    else if (strcmp(arg[iarg],"temp_i")==0)      { value[iv] = TI; }
    else if (strcmp(arg[iarg],"temp_e")==0)      { value[iv] = TE; }
    else if (strcmp(arg[iarg],"dens_i")==0)      { value[iv] = NI; }
    else if (strcmp(arg[iarg],"dens_e")==0)      { value[iv] = NE; }
    else if (strcmp(arg[iarg],"parrflow")==0) { value[iv] = PARRFLOW; }
    else if (strcmp(arg[iarg],"nesheath")==0) { value[iv] = NESHEATH; }
    else error->all(FLERR,"Illegal plasma/fields value (use mindist/surfid/bx/by/bz/ex/ey/ez)");
  }


  // values parsed into value[0..nvalue-1]
  per_grid_flag = 1;
  size_per_grid_cols = nvalue;      // advertise #columns
  post_process_grid_flag = 1;       // tell SPARTA we’ll fill vector_grid per column

  // trivial mapping: each exposed column reads vals[icell][iv]
  nmap = new int[nvalue];
  memory->create(map, nvalue, 1, "plasma/fields:map");
  
  for (int iv = 0; iv < nvalue; ++iv) {
    nmap[iv] = 1;
    map[iv][0] = iv;
  }

  // storage
  nglocal = 0;
  vector_grid = NULL;
  vals = NULL;

}

/* ---------------------------------------------------------------------- */

ComputePlasmaFieldsConstant::~ComputePlasmaFieldsConstant()
{
  if (copymode) return;
  delete [] value;
  memory->destroy(vector_grid);
  memory->destroy(vals);
  delete [] nmap;
  memory->destroy(map);
  
}

/* ---------------------------------------------------------------------- */

void ComputePlasmaFieldsConstant::init()
{
  reallocate();
}

/* ---------------------------------------------------------------------- */

void ComputePlasmaFieldsConstant::compute_per_grid()
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

  memory->create(eflag,ntotal,"plasma/fields/constant:eflag");
  memory->create(slist,ntotal,"plasma/fields/constant:slist");

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

    // Base fields (global constants if present)
    const double Bx = have_bconst ? bconst[0] : 0.0;
    const double By = have_bconst ? bconst[1] : 0.0;
    const double Bz = have_bconst ? bconst[2] : 0.0;

    double Ex = have_econst ? econst[0] : 0.0;
    double Ey = have_econst ? econst[1] : 0.0;
    double Ez = have_econst ? econst[2] : 0.0;

    const double Ti = have_ticonst ? ticonst : 0.0;   // eV
    const double Te = have_teconst ? teconst : 0.0;   // eV
    const double ni = have_niconst ? niconst : 0.0;   // m^-3
    const double ne = have_neconst ? neconst : 0.0;   // m^-3
    const double parrflow = have_parrflowconst ? parrflowconst : 0.0;
    double nesheath = have_nesheathconst ? nesheathconst : 0.0;

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
            case EX:        vals[icell][iv] =  0.0; break;
            case EY:        vals[icell][iv] = 0.0; break;
            case EZ:        vals[icell][iv] = 0.0; break;
            case TI:        vals[icell][iv] = Ti; break;
            case TE:        vals[icell][iv] = Te; break;
            case NI:        vals[icell][iv] = ni; break;
            case NE:        vals[icell][iv] = ne; break;
            case PARRFLOW:  vals[icell][iv] = parrflow; break;
            case NESHEATH:  vals[icell][iv] = nesheath; break;
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

    // Physical constants
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
          default: break;
        }
      }
      continue;
    }

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
  const Out S = eval_ds_mps(d_m, Te, Ti, ne, B_mag, alpha_deg, MI_D);
  // store sheath density for chemistry/output
  nesheath = S.ne;

  // project field along -normal so ions accelerate toward the wall
  // double Ex, Ey, Ez;
  field_vector_along_minus_n(S, normal, Ex, Ey, Ez);
  double Emag = S.E_mag;
  // printf("ComputePlasmaFieldsConstant:: icell=%d d(m)=%g alpha_deg=%g Phi(V)=%g E_mag(V/m)=%g ne_sheath(m^-3)=%g\n",
         icell, d_m, alpha_deg, S.phi, Emag, nesheath);

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

void ComputePlasmaFieldsConstant::reallocate() {
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

bigint ComputePlasmaFieldsConstant::memory_usage()
{
  bigint bytes = 0;
  bytes += (bigint) nglocal * sizeof(double);          // vector_grid
  bytes += (bigint) nglocal * nvalue * sizeof(double); // vals
  return bytes;
}



int ComputePlasmaFieldsConstant::query_tally_grid(int index, double **&array, int *&cols)
{
  index--;
  int ivalue = index % nvalue;

  array = vals;            // <-- IMPORTANT: give SPARTA the data source
  cols  = map[ivalue];     // pick the column for this output
  return nmap[ivalue];     // (# inputs per output col, here 1)
}


void ComputePlasmaFieldsConstant::
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
Out ComputePlasmaFieldsConstant::eval_ds_mps(double d_m,        // distance from wall (m), d >= 0
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
  Out out{0.0, 0.0, ne0_m3};

  // basic guards
  if (d_m < 0.0) d_m = 0.0;
  if (Te_eV <= 0.0 || ne0_m3 <= 0.0 || mi_kg <= 0.0) return out;

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

  out.phi   = phi;
  out.E_mag = E_mag;
  out.ne    = ne;
  return out;
}

// Project the field along -n so ions accelerate toward the wall
void ComputePlasmaFieldsConstant::
field_vector_along_minus_n(const Out& o,
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
