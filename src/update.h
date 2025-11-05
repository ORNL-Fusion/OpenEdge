/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifndef SPARTA_UPDATE_H
#define SPARTA_UPDATE_H

#include "math.h"
#include "pointers.h"

#include <tuple>
#include <map>
#include <string>
#include <vector> 
#include <H5Cpp.h>
#include <map>
#include <string>
#include "particle.h"
#include <unordered_map>

#define BIG 1.0e20
// Define a hash function for std::tuple
namespace std {
  template <>
  struct hash<std::tuple<double, double>> {
      size_t operator()(const std::tuple<double, double>& t) const {
          auto hash1 = std::hash<double>{}(std::get<0>(t));
          auto hash2 = std::hash<double>{}(std::get<1>(t));
          return hash1 ^ (hash2 << 1);
      }
  };
}

// Define a hash function for a pair of doubles to use as cache keys
struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
      return std::hash<T1>()(pair.first) ^ (std::hash<T2>()(pair.second) << 1);
  }
};

namespace SPARTA_NS {

  // Structs for plasma data and parameters
struct PlasmaData{
  std::vector<double> r;   
  std::vector<double> z;  
  std::vector<std::vector<double>> dens_e, temp_e;
  std::vector<std::vector<double>> dens_i, temp_i;
  std::vector<std::vector<double>> parr_flow_r, parr_flow_t, parr_flow_z, parr_flow;
  std::vector<std::vector<double>> grad_temp_e_r, grad_temp_e_t, grad_temp_e_z;
  std::vector<std::vector<double>> grad_temp_i_r, grad_temp_i_t, grad_temp_i_z;
};

struct PlasmaDataParams {
  double dens_e;
  double temp_e;
  double dens_i;
  double temp_i;
  double parr_flow;
  double parr_flow_r;
  double parr_flow_t;
  double parr_flow_z;
  double grad_temp_e_r;
  double grad_temp_e_t;
  double grad_temp_e_z;
  double grad_temp_i_r;
  double grad_temp_i_t;
  double grad_temp_i_z;
};


// Structs for magnetic field data and parameters
struct MagneticFieldData {
  std::vector<double> r;   
  std::vector<double> z;  
  std::vector<std::vector<double>> br;
  std::vector<std::vector<double>> bt;
  std::vector<std::vector<double>> bz;
};
struct MagneticFieldDataParams {
  double br;
  double bt;
  double bz;
  double r;
  double z;
};


class Update : protected Pointers {
 public:

struct BoundaryInfo {
  double minDistance{ -1.0 };
  double angle{ 0.0 };
  int    surf_local{ -1 };   // index into surf->lines (2D) or surf->tris (3D)
  int    surf_id{ -1 };      // external ID from file (lines[i].id / tris[i].id)
  int    type_group{ 0 }, mask_group{ 0 };
  double normal[3]{ 0.0, 0.0, 1.0 };
  double surf_center[3]{ 0.0, 0.0, 0.0 };
  double plane_d{ 0.0 };
};

struct SurfHit2D {
  int    surf_local = -1;   // local index in surf->lines
  int    file_id    = -1;   // line ID from file (if present)
  double footR = 0.0, footZ = 0.0;
  double dist  = BIG;
  double nx = 0.0, nz = 0.0; // outward normal (unit)
  int    type = 0, mask = 0;
};


  bigint ntimestep;               // current timestep
  int nsteps;                     // # of steps to run
  int runflag;                    // 0 for unset, 1 for run
  bigint firststep,laststep;      // 1st & last step of this run
  bigint beginstep,endstep;       // 1st and last step of multiple runs
  int first_update;               // 0 before initial update, 1 after

  double time;                    // simulation time at time_last_update
  bigint time_last_update;        // last timestep that time was updated

  double dt;                      // timestep size

  char *unit_style;      // style of units used throughout simulation
  double boltz;          // Boltzmann constant (eng/degree K)
  double mvv2e;          // conversion of mv^2 to energy

  double echarge;        // charge of an electron
  double ev2kelvin;   // conversion of eV to Kelvin
  double proton_mass;   // mass of a proton
  double electron_mass; // mass of an electron
  double epsilon_0;   // vacuum permittivity
  double c;           // speed of light
  double hbar; // Planck constant
  double ANGSTROM;
  double joule2ev;       // conversion of joules to eV

  double fnum;           // ratio of real particles to simulation particles
  double nrho;           // number density of background gas
  double vstream[3];     // streaming velocity of background gas
  double temp_thermal;   // thermal temperature of background gas
  int optmove_flag;      // global optmove option set

  int fstyle;            // external field: NOFIELD, CFIELD, PFIELD, GFIELD, 
  int efstyle;            // external electric field: NOFIELD, PFIELD
  double field[3];       // constant external field
  char *fieldID;         // fix ID for PFIELD or GFIELD
  int ifieldfix;         // index of external field fix
  int *field_active;     // ptr to field_active flags in fix
  int fieldfreq;         // update GFIELD every this many timsteps

  
  char *efieldID;       // fix ID for PFIELD
  int efieldfix;         // index of external electric field fix
  int *efield_active;   // ptr to field_active flags in fix

  int bfstyle;            // external magnetic field: NOFIELD, PFIELD
  char *bfieldID;       // fix ID for PFIELD
  int bfieldfix;         // index of external magnetic field fix
  int *bfield_active;   // ptr to field_active flags in fix

  int ethermalflag;     // external electron thermal gradient field: NOFIELD, PFIELD
  int ethermalstyle;            // external electron thermal gradient field: NOFIELD, PFIELD
  char *ethermalID;     // fix ID for PFIELD
  int ethermalfix;       // index of external electron thermal gradient fix ethermalfix
  int *ethermal_active; // ptr to field_active flags in fix

  // for the ion now
  int ithermalflag;     // external ion thermal gradient field: NOFIELD, PFIELD
  int ithermalstyle;            // external ion thermal gradient field: NOFIELD, PF
  char *ithermalID;     // fix ID for PFIELD
  int ithermalfix;       // index of external ion thermal gradient fix ithermal
  int *ithermal_active; // ptr to field_active flags in fix

  int nmigrate;          // # of particles to migrate to new procs
  int *mlist;            // indices of particles to migrate

                         // current step counters
  int niterate;          // iterations of move/comm
  int ntouch_one;        // particle-cell touches
  int ncomm_one;         // particles migrating to new procs
  int nboundary_one;     // particles colliding with global boundary
  int nexit_one;         // particles exiting outflow boundary
  int nscheck_one;       // surface elements checked for collisions
  int nscollide_one;     // particle/surface collisions

  bigint first_running_step; // timestep running counts start on
  int niterate_running;      // running count of move/comm interations
  bigint nmove_running;      // running count of total particle moves
  bigint ntouch_running;     // running count of current step counters
  bigint ncomm_running;
  bigint nboundary_running;
  bigint nexit_running;
  bigint nscheck_running;
  bigint nscollide_running;

  int cross_field_diffusion_flag; // 1 if cross-field diffusion
  int cross_diffusion_flag;
  int background_collision_flag; // 1 if background collisions are enabled
  int nstuck;                // # of particles stuck on surfs and deleted
  int naxibad;               // # of particles where axisymm move was bad
                             // in this case, bad means particle ended up
                             // outside of final cell curved surf by epsilon
                             // when move logic thinks it is inside cell

  int reorder_period;        // # of timesteps between particle reordering
  int global_mem_limit;      // max # of bytes in arrays for rebalance and reordering
  int mem_limit_grid_flag;   // 1 if using size of grid as memory limit
  void set_mem_limit_grid(int gnlocal = 0);
  int have_mem_limit();      // 1 if have memory limit

  int copymode;          // 1 if copy of class (prevents deallocation of
                         //  base class when child copy is destroyed)

  class RanMars *ranmaster;   // master random number generator

  double rcblo[3],rcbhi[3];    // debug info from RCB for dump image

  // this info accessed by other classe to perform surface tallying
  // by SurfReactAdsorb for on-surface reactions
  // by FixEmitSurf for particles emitted from surfs

  int nsurf_tally;         // # of Cmp tallying surf bounce info this step
  int nboundary_tally;     // # of Cmp tallying boundary bounce info this step
  class Compute **slist_active;   // list of active surf Computes this step
  class Compute **blist_active;   // list of active boundary Computes this step

  // public methods

  Update(class SPARTA *);
  ~Update();
  void set_units(const char *);
  virtual void init();
  virtual void setup();
  virtual void run(int);
  void global(int, char **);
  void reset_timestep(int, char **);

  int split3d(int, double *);
  int split2d(int, double *);

      PlasmaData plasma_data;
  // PMI
  void read_plasma_state(int , char **);
  void read_magnetic_fields(int , char **);
  void broadcastMagneticData(MagneticFieldData& data);
  void initializeMagneticData();

  PlasmaData   readPlasmaData(const std::string& filePath);
  double bilinearInterpolation(double r, double z,
  const std::vector<double>& r_values,
  const std::vector<double>& z_values,
  const std::vector<std::vector<double>>& dens_e_grid);
  PlasmaDataParams bilinearInterpolationPlasma(int icell, const PlasmaData& data);
  std::unordered_map<int, PlasmaDataParams> plasmaDataCache;
  std::unordered_map<int, PlasmaDataParams> plasma_data_map;
  MagneticFieldDataParams bilinearInterpolationMagneticField(int icell, const MagneticFieldData& data);
  std::unordered_map<int, MagneticFieldDataParams> magneticFieldDataCache;
  std::unordered_map<int, MagneticFieldDataParams> magnetic_data_map;
  void initializePlasmaData();
  void broadcastPlasmaData(PlasmaData& data);
  void droplet_evaporation_model(double& droplet_mass, double& T, double Qs);
  void apply_bgk_collision_step(double dt,
                                double R, double Z, double phi,
                                double charge, double mass,
                                double &vx, double &vy, double &vz);  // NOTE: no const
  std::vector<int> materials;

     char *target_material;
double target_material_charge;
double target_material_mass;
double target_material_binding_energy;
void pusherBoris2D( int i, int icell, double dt, double *x, double *v, double *xnew, double charge, double mass);

double distance_to_surface_for_particle(int icell, const double x[3], const BoundaryInfo& info);
PlasmaDataParams interpolatePlasma_RZ_clamped(double R, double Z, const PlasmaData& data);
PlasmaDataParams interpolatePlasma_RZ_constant();
inline double signed_plane_distance(const BoundaryInfo& info, const double x[3]) const {
  // n is unit; plane n·r = plane_d
  // positive if particle is on "outside" side along normal
  const double s = info.normal[0]*x[0] + info.normal[1]*x[1] + info.normal[2]*x[2];
  return info.plane_d - s;
}

// ---- local geometry helpers (no dependencies on SPARTA internals) ----
static inline double clamp01(double t) { return (t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t)); }

static inline double dist_point_seg(const double p[3], const double a[3], const double b[3]) {
  double ab[3] = { b[0]-a[0], b[1]-a[1], b[2]-a[2] };
  double ap[3] = { p[0]-a[0], p[1]-a[1], p[2]-a[2] };
  double ab2 = ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2];
  if (ab2 == 0.0) {
    double dx = p[0]-a[0], dy = p[1]-a[1], dz = p[2]-a[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
  }
  double t = (ap[0]*ab[0] + ap[1]*ab[1] + ap[2]*ab[2]) / ab2;
  t = clamp01(t);
  double qx = a[0] + t*ab[0], qy = a[1] + t*ab[1], qz = a[2] + t*ab[2];
  double dx = p[0]-qx, dy = p[1]-qy, dz = p[2]-qz;
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

static inline void cross3(const double u[3], const double v[3], double w[3]) {
  w[0] = u[1]*v[2] - u[2]*v[1];
  w[1] = u[2]*v[0] - u[0]*v[2];
  w[2] = u[0]*v[1] - u[1]*v[0];
}
static inline double dot3(const double u[3], const double v[3]) {
  return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

static double dist_point_tri(const double p[3], const double a[3],
                             const double b[3], const double c[3]) {
  // Algorithm: project to plane, test barycentric; else min edge distance
  double ab[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
  double ac[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
  double n[3]; cross3(ab, ac, n);
  double nn = std::sqrt(dot3(n,n));
  if (nn == 0.0) {
    // degenerate: fall back to closest edge
    double d1 = dist_point_seg(p, a, b);
    double d2 = dist_point_seg(p, b, c);
    double d3 = dist_point_seg(p, c, a);
    return std::min(d1, std::min(d2, d3));
  }
  n[0]/=nn; n[1]/=nn; n[2]/=nn;

  // plane projection
  double ap[3] = {p[0]-a[0], p[1]-a[1], p[2]-a[2]};
  double dist_plane = dot3(ap, n);
  double proj[3] = { p[0]-dist_plane*n[0],
                     p[1]-dist_plane*n[1],
                     p[2]-dist_plane*n[2] };

  // barycentric coords on triangle (Möller–Trumbore-ish)
  double v0[3] = {ab[0], ab[1], ab[2]};
  double v1[3] = {ac[0], ac[1], ac[2]};
  double v2[3] = {proj[0]-a[0], proj[1]-a[1], proj[2]-a[2]};

  double d00 = dot3(v0,v0);
  double d01 = dot3(v0,v1);
  double d11 = dot3(v1,v1);
  double d20 = dot3(v2,v0);
  double d21 = dot3(v2,v1);
  double denom = d00*d11 - d01*d01;

  if (denom != 0.0) {
    double v = (d11*d20 - d01*d21) / denom;
    double w = (d00*d21 - d01*d20) / denom;
    double u = 1.0 - v - w;
    // inside (with small tolerance)
    if (u >= -1e-12 && v >= -1e-12 && w >= -1e-12) {
      return std::fabs(dist_plane);
    }
  }

  // outside → min distance to edges
  double d1 = dist_point_seg(p, a, b);
  double d2 = dist_point_seg(p, b, c);
  double d3 = dist_point_seg(p, c, a);
  return std::min(d1, std::min(d2, d3));
}

  int plasmaStyle; // UNIFORM, FILE

 protected:

 int sgroupbit;


  BoundaryInfo* boundary_info;   // Array to store boundary information for each cell
  double** efield_sheath;        // 2D array to store the sheath electric field for each cell

  double normal01_from_uniforms_();  // Box–Muller using ranmaster

   // PMI
   double temp_e;           // electron temperature in eV
   double temp_i;           // ion temperature in eV
   double dens_i;           // ion density in m^-3
   double dens_e;           // electron density in m^-3
   double flow_v[3];           // flow velocity in m/s
   double bfield[3];        // magnetic field in T
   double efield[3];        // electric field in V/m
   double grad_temp_i[3];   // ion temperature gradient in eV/m
   double grad_temp_e[3];   // electron temperature gradient in eV/m
   double grad_te_r;        // radial temperature gradient in eV/m
   double grad_te_z;        // axial temperature gradient in eV/m
   double grad_ti_r;        // radial temperature gradient in eV/m
   double grad_ti_z;        // axial temperature gradient in eV/m
   double grad_te_t;        // azimuthal temperature gradient in eV/m
   double grad_ti_t;        // azimuthal temperature gradient in eV/m
   double d_perp;  // perpendicular diffusion
   double d_flow_scale; // flow scale for background collisions
   char *plasma_state; // file containing plasma state
   MagneticFieldData magnetic_data;
   std::string plasmaStatePath;
   int recombination_flag; // 1 if recombination is enabled
   int ionization_flag; // 1 if ionization is enabled
   std::string adas_rates_path;
   std::string magneticFieldsPath;


   int sheath_field_flag; // 1 if sheath electric field is enabled
   char *plasma_background_material; // material for background plasma
   int thermal_gradient_forces_flag; 
   int plasma_background_mass;
   int plasma_background_charge;
 

  int magneticFieldsStyle; // 0 = constant, 1 = file
  int me,nprocs;
  int maxmigrate;            // max # of particles in mlist
  class RanKnuth *random;     // RNG for particle timestep moves

  int collide_react;         // 1 if any SurfCollide or React classes defined
  int nsc,nsr;               // copy of Collide/React data in Surf class
  class SurfCollide **sc;
  class SurfReact **sr;

  int bounce_tally;               // 1 if any bounces are ever tallied
  int nslist_compute;             // # of computes that tally surf bounces
  int nblist_compute;             // # of computes that tally boundary bounces
  class Compute **slist_compute;  // list of all surf bounce Computes
  class Compute **blist_compute;  // list of all boundary bounce Computes

  int surf_pre_tally;       // 1 to log particle stats before surf collide
  int boundary_pre_tally;   // 1 to log particle stats before boundary collide

  int collide_react_setup();
  void collide_react_reset();
  void collide_react_update();

  int bounce_setup();
  virtual void bounce_set(bigint);

  int nulist_surfcollide;
  SurfCollide **ulist_surfcollide;

  int dynamic;              // 1 if any classes do dynamic updates of params
  void dynamic_setup();
  void dynamic_update();

  void reset_timestep(bigint);

  //int axi_vertical_line(double, double *, double *, double, double, double,
  //                     double &);

  // remap x and v components into axisymmetric plane
  // input x at end of linear move (x = xold + dt*v)
  // change x[1] = sqrt(x[1]^2 + x[2]^2), x[2] = 0.0
  // change vy,vz by rotation into axisymmetric plane
  inline void axi_remap(double *x, double *v) {
    double ynew = x[1];
    double znew = x[2];
    x[1] = sqrt(ynew*ynew + znew*znew);
    x[2] = 0.0;
    double rn = ynew / x[1];
    double wn = znew / x[1];
    double vy = v[1];
    double vz = v[2];
    v[1] = vy*rn + vz*wn;
    v[2] = -vy*wn + vz*rn;
  };

  typedef void (Update::*FnPtr)();
  FnPtr moveptr;             // ptr to move method
  template < int, int, int > void move();

  int perturbflag;
  int eperturbflag;
  int bperturbflag;

  typedef void (Update::*FnPtr2)(int, int, double, double *, double *);
  FnPtr2 moveperturb;        // ptr to moveperturb method

  // variants of moveperturb method
  // adjust end-of-move x,v due to perturbation on straight-line advection

  inline void field2d(int i, int icell, double dt, double *x, double *v) {
    double dtsq = 0.5*dt*dt;
    x[0] += dtsq*field[0];
    x[1] += dtsq*field[1];
    v[0] += dt*field[0];
    v[1] += dt*field[1];
  };

  inline void field3d(int i, int icell, double dt, double *x, double *v) {
    double dtsq = 0.5*dt*dt;
    x[0] += dtsq*field[0];
    x[1] += dtsq*field[1];
    x[2] += dtsq*field[2];
    v[0] += dt*field[0];
    v[1] += dt*field[1];
    v[2] += dt*field[2];
  };

  // NOTE: cannot be inline b/c ref to modify->fix[] is not supported
  //       unless possibly include modify.h and fix.h in this file
  void field_per_particle(int, int, double, double *, double *);
  void field_per_grid(int, int, double, double *, double *);

  void initializeIonizationRates();
  MagneticFieldData readMagneticFieldData(const std::string& filePath);
  void pusher_boris3D(int i, int icell, double dt, double *x, double *v, double *xnew, double mass, double charge);
  void pusher_guiding_center_2D(int i, int icell, double dt, double *x, double *v, double mass, double charge);
  // void compute_dist_grid(int icell, double& mindistance, double*& norm, int& surf_id, int& type_group, int& mask_group, double surf_center[3]);
//   void compute_dist_grid(
//     int icell,
//     double& mindist,
//     double out_normal[3],
//     int& surf_id,
//     int& type_group,
//     int& mask_group,
//     double out_center[3]
// );

// void compute_dist_grid(int icell,
//                                double& mindist,
//                                double out_normal[3],
//                                int&    surf_id,          // external file ID
//                                int&    surf_local,       // local array index
//                                int&    type_group,
//                                int&    mask_group,
//                                double  out_center[3]);

  // double calculate_angle(const double* Bfield, const double* normal);
  
  static inline double box_muller_normal(double u1, double u2)
{
  // u1 in (0,1], u2 in [0,1)
  const double two_pi = 6.283185307179586;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
}

// B_phi(R) = B0 * R0 / R  ;  Br = 0, Bz = 0
inline void axisymm_toroidal_field_cart(double R, double phi,
                                        double R0, double B0,
                                        double &Bx, double &By, double &Bz)
{
    const double Bphi = (R > 1e-12) ? B0 * R0 / R : 0.0;
    const double c = std::cos(phi), s = std::sin(phi);
    // e_phi = (-sin phi, cos phi, 0)
    Bx = -Bphi * s;
    By =  Bphi * c;
    Bz =  0.0;
}


// Analytic B-field from the paper (Eq. 24), scaled by B0
static inline void Bfield_qin24_cart(
    double X, double Y, double Z,
    double R0, double B0,
    double &Bx, double &By, double &Bz)
{
  const double eps = 1e-14;
  const double R   = std::sqrt(X*X + Y*Y + Z*Z);
  const double R2  = std::max(R*R, eps);

  const double Bx_hat = -(2.0*R0*Y + X*Z) / (2.0*R2);
  const double By_hat =  (2.0*R0*X - Y*Z) / (2.0*R2);
  const double Bz_hat =  (R - 1.0)       / (2.0*std::max(R, eps));

  Bx = B0 * Bx_hat;
  By = B0 * By_hat;
  Bz = B0 * Bz_hat;
}
static inline void cart_to_cyl_B(double phi,
                                 double Bx, double By, double Bz,
                                 double &Br, double &Bphi, double &Bz_c)
{
  const double c = std::cos(phi), s = std::sin(phi);
  Br   =  Bx*c + By*s;
  Bphi = -Bx*s + By*c;
  Bz_c =  Bz;
}

// // Box–Muller using your RNG (expects rng->uniform() in (0,1))
// inline static double normal01(Update::RNG *rng) {
//   const double u1 = std::max(1e-12, rng->uniform());
//   const double u2 = std::max(1e-12, rng->uniform());
//   return std::sqrt(-2.0*std::log(u1)) * std::cos(2.0*M_PI*u2);
// }


  #ifdef SPARTA_MAP
  typedef std::map<bigint, BoundaryInfo> boundaryInfoHash;
#else
  typedef std::unordered_map<bigint, BoundaryInfo> boundaryInfoHash;
#endif
boundaryInfoHash hash2;
  void getBoundaryInfo(int icell, double *Bfield, BoundaryInfo& info);

  void sheathEfieldBrooks(double *B, double temp_e, double dens_e, double temp_i, double t_sheath, double alpha, double minDistance, const double* normal_with_min_distance, double& Emag, double& densitySheath);
  void apply_cross_field_diffusion(int icell, double dt, double* B, double* x);
  void thermal_gradient_Efield(const double mass, const double charge,
                                    const double gradTe_e[3],  // ∇T_e in eV/m (R,φ,Z)
                                    const double gradTe_i[3],  // ∇T_i in eV/m (R,φ,Z)
                                    double E_th[3]);            // out: (Er,Eφ,Ez) in V/
  void apply_cross_field_diffusion_cart( const double dt,  const double Bx, const double By, const double Bz,
  const double u01, double &Xn, double &Yn, double &Zn);
};

}

#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Gravity in z not allowed for 2d

Self-explanatory.

E: Gravity in y not allowed for axi-symmetric model

Self-explanatory.

E: Particle %d on proc %d hit inside of surf %d on step %ld

This error should not happen if particles start outside of physical
objects.  Please report the issue to the SPARTA developers.

E: Sending particle to self

This error should not occur.  Please report the issue to the SPARTA
developers.

E: Cannot set global surfmax when surfaces already exist

This setting must be made before any surfac elements are
read via the read_surf command.

E: Global mem/limit setting cannot exceed 2GB

Self-expanatory, prevents 32-bit interger overflow

E: Timestep must be >= 0

Reset_timestep cannot be used to set a negative timestep.

E: Too big a timestep

Reset_timestep timestep value must fit in a SPARTA big integer, as
specified by the -DSPARTA_SMALL, -DSPARTA_BIG, or -DSPARTA_BIGBIG
options in the low-level Makefile used to build SPARTA.  See
Section 2.2 of the manual for details.

E: Cannot reset timestep with a time-dependent fix defined

The timestep cannot be reset when a fix that keeps track of elapsed
time is in place.

*/
