/* ----------------------------------------------------------------------
   OpenEdge extension of SPARTA UpdateKokkos.
   Adds Boris/GCA pusher support with separate E-field and B-field fixes.
   Replaces src/KOKKOS/update_kokkos.h when PKG_OPENEDGE is ON.
------------------------------------------------------------------------- */

#ifndef SPARTA_UPDATE_KOKKOS_H
#define SPARTA_UPDATE_KOKKOS_H

#include "update.h"
#include "kokkos_type.h"
#include "particle.h"
#include "grid_kokkos.h"
#include "domain_kokkos.h"
#include "kokkos_copy.h"
#include "surf_collide_diffuse_kokkos.h"
#include "surf_collide_specular_kokkos.h"
#include "surf_collide_vanish_kokkos.h"
#include "surf_collide_piston_kokkos.h"
#include "surf_collide_transparent_kokkos.h"
#include "compute_boundary_kokkos.h"
#include "compute_surf_kokkos.h"
#include "boris_grid_kokkos.h"

namespace SPARTA_NS {

#define KOKKOS_MAX_SURF_COLL_PER_TYPE 2
#define KOKKOS_MAX_TOT_SURF_COLL 10
#define KOKKOS_MAX_BLIST 2
#define KOKKOS_MAX_SLIST 2

struct s_UPDATE_REDUCE {
  int ntouch_one,nexit_one,nboundary_one,
      entryexit,ncomm_one,
      nscheck_one,nscollide_one,nreact_one,nstuck,
      naxibad,error_flag;
  KOKKOS_INLINE_FUNCTION
  s_UPDATE_REDUCE() {
    ntouch_one = nexit_one = nboundary_one = ncomm_one = 0;
    nscheck_one = nscollide_one = nreact_one = nstuck = naxibad = 0;
  }
  KOKKOS_INLINE_FUNCTION
  void operator+=(const s_UPDATE_REDUCE &rhs) {
    ntouch_one += rhs.ntouch_one; nexit_one += rhs.nexit_one;
    nboundary_one += rhs.nboundary_one; ncomm_one += rhs.ncomm_one;
    nscheck_one += rhs.nscheck_one; nscollide_one += rhs.nscollide_one;
    nreact_one += rhs.nreact_one; nstuck += rhs.nstuck; naxibad += rhs.naxibad;
  }
};
typedef struct s_UPDATE_REDUCE UPDATE_REDUCE;

template<int DIM, int SURF, int REACT, int OPT, int ATOMIC_REDUCTION>
struct TagUpdateMove{};

class UpdateKokkos : public Update {
 public:
  typedef UPDATE_REDUCE value_type;

  DAT::tdual_int_1d k_mlist;
  DAT::tdual_int_1d k_mlist_small;

  UpdateKokkos(class SPARTA *);
  ~UpdateKokkos();
  void init();
  void setup();
  void run(int);

  template<int DIM, int SURF, int REACT, int OPT, int ATOMIC_REDUCTION>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagUpdateMove<DIM,SURF,REACT,OPT,ATOMIC_REDUCTION>, const int&) const;

  template<int DIM, int SURF, int REACT, int OPT, int ATOMIC_REDUCTION>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagUpdateMove<DIM,SURF,REACT,OPT,ATOMIC_REDUCTION>, const int&, UPDATE_REDUCE&) const;

 private:

  double dt;
  int field_active[3];

  double dx,dy,dz,Lx,Ly,Lz;
  double xlo,ylo,zlo,xhi,yhi,zhi;
  int ncx,ncy,ncz;
  GridKokkos::hash_type hash_kk;

  t_cell_1d d_cells;
  t_sinfo_1d d_sinfo;
  t_pcell_1d d_pcells;

  Kokkos::Crs<int, DeviceType, void, int> d_csurfs;
  Kokkos::Crs<int, DeviceType, void, int> d_csplits;
  Kokkos::Crs<int, DeviceType, void, int> d_csubs;

  t_line_1d d_lines;
  t_tri_1d d_tris;
  t_particle_1d d_particles;

  // Base SPARTA field perturbation (fstyle)
  DAT::t_float_2d_lr d_fieldfix_array_particle;
  DAT::t_float_2d_lr d_fieldfix_array_grid;
  class KokkosBase* KKBaseFieldFix;

  // OpenEdge: separate B-field and E-field fix device views
  DAT::t_float_2d_lr d_bfieldfix_array_grid;
  DAT::t_float_2d_lr d_efieldfix_array_grid;
  class KokkosBase* KKBaseBFieldFix;
  class KokkosBase* KKBaseEFieldFix;

  // OpenEdge: Boris config (copied from Update base at init)
  int oe_boris_subcycles;
  double oe_echarge;

  KKCopy<GridKokkos> grid_kk_copy;
  KKCopy<DomainKokkos> domain_kk_copy;

  int sc_type_list[KOKKOS_MAX_TOT_SURF_COLL];
  int sc_map[KOKKOS_MAX_TOT_SURF_COLL];
  KKCopy<SurfCollideSpecularKokkos> sc_kk_specular_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollideDiffuseKokkos> sc_kk_diffuse_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollideVanishKokkos> sc_kk_vanish_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollidePistonKokkos> sc_kk_piston_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<SurfCollideTransparentKokkos> sc_kk_transparent_copy[KOKKOS_MAX_SURF_COLL_PER_TYPE];
  KKCopy<ComputeBoundaryKokkos> blist_active_copy[KOKKOS_MAX_BLIST];
  KKCopy<ComputeSurfKokkos> slist_active_copy[KOKKOS_MAX_SLIST];

  typedef Kokkos::DualView<int[14], DeviceType::array_layout, DeviceType> tdual_int_14;
  typedef tdual_int_14::t_dev t_int_14;
  typedef tdual_int_14::t_host t_host_int_14;
  t_int_14 d_scalars;
  t_host_int_14 h_scalars;

  DAT::t_int_scalar d_ntouch_one;     HAT::t_int_scalar h_ntouch_one;
  DAT::t_int_scalar d_nexit_one;      HAT::t_int_scalar h_nexit_one;
  DAT::t_int_scalar d_nboundary_one;  HAT::t_int_scalar h_nboundary_one;
  DAT::t_int_scalar d_nmigrate;       HAT::t_int_scalar h_nmigrate;
  DAT::t_int_scalar d_entryexit;      HAT::t_int_scalar h_entryexit;
  DAT::t_int_scalar d_ncomm_one;      HAT::t_int_scalar h_ncomm_one;
  DAT::t_int_scalar d_nscheck_one;    HAT::t_int_scalar h_nscheck_one;
  DAT::t_int_scalar d_nscollide_one;  HAT::t_int_scalar h_nscollide_one;
  DAT::t_int_scalar d_nreact_one;     HAT::t_int_scalar h_nreact_one;
  DAT::t_int_scalar d_nstuck;         HAT::t_int_scalar h_nstuck;
  DAT::t_int_scalar d_naxibad;        HAT::t_int_scalar h_naxibad;
  DAT::t_int_scalar d_error_flag;     HAT::t_int_scalar h_error_flag;
  DAT::t_int_scalar d_retry;          HAT::t_int_scalar h_retry;
  DAT::t_int_scalar d_nlocal;         HAT::t_int_scalar h_nlocal;

  void backup();
  void restore();
  t_particle_1d d_particles_backup;
  void bounce_set(bigint);

  KOKKOS_INLINE_FUNCTION
  void axi_remap(double *x, double *v) const {
    double ynew = x[1], znew = x[2];
    x[1] = sqrt(ynew*ynew + znew*znew); x[2] = 0.0;
    double rn = ynew / x[1], wn = znew / x[1];
    double vy = v[1], vz = v[2];
    v[1] = vy*rn + vz*wn; v[2] = -vy*wn + vz*rn;
  };

  typedef void (UpdateKokkos::*FnPtr)();
  FnPtr moveptr;
  template <int, int, int, int> void move();

  KOKKOS_INLINE_FUNCTION
  void field2d(double dt, double *x, double *v) const {
    const double dtsq = 0.5*dt*dt;
    x[0] += dtsq*field[0]; x[1] += dtsq*field[1];
    v[0] += dt*field[0];   v[1] += dt*field[1];
  };
  KOKKOS_INLINE_FUNCTION
  void field3d(double dt, double *x, double *v) const {
    const double dtsq = 0.5*dt*dt;
    x[0] += dtsq*field[0]; x[1] += dtsq*field[1]; x[2] += dtsq*field[2];
    v[0] += dt*field[0];   v[1] += dt*field[1];   v[2] += dt*field[2];
  };
  KOKKOS_INLINE_FUNCTION
  void field_per_particle(int i, int icell, double dt, double *x, double *v) const {
    const double dtsq = 0.5*dt*dt;
    auto &d_array = d_fieldfix_array_particle;
    int icol = 0;
    if (field_active[0]) { x[0] += dtsq*d_array(i,icol); v[0] += dt*d_array(i,icol); icol++; }
    if (field_active[1]) { x[1] += dtsq*d_array(i,icol); v[1] += dt*d_array(i,icol); icol++; }
    if (field_active[2]) { x[2] += dtsq*d_array(i,icol); v[2] += dt*d_array(i,icol); icol++; }
  };
  KOKKOS_INLINE_FUNCTION
  void field_per_grid(int i, int icell, double dt, double *x, double *v) const {
    const double dtsq = 0.5*dt*dt;
    auto &d_array = d_fieldfix_array_grid;
    int icol = 0;
    if (field_active[0]) { x[0] += dtsq*d_array(icell,icol); v[0] += dt*d_array(icell,icol); icol++; }
    if (field_active[1]) { x[1] += dtsq*d_array(icell,icol); v[1] += dt*d_array(icell,icol); icol++; }
    if (field_active[2]) { x[2] += dtsq*d_array(icell,icol); v[2] += dt*d_array(icell,icol); icol++; }
  };

  // OpenEdge: device-callable Boris 3D pusher (reads E/B from grid fix views)
  KOKKOS_INLINE_FUNCTION
  void oe_boris3d(int i, int icell, double dt_full,
                  double *x, double *v, double *xnew,
                  double charge, double mass) const;

  KOKKOS_INLINE_FUNCTION
  int split3d(int, double*) const;
  KOKKOS_INLINE_FUNCTION
  int split2d(int, double*) const;
};

}

#endif
