/* ----------------------------------------------------------------------
    OpenEdge: Kokkos/GPU port of the plasma-wall interaction (PWI)
    surface reaction model, narrow first pass.

    Supported (the WEST pure-W monoblock production path):
      T <trim_table>   TRIM reflection (pure 2D R_N + quantile sampling)
      A <R>            absorb/retain, optional simple-return re-emission
                       (reactant == product only)
      S <entry>        additive self-sputtering, 2D (E,theta) yield table
                       or analytic Eckstein, Thompson energy, cosine angle
    plus physical particle weights (pweight), areal-density (adens)
    deposition/erosion deltas, impact-energy histograms, and Kokkos
    react/retry growth semantics.

    Device accumulates compact per-step deltas; the host CPU class stays
    authoritative for sync_sigma(), strata, MPI reductions, and histogram
    output. Unsupported general PWI modes (compound targets, T- or
    composition-dependent tables, per-surf twall/R, molecular channels)
    fail explicitly at init - no silent physics approximation.

    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2026)
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef SURF_REACT_CLASS

SurfReactStyle(surface/pwi/kk,SurfReactSurfacePWIKokkos)

#else

#ifndef SPARTA_SURF_REACT_SURFACE_PWI_KOKKOS_H
#define SPARTA_SURF_REACT_SURFACE_PWI_KOKKOS_H

#include "kokkos_type.h"
#include "surf_react_surface_pwi.h"
#include "particle_kokkos.h"
#include "rand_pool_wrap.h"
#include "Kokkos_Random.hpp"
#include "eckstein_sputter.h"

namespace SPARTA_NS {

class SurfReactSurfacePWIKokkos : public SurfReactSurfacePWI {
 public:
  // reaction channel types, must match the enum in surf_react_surface_pwi.cpp
  enum{PWI_DISSOCIATION,PWI_EXCHANGE,PWI_RECOMBINATION,
       PWI_TRIM_REFLECT,PWI_ABSORB_REEMIT,PWI_SPUTTER};
  enum{PWI_PKEEP,PWI_PINSERT,PWI_PDONE,PWI_PDISCARD,
       PWI_PENTRY,PWI_PEXIT,PWI_PSURF};    // matches update.cpp

  static constexpr int PWI_NANG = 18;      // = base EHIST_NANG (5-deg bins)
  static constexpr double PWI_2PI = 6.283185307179586;
  static constexpr double PWI_EV2J = 1.602176634e-19;
  static constexpr double PWI_KB = 1.380649e-23;

  SurfReactSurfacePWIKokkos(class SPARTA *, int, char **);
  SurfReactSurfacePWIKokkos(class SPARTA *);   // KKCopy shallow copy
  ~SurfReactSurfacePWIKokkos();
  void init() override;
  void tally_reset() override;
  void tally_update() override;

  void pre_react();
  void post_react();
  void backup();
  void restore();

 private:
  void init_device_tables();
  void check_supported();
  void fold_sigma();
  void fold_ehist();

  // ---- device-resident reaction data (flattened in init) ----

  DAT::t_int_1d d_reactions_n;      // [nspecies] # reactions per species
  DAT::t_int_2d d_list;             // [nspecies][nmax] -> reaction index

  DAT::t_int_1d d_type;             // [nlist] channel type
  DAT::t_int_1d d_prod;             // [nlist] product species (or -1)
  DAT::t_int_1d d_trim;             // [nlist] TRIM table index or -1
  DAT::t_int_1d d_sput;             // [nlist] sputter table index or -1
  DAT::t_float_1d d_prob;           // [nlist] fixed probability
  DAT::t_float_1d d_Rrec;           // [nlist] A-channel recycling coeff R
  DAT::t_float_2d_lr d_spp;         // [nlist][4] Eckstein Es,Eth,Q,ETF

  // TRIM reflection tables, fixed EIRENE-schema sizes (NE=12,NTHETA=7,NQ=5)
  DAT::t_float_2d_lr d_tr_E, d_tr_th, d_tr_raar;
  DAT::t_float_2d_lr d_tr_RN, d_tr_Eq, d_tr_Emin, d_tr_Emax;
  DAT::t_float_2d_lr d_tr_cp, d_tr_ca;

  // 2D sputter-yield tables, padded to max dims
  DAT::t_int_1d d_su_NE, d_su_NT;
  DAT::t_float_2d_lr d_su_E, d_su_th, d_su_Y;

  // ---- per-step device tallies and deltas ----

  DAT::t_int_1d d_scalars;          // [0]=nsingle, [1..nlist]=tally_single
  HAT::t_int_1d h_scalars;
  DAT::t_int_scalar d_nsingle;
  DAT::t_int_1d d_tally_single;

  DAT::t_float_1d d_sigma_delta;    // [nsurf*ncols] like host sigma_delta
  DAT::t_float_1d d_dep_delta;      // [nsurf]
  DAT::t_float_1d d_ehist_delta;    // [2*nbin + 2*NANG + nsp*nbin]
  HAT::t_float_1d h_sigma_delta, h_dep_delta, h_ehist_delta;

  DAT::t_float_1d d_area;           // [nlocal+nghost] per-surf area
  DAT::t_int_1d d_gid0;             // [nlocal+nghost] global surf ID - 1

  // react/retry rollback snapshots
  DAT::t_int_1d d_scalars_bak;
  DAT::t_float_1d d_sigma_bak, d_dep_bak, d_ehist_bak;

  // ---- particle access ----

  t_particle_1d d_particles;
  t_species_1d d_species;
  ParticleKokkos::DeviceCustom custom_;
  int pw_slot;                      // edvec slot of pweight, -1 if absent

  // ---- captured scalars ----

  int sigma_on, ehist_on;
  int ncols;                        // = sigma_ncols = nspecies
  int nbin;                         // ehist_nbin
  int nsp;                          // ehist_nsp
  double emax;                      // ehist_emax
  double fnum_c, evconv, twall_c, rough_c;

#ifndef SPARTA_KOKKOS_EXACT
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;
#else
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
#endif

  RanKnuth* random_backup;

  // ---- device helpers ----

  KOKKOS_INLINE_FUNCTION
  Reflection::View trim_view(int it) const
  {
    Reflection::View tv;
    tv.E           = &d_tr_E(it,0);
    tv.theta_deg   = &d_tr_th(it,0);
    tv.raar        = &d_tr_raar(it,0);
    tv.R_N         = &d_tr_RN(it,0);
    tv.Eout_q      = &d_tr_Eq(it,0);
    tv.Eout_min    = &d_tr_Emin(it,0);
    tv.Eout_max    = &d_tr_Emax(it,0);
    tv.cos_polar_q = &d_tr_cp(it,0);
    tv.cos_azim_q  = &d_tr_ca(it,0);
    return tv;
  }

  // 2D sputter-yield lookup: log-E/linear-theta bilinear, clamped to the
  // grid, zero below the lowest tabulated energy. Same math as
  // ProcessLibrary::TrimSputterTable::slice_yield().

  KOKKOS_INLINE_FUNCTION
  double sput_yield(int it, double E_eV, double theta_deg) const
  {
    const int NE = d_su_NE(it);
    const int NT = d_su_NT(it);
    if (NE < 2 || NT < 2) return 0.0;
    if (E_eV < d_su_E(it,0)) return 0.0;
    const double x = (E_eV < d_su_E(it,NE-1)) ? E_eV : d_su_E(it,NE-1);
    const double le = log(x);
    double a = theta_deg;
    if (a < d_su_th(it,0)) a = d_su_th(it,0);
    if (a > d_su_th(it,NT-1)) a = d_su_th(it,NT-1);

    // lower_bound on E then theta
    int lo = 0, hi = NE;
    while (lo < hi) { int mid = (lo+hi)/2; if (d_su_E(it,mid) < x) lo = mid+1; else hi = mid; }
    int ie = lo;
    if (ie <= 0) ie = 1;
    if (ie >= NE) ie = NE - 1;
    lo = 0; hi = NT;
    while (lo < hi) { int mid = (lo+hi)/2; if (d_su_th(it,mid) < a) lo = mid+1; else hi = mid; }
    int ia = lo;
    if (ia <= 0) ia = 1;
    if (ia >= NT) ia = NT - 1;

    const double le1 = log(d_su_E(it,ie-1)), le2 = log(d_su_E(it,ie));
    const double a1 = d_su_th(it,ia-1), a2 = d_su_th(it,ia);
    const double te = (le2 != le1) ? (le - le1) / (le2 - le1) : 0.0;
    const double ta = (a2 != a1) ? (a - a1) / (a2 - a1) : 0.0;
    const double y00 = d_su_Y(it,(ie-1)*NT + (ia-1));
    const double y10 = d_su_Y(it,ie*NT + (ia-1));
    const double y01 = d_su_Y(it,(ie-1)*NT + ia);
    const double y11 = d_su_Y(it,ie*NT + ia);
    double y = (1.0-te)*(1.0-ta)*y00 + te*(1.0-ta)*y10
             + (1.0-te)*ta*y01 + te*ta*y11;
    return (Kokkos::isfinite(y) && y > 0.0) ? y : 0.0;
  }

  // Thompson sputtered-atom energy with recoil cutoff; same proposal/
  // rejection as the CPU pwi_sample_thompson()

  KOKKOS_INLINE_FUNCTION
  double sample_thompson(double Ub_eV, double Emax_eV, rand_type &rand_gen) const
  {
    const double denom = Emax_eV + Ub_eV;
    const double rmax = Emax_eV / denom;
    const double vmax = rmax * rmax;
    for (int it = 0; it < 64; it++) {
      double s = sqrt(rand_gen.drand() * vmax);
      double E = Ub_eV * s / (1.0 - s);
      if (rand_gen.drand() < 1.0 - sqrt((E + Ub_eV) / denom)) return E;
    }
    return 0.5 * Emax_eV;
  }

  KOKKOS_INLINE_FUNCTION
  void cosine_velocity(double *v, const double *norm, double energy_eV,
                       double mass, rand_type &rand_gen) const
  {
    double speed = 0.0;
    if (mass > 0.0 && energy_eV > 0.0)
      speed = sqrt(2.0 * energy_eV * PWI_EV2J / mass);

    double xi1 = rand_gen.drand();
    double xi2 = rand_gen.drand();
    double cosTheta = sqrt(xi1);
    double s2 = 1.0 - cosTheta*cosTheta;
    double sinTheta = sqrt(s2 > 0.0 ? s2 : 0.0);
    double phi = PWI_2PI * xi2;
    double cosPhi = cos(phi);
    double sinPhi = sin(phi);

    // basis: norm, tangent1, tangent2 (same construction as CPU)
    double tmp[3] = {0.0, 0.0, 1.0};
    if (fabs(norm[2]) > 0.9) { tmp[0] = 1.0; tmp[2] = 0.0; }
    double t1[3];
    t1[0] = norm[1]*tmp[2] - norm[2]*tmp[1];
    t1[1] = norm[2]*tmp[0] - norm[0]*tmp[2];
    t1[2] = norm[0]*tmp[1] - norm[1]*tmp[0];
    double t1len = sqrt(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2]);
    t1[0] /= t1len; t1[1] /= t1len; t1[2] /= t1len;
    double t2[3];
    t2[0] = norm[1]*t1[2] - norm[2]*t1[1];
    t2[1] = norm[2]*t1[0] - norm[0]*t1[2];
    t2[2] = norm[0]*t1[1] - norm[1]*t1[0];

    v[0] = speed * (sinTheta*cosPhi*t1[0] + sinTheta*sinPhi*t2[0] + cosTheta*norm[0]);
    v[1] = speed * (sinTheta*cosPhi*t1[1] + sinTheta*sinPhi*t2[1] + cosTheta*norm[1]);
    v[2] = speed * (sinTheta*cosPhi*t1[2] + sinTheta*sinPhi*t2[2] + cosTheta*norm[2]);
  }

  KOKKOS_INLINE_FUNCTION
  void thermal_flux_velocity(double *v, const double *norm, double T_K,
                             double mass, rand_type &rand_gen) const
  {
    if (mass <= 0.0 || T_K <= 0.0) { v[0] = v[1] = v[2] = 0.0; return; }

    double vrm = sqrt(2.0 * PWI_KB * T_K / mass);
    double vtan_scale = vrm / sqrt(2.0);

    double u1 = rand_gen.drand();
    if (u1 < 1e-300) u1 = 1e-300;
    double vn = vrm * sqrt(-log(u1));

    double u2 = rand_gen.drand();
    double u3 = rand_gen.drand();
    if (u2 < 1e-300) u2 = 1e-300;
    double r  = vtan_scale * sqrt(-2.0 * log(u2));
    double ph = PWI_2PI * u3;
    double vt1 = r * cos(ph);
    double vt2 = r * sin(ph);

    double nlen = sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
    double nh[3] = {norm[0]/nlen, norm[1]/nlen, norm[2]/nlen};
    double arb[3] = {1.0, 0.0, 0.0};
    if (fabs(nh[0]) > 0.9) { arb[0] = 0.0; arb[1] = 1.0; }
    double dot = arb[0]*nh[0] + arb[1]*nh[1] + arb[2]*nh[2];
    double t1[3] = {arb[0]-dot*nh[0], arb[1]-dot*nh[1], arb[2]-dot*nh[2]};
    double t1len = sqrt(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2]);
    t1[0] /= t1len; t1[1] /= t1len; t1[2] /= t1len;
    double t2[3] = {nh[1]*t1[2] - nh[2]*t1[1],
                    nh[2]*t1[0] - nh[0]*t1[2],
                    nh[0]*t1[1] - nh[1]*t1[0]};

    v[0] = vn*nh[0] + vt1*t1[0] + vt2*t2[0];
    v[1] = vn*nh[1] + vt1*t1[1] + vt2*t2[1];
    v[2] = vn*nh[2] + vt1*t1[2] + vt2*t2[2];
  }

  KOKKOS_INLINE_FUNCTION
  void reflected_velocity(double *v_out, const double *v_in,
                          const double *norm, double E_out_eV,
                          double cos_polar, double cos_azim,
                          double mass_out, rand_type &rand_gen) const
  {
    double E_out_J = E_out_eV / evconv;   // evconv = joule2ev*mvv2e
    double vmag = (mass_out > 0.0 && E_out_J > 0.0)
                ? sqrt(2.0 * E_out_J / mass_out)
                : 0.0;

    double nlen = sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
    double nh[3] = {norm[0]/nlen, norm[1]/nlen, norm[2]/nlen};
    double vn = v_in[0]*nh[0] + v_in[1]*nh[1] + v_in[2]*nh[2];
    double tang[3] = {v_in[0]-vn*nh[0], v_in[1]-vn*nh[1], v_in[2]-vn*nh[2]};
    double tlen = sqrt(tang[0]*tang[0] + tang[1]*tang[1] + tang[2]*tang[2]);
    if (tlen < 1e-12) {
      double arb[3] = {1.0, 0.0, 0.0};
      if (fabs(nh[0]) > 0.9) { arb[0] = 0.0; arb[1] = 1.0; }
      double dot = arb[0]*nh[0] + arb[1]*nh[1] + arb[2]*nh[2];
      tang[0] = arb[0]-dot*nh[0];
      tang[1] = arb[1]-dot*nh[1];
      tang[2] = arb[2]-dot*nh[2];
      tlen = sqrt(tang[0]*tang[0] + tang[1]*tang[1] + tang[2]*tang[2]);
    }
    double tin[3] = {-tang[0]/tlen, -tang[1]/tlen, -tang[2]/tlen};
    double bh[3] = {nh[1]*tin[2] - nh[2]*tin[1],
                    nh[2]*tin[0] - nh[0]*tin[2],
                    nh[0]*tin[1] - nh[1]*tin[0]};

    double sp2 = 1.0 - cos_polar*cos_polar;
    double sin_polar = sqrt(sp2 > 0.0 ? sp2 : 0.0);
    double sa2 = 1.0 - cos_azim*cos_azim;
    double sin_azim = sqrt(sa2 > 0.0 ? sa2 : 0.0);
    if (rand_gen.drand() < 0.5) sin_azim = -sin_azim;

    for (int d = 0; d < 3; d++)
      v_out[d] = vmag * (cos_polar * nh[d]
                       + sin_polar * (cos_azim * tin[d] + sin_azim * bh[d]));
  }

  // areal-density ledger: mirror of the CPU sigma_accumulate(); always
  // atomic (many surface impacts can hit one element concurrently)

  KOKKOS_INLINE_FUNCTION
  void sigma_acc(int isurf, int isp, double datoms) const
  {
    double area = d_area(isurf);
    if (area <= 0.0) return;
    int g = d_gid0(isurf);
    Kokkos::atomic_add(&d_sigma_delta(g*ncols + isp), datoms/area);
    if (datoms > 0.0) Kokkos::atomic_add(&d_dep_delta(g), datoms/area);
  }

  // impact histogram: layout [all_E | sput_E | all_A | sput_A | perspecies_E]

  KOKKOS_INLINE_FUNCTION
  void ehist_acc(double E_eV, double theta_deg, double pw, int sput,
                 int isp) const
  {
    int ie = (int) (E_eV / emax * nbin);
    if (ie < 0) ie = 0;
    if (ie >= nbin) ie = nbin - 1;
    int ia = (int) (theta_deg / 5.0);
    if (ia < 0) ia = 0;
    if (ia >= PWI_NANG) ia = PWI_NANG - 1;
    Kokkos::atomic_add(&d_ehist_delta(ie), pw);
    Kokkos::atomic_add(&d_ehist_delta(2*nbin + ia), pw);
    if (sput) {
      Kokkos::atomic_add(&d_ehist_delta(nbin + ie), pw);
      Kokkos::atomic_add(&d_ehist_delta(2*nbin + PWI_NANG + ia), pw);
    }
    if (nsp > 0 && isp >= 0 && isp < nsp)
      Kokkos::atomic_add(&d_ehist_delta(2*nbin + 2*PWI_NANG + isp*nbin + ie), pw);
  }

 public:

  /* ----------------------------------------------------------------------
     PWI surface reaction for particle IP on local surf ISURF.
     Order matches the CPU react(): incident (E,theta) -> additive
     sputter emission -> impact histogram -> reflect/absorb lottery.
     Emitted sputter products are complete on return: base record, PSURF
     flag with source-surface exclusion, remaining dtremain, cell weight,
     zeroed customs, inherited pweight. Returns reaction index+1 or 0.
     Sets d_retry and returns 0 if particle storage must grow.
  ------------------------------------------------------------------------- */

  template<int ATOMIC_REDUCTION>
  KOKKOS_INLINE_FUNCTION
  int react_kokkos(Particle::OnePart *&ip, double dtremain, int isurf,
                   const double *norm, Particle::OnePart *&jp, int &velreset,
                   const DAT::t_int_scalar &d_retry,
                   const DAT::t_int_scalar &d_nlocal) const
  {
    int n = d_reactions_n(ip->ispecies);
    if (n == 0) return 0;

    rand_type rand_gen = rand_pool.get_state();

    // incident impact energy [eV] and polar angle [deg from normal];
    // v already includes the inbound sheath boost applied by the mover

    const double mass_in = d_species[ip->ispecies].mass;
    const double v2 = ip->v[0]*ip->v[0] + ip->v[1]*ip->v[1] + ip->v[2]*ip->v[2];
    double E_in_eV = 0.0;
    if (mass_in > 0.0) E_in_eV = 0.5 * mass_in * v2 * evconv;
    double theta_in_deg = 0.0;
    {
      const double nlen = sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
      const double vlen = sqrt(v2);
      if (nlen > 0.0 && vlen > 0.0) {
        double cos_th = -(ip->v[0]*norm[0] + ip->v[1]*norm[1] + ip->v[2]*norm[2])
                       / (nlen * vlen);
        if (cos_th < 0.0) cos_th = 0.0;
        if (cos_th > 1.0) cos_th = 1.0;
        theta_in_deg = acos(cos_th) * 180.0 / Reflection::PI_CONST;
      }
    }

    const int i_idx = ip - d_particles.data();
    double pw_inc = fnum_c;
    if (pw_slot >= 0) pw_inc = custom_.get_dvec(pw_slot, i_idx);

    const double theta_eff =
        (theta_in_deg > rough_c) ? theta_in_deg - rough_c : 0.0;

    // ---- additive self-sputtering (before the reflect/absorb lottery) ----

    int nsput_total = 0;
    for (int i = 0; i < n; i++) {
      const int m = d_list(ip->ispecies,i);
      if (d_type(m) != PWI_SPUTTER) continue;

      double Y;
      if (d_sput(m) >= 0) Y = sput_yield(d_sput(m), E_in_eV, theta_eff);
      else {
        Eckstein::SputterParams p;
        p.Es = d_spp(m,0); p.Eth = d_spp(m,1); p.Q = d_spp(m,2); p.ETF = d_spp(m,3);
        Y = Eckstein::sputter_yield(E_in_eV, theta_eff, p);
      }
      if (Y <= 0.0) continue;

      int nemit = (int) Y;
      if (rand_gen.drand() < Y - nemit) nemit++;
      if (nemit == 0) continue;
      if (nemit > 20) nemit = 20;    // corrupt-table guard (CPU warns once)

      const int sp = d_prod(m);
      const double mass = d_species[sp].mass;
      const double Es = d_spp(m,0);
      const double gamma = 4.0 * mass_in * mass / ((mass_in + mass) * (mass_in + mass));
      const double Emax_ej = gamma * E_in_eV - Es;
      if (Emax_ej <= 0.0) continue;

      nsput_total += nemit;
      for (int k = 0; k < nemit; k++) {
        double E_eV = sample_thompson(Es, Emax_ej, rand_gen);
        double x[3], v[3];
        x[0] = ip->x[0]; x[1] = ip->x[1]; x[2] = ip->x[2];
        cosine_velocity(v, norm, E_eV, mass, rand_gen);

        int id = MAXSMALLINT*rand_gen.drand();
        int index;
        if (ATOMIC_REDUCTION == 0) {
          index = d_nlocal();
          d_nlocal()++;
        } else
          index = Kokkos::atomic_fetch_add(&d_nlocal(),1);

        int reallocflag = ParticleKokkos::add_particle_kokkos(d_particles,index,
                                          id,sp,ip->icell,x,v,0.0,0.0);
        if (reallocflag) {
          d_retry() = 1;
          rand_pool.free_state(rand_gen);
          return 0;
        }

        // complete the newborn: fly like jpart from the collision point
        // with the remaining timestep, excluding the source surface

        Particle::OnePart *np = &d_particles[index];
        np->flag = PWI_PSURF + 1 + isurf;
        np->dtremain = dtremain;
        np->weight = ip->weight;
        custom_.zero_all(index);
        if (pw_slot >= 0) custom_.set_dvec(pw_slot, index, pw_inc);

        if (ATOMIC_REDUCTION == 0) {
          d_nsingle()++;
          d_tally_single(m)++;
        } else {
          Kokkos::atomic_inc(&d_nsingle());
          Kokkos::atomic_inc(&d_tally_single(m));
        }
      }

      if (sigma_on)
        sigma_acc(isurf, sp, -((double) nemit) * pw_inc);
    }

    if (ehist_on)
      ehist_acc(E_in_eV, theta_in_deg, pw_inc, nsput_total > 0, ip->ispecies);

    // ---- first-to-fire reflect/absorb lottery ----

    double react_prob = 0.0;
    const double random_prob = rand_gen.drand();

    for (int i = 0; i < n; i++) {
      const int m = d_list(ip->ispecies,i);
      const int type = d_type(m);
      if (type == PWI_SPUTTER) continue;

      double p_this;
      if (type == PWI_TRIM_REFLECT)
        p_this = Reflection::R_N_interp(trim_view(d_trim(m)), E_in_eV, theta_in_deg);
      else
        p_this = d_prob(m);

      react_prob += p_this;
      if (react_prob > random_prob) {
        if (ATOMIC_REDUCTION == 0) {
          d_nsingle()++;
          d_tally_single(m)++;
        } else {
          Kokkos::atomic_inc(&d_nsingle());
          Kokkos::atomic_inc(&d_tally_single(m));
        }
        velreset = 1;

        if (type == PWI_TRIM_REFLECT) {
          const int sp0 = d_prod(m);
          double u1 = rand_gen.drand();
          double u2 = rand_gen.drand();
          double u3 = rand_gen.drand();
          double E_out_eV = 0.0, cos_polar = 1.0, cos_azim = 1.0;
          Reflection::sample_reflection(trim_view(d_trim(m)), E_in_eV,
                                        theta_in_deg, u1, u2, u3,
                                        &E_out_eV, &cos_polar, &cos_azim);
          double v_out[3];
          reflected_velocity(v_out, ip->v, norm, E_out_eV, cos_polar,
                             cos_azim, d_species[sp0].mass, rand_gen);
          // internal energy: products are guarded monatomic at init, so
          // the twall accommodation branch is identically zero
          ip->erot = 0.0;
          ip->evib = 0.0;
          ip->ispecies = sp0;
          ip->v[0] = v_out[0];
          ip->v[1] = v_out[1];
          ip->v[2] = v_out[2];
          rand_pool.free_state(rand_gen);
          return (m + 1);

        } else {  // PWI_ABSORB_REEMIT, simple return (reactant == product)
          const double R_rec = d_Rrec(m);
          const double u = rand_gen.drand();
          if (u < R_rec) {
            // atomic re-emission at the wall temperature
            const int sp0 = ip->ispecies;
            const double T_out = (twall_c > 0.0) ? twall_c : 300.0;
            thermal_flux_velocity(ip->v, norm, T_out, d_species[sp0].mass,
                                  rand_gen);
            ip->erot = 0.0;
            ip->evib = 0.0;
            rand_pool.free_state(rand_gen);
            return (m + 1);
          } else {
            // retained: deposit into the areal-density ledger, delete
            if (sigma_on)
              sigma_acc(isurf, ip->ispecies, pw_inc);
            ip = NULL;
            rand_pool.free_state(rand_gen);
            return (m + 1);
          }
        }
      }
    }

    rand_pool.free_state(rand_gen);
    return 0;
  }
};

}

#endif
#endif
