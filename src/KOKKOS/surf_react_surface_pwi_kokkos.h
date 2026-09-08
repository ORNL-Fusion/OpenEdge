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
  // item 2 (2026-09-08): D/E/R channels, molecular A, per-surf twall/R
  DAT::t_int_1d d_prod2;            // [nlist] D-channel second product, -1 otherwise
  DAT::t_float_1d d_e0, d_e1;       // [nlist] energy[0], energy[1] (D/E: eV per product; A: R, f_mol)
  DAT::t_float_1d d_twall_surf;     // per-surf wall temperature custom (twall_surf)
  DAT::t_float_1d d_R_surf;         // per-surf recycling coefficient custom (R_surf)
  DAT::t_float_2d_lr d_spp;         // [nlist][4] Eckstein Es,Eth,Q,ETF
  DAT::t_float_1d d_yscale;         // [nlist] S `yscale` yield multiplier (slag 2026-08-28)

  // TRIM reflection tables, fixed EIRENE-schema sizes (NE=12,NTHETA=7,NQ=5)
  DAT::t_float_2d_lr d_tr_E, d_tr_th, d_tr_raar;
  DAT::t_float_2d_lr d_tr_RN, d_tr_Eq, d_tr_Emin, d_tr_Emax;
  DAT::t_float_2d_lr d_tr_cp, d_tr_ca;

  // 2D sputter-yield tables, padded to max dims
  DAT::t_int_1d d_su_NE, d_su_NT;
  DAT::t_float_2d_lr d_su_E, d_su_th, d_su_Y;   // Y: [il*NE*NTHETA + ie*NTHETA + ia]
  // Phase D (2026-09-08): lead axis of a 3D table -- kind 0 = plain 2D,
  // 1 = concentration axis (compound target / rtable), 2 = wall-T axis
  DAT::t_int_1d d_su_NL, d_su_kind;
  DAT::t_float_2d_lr d_su_ax;                  // [nsu][maxNL] C or T_K, ascending
  DAT::t_int_1d d_mat_isp, d_conc_isp, d_refl_tbl;  // [nlist] per-reaction
  DAT::t_float_2d_lr d_sconc;                  // [nslocal][sigma_ncols] surface conc
  int conc_dev_on_;                            // d_sconc live (sigma_feedback && conc custom)
  // deposit_as <element> <species> (slag e6fd8b8d): credit retained atoms
  // to the deposit material column, debit erosion from the exposed
  // materials in proportion to their reaction-zone concentrations
  DAT::t_int_1d d_dep_alias;                   // [ncols] species col -> credited col
  DAT::t_int_2d d_dep_cols;                    // [ncols][maxdep] debit candidate cols
  DAT::t_int_1d d_dep_ncols;                   // [ncols] number of candidates
  int dep_alias_on_;
  int conc_dirty_;                             // host conc changed since last upload
  void upload_conc();

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
  int twall_surf_on, R_surf_on;     // per-surf customs bound for this call
  int collide_rot_c, vibstyle_c;    // erot/evib gates (twins of ParticleKokkos)
  double boltz_c;

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
  double sput_slice(int it, int il, double E_eV, double theta_deg) const
  {
    const int NE = d_su_NE(it);
    const int NT = d_su_NT(it);
    if (NE < 2 || NT < 2) return 0.0;
    const int off = il * NE * NT;
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
    const double y00 = d_su_Y(it,off + (ie-1)*NT + (ia-1));
    const double y10 = d_su_Y(it,off + ie*NT + (ia-1));
    const double y01 = d_su_Y(it,off + (ie-1)*NT + ia);
    const double y11 = d_su_Y(it,off + ie*NT + ia);
    double y = (1.0-te)*(1.0-ta)*y00 + te*(1.0-ta)*y10
             + (1.0-te)*ta*y01 + te*ta*y11;
    return (Kokkos::isfinite(y) && y > 0.0) ? y : 0.0;
  }

  // plain 2D table (or slice 0 of a 3D one): TrimSputterTable::yield(E,theta)
  KOKKOS_INLINE_FUNCTION
  double sput_yield(int it, double E_eV, double theta_deg) const
  {
    return sput_slice(it, 0, E_eV, theta_deg);
  }

  // 3D table: linear between the two bracketing lead-axis slices, clamped
  // to the axis -- TrimSputterTable::yield(E,theta,c) / yield_at_T()
  KOKKOS_INLINE_FUNCTION
  double sput_yield_lead(int it, double E_eV, double theta_deg, double x) const
  {
    const int NL = d_su_NL(it);
    if (NL <= 1) return sput_slice(it, 0, E_eV, theta_deg);
    double xx = x;
    if (xx < d_su_ax(it,0)) xx = d_su_ax(it,0);
    if (xx > d_su_ax(it,NL-1)) xx = d_su_ax(it,NL-1);
    int lo = 0, hi = NL;
    while (lo < hi) { int mid = (lo+hi)/2; if (d_su_ax(it,mid) < xx) lo = mid+1; else hi = mid; }
    int il = lo;
    if (il <= 0) il = 1;
    if (il >= NL) il = NL - 1;
    const double x1 = d_su_ax(it,il-1), x2 = d_su_ax(it,il);
    const double f = (x2 != x1) ? (xx - x1) / (x2 - x1) : 0.0;
    return (1.0 - f) * sput_slice(it, il-1, E_eV, theta_deg)
         + f * sput_slice(it, il, E_eV, theta_deg);
  }

  // device twin of SurfReactSurfacePWI::mat_conc(): 1.0 when feedback is
  // off or no conc custom, else the synced per-surf concentration
  KOKKOS_INLINE_FUNCTION
  double mat_conc_dev(int isurf, int isp) const
  {
    if (!conc_dev_on_ || isp < 0) return 1.0;
    return d_sconc(isurf, isp);
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

  // device twins of Particle::erot()/evib() (same gates and sampling as
  // ParticleKokkos::erot/evib); the CPU resamples internal energy of every
  // re-emitted / created product at twall_eff (or its 300 K fallback)
  KOKKOS_INLINE_FUNCTION
  double erot_dev(int isp, double T, rand_type &g) const
  {
    if (!collide_rot_c) return 0.0;
    const int rotdof = d_species[isp].rotdof;
    if (rotdof < 2) return 0.0;
    if (rotdof == 2) return -log(g.drand()) * boltz_c * T;
    const double a = 0.5*rotdof - 1.0;
    const double xmax = a + 1.0 + 9.0*sqrt(a+1.0);
    double erm;
    while (1) {
      erm = xmax*g.drand();
      const double b = pow(erm/a,a) * exp(a-erm);
      if (b > g.drand()) break;
    }
    return erm * boltz_c * T;
  }
  KOKKOS_INLINE_FUNCTION
  double evib_dev(int isp, double T, rand_type &g) const
  {
    enum{NONE,DISCRETE,SMOOTH};
    const int vibdof = d_species[isp].vibdof;
    if (vibstyle_c == NONE || vibdof < 2) return 0.0;
    double eng = 0.0;
    if (vibstyle_c == DISCRETE && vibdof == 2) {
      const int ivib = static_cast<int> (-log(g.drand()) * T / d_species[isp].vibtemp[0]);
      eng = ivib * boltz_c * d_species[isp].vibtemp[0];
    } else if (vibstyle_c == SMOOTH || vibdof >= 2) {
      if (vibdof == 2) eng = -log(g.drand()) * boltz_c * T;
      else if (vibdof > 2) {
        const double a = 0.5*vibdof - 1.0;
        const double xmax = a + 1.0 + 9.0*sqrt(a+1.0);
        double erm;
        while (1) {
          erm = xmax*g.drand();
          const double b = pow(erm/a,a) * exp(a-erm);
          if (b > g.drand()) break;
        }
        eng = erm * boltz_c * T;
      }
    }
    return eng;
  }

  // device twins of deposit_species() and sigma_debit_element()
  KOKKOS_INLINE_FUNCTION
  int deposit_species_dev(int isp) const
  {
    return dep_alias_on_ ? d_dep_alias(isp) : isp;
  }

  KOKKOS_INLINE_FUNCTION
  void sigma_debit_element_dev(int isurf, int isp, double datoms) const
  {
    if (datoms <= 0.0) return;
    if (!dep_alias_on_ || d_dep_ncols(isp) < 2 || !conc_dev_on_) {
      sigma_acc(isurf, isp, -datoms);
      return;
    }
    const int nc = d_dep_ncols(isp);
    double csum = 0.0;
    for (int k = 0; k < nc; k++) {
      const double c = d_sconc(isurf, d_dep_cols(isp,k));
      if (c > 0.0) csum += c;
    }
    if (csum <= 0.0) {
      sigma_acc(isurf, isp, -datoms);
      return;
    }
    for (int k = 0; k < nc; k++) {
      const int col = d_dep_cols(isp,k);
      const double c = d_sconc(isurf, col);
      if (c > 0.0) sigma_acc(isurf, col, -datoms * c / csum);
    }
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
    double twall_eff = twall_c;                       // CPU: twall_surf custom if bound
    if (twall_surf_on) twall_eff = d_twall_surf(isurf);

    // ---- additive self-sputtering (before the reflect/absorb lottery) ----

    int nsput_total = 0;
    for (int i = 0; i < n; i++) {
      const int m = d_list(ip->ispecies,i);
      if (d_type(m) != PWI_SPUTTER) continue;

      // CPU parity (surf_react_surface_pwi.cpp react(), S channel):
      // compound table -> Y(E,theta,c) at the local conc of the `conc`
      // species, no mat rescale; T-axis table -> Y(E,theta,twall) then
      // optional mat weight; plain 2D / Eckstein -> optional mat weight.
      double Y;
      if (d_sput(m) >= 0) {
        const int it = d_sput(m);
        const int kind = d_su_kind(it);
        if (kind == 1) {
          const double c = (d_conc_isp(m) >= 0) ? mat_conc_dev(isurf, d_conc_isp(m)) : 1.0;
          Y = sput_yield_lead(it, E_in_eV, theta_eff, c);
        } else if (kind == 2) {
          Y = sput_yield_lead(it, E_in_eV, theta_eff, twall_eff);
          if (d_mat_isp(m) >= 0) Y *= mat_conc_dev(isurf, d_mat_isp(m));
        } else {
          Y = sput_yield(it, E_in_eV, theta_eff);
          if (d_mat_isp(m) >= 0) Y *= mat_conc_dev(isurf, d_mat_isp(m));
        }
      } else {
        Eckstein::SputterParams p;
        p.Es = d_spp(m,0); p.Eth = d_spp(m,1); p.Q = d_spp(m,2); p.ETF = d_spp(m,3);
        Y = Eckstein::sputter_yield(E_in_eV, theta_eff, p);
        if (d_mat_isp(m) >= 0) Y *= mat_conc_dev(isurf, d_mat_isp(m));
      }
      Y *= d_yscale(m);              // CPU: Y *= r->sp_yscale
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

      if (sigma_on)   // CPU: sigma_debit_element (exposed-material split)
        sigma_debit_element_dev(isurf, sp, ((double) nemit) * pw_inc);
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
      if (type == PWI_TRIM_REFLECT) {
        if (d_refl_tbl(m) >= 0) {
          // composition-resolved (or T-axis) reflection coefficient
          // R(E,theta_eff,c) -- CPU `rtable`; roughness shift as on CPU
          const int it = d_refl_tbl(m);
          const double cval = (d_conc_isp(m) >= 0) ? mat_conc_dev(isurf, d_conc_isp(m)) : 1.0;
          p_this = (d_su_kind(it) == 2)
            ? sput_yield_lead(it, E_in_eV, theta_eff, twall_eff)
            : sput_yield_lead(it, E_in_eV, theta_eff, cval);
          if (p_this < 0.0) p_this = 0.0;
          if (p_this > 1.0) p_this = 1.0;
        } else {
          p_this = Reflection::R_N_interp(trim_view(d_trim(m)), E_in_eV, theta_in_deg);
          if (d_mat_isp(m) >= 0) p_this *= mat_conc_dev(isurf, d_mat_isp(m));
        }
      } else
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

        if (type == PWI_DISSOCIATION) {
          const int sp0 = d_prod(m);
          const int sp1 = d_prod2(m);
          const double erot0 = (twall_eff > 0.0) ? erot_dev(sp0, twall_eff, rand_gen) : 0.0;
          const double evib0 = (twall_eff > 0.0) ? evib_dev(sp0, twall_eff, rand_gen) : 0.0;
          const double erot1 = (twall_eff > 0.0) ? erot_dev(sp1, twall_eff, rand_gen) : 0.0;
          const double evib1 = (twall_eff > 0.0) ? evib_dev(sp1, twall_eff, rand_gen) : 0.0;
          ip->ispecies = sp0;
          ip->erot = erot0;
          ip->evib = evib0;
          cosine_velocity(ip->v, norm, d_e0(m), d_species[sp0].mass, rand_gen);

          double x[3], v[3];
          x[0] = ip->x[0]; x[1] = ip->x[1]; x[2] = ip->x[2];
          cosine_velocity(v, norm, d_e1(m), d_species[sp1].mass, rand_gen);
          const int id = MAXSMALLINT*rand_gen.drand();
          int index;
          if (ATOMIC_REDUCTION == 0) {
            index = d_nlocal();
            d_nlocal()++;
          } else
            index = Kokkos::atomic_fetch_add(&d_nlocal(),1);
          const int reallocflag = ParticleKokkos::add_particle_kokkos(
              d_particles,index,id,sp1,ip->icell,x,v,erot1,evib1);
          if (reallocflag) {
            d_retry() = 1;
            rand_pool.free_state(rand_gen);
            return 0;
          }
          // CPU: modify->update_custom(nlocal-1, 0,0,0, zero) -> customs
          // zeroed, pweight defaults to fnum; the move kernel sets
          // flag/dtremain/weight of jp like the CPU caller
          custom_.zero_all(index);
          if (pw_slot >= 0) custom_.set_dvec(pw_slot, index, fnum_c);
          jp = &d_particles[index];
          rand_pool.free_state(rand_gen);
          return (m + 1);

        } else if (type == PWI_EXCHANGE) {
          const int sp0 = d_prod(m);
          if (twall_eff > 0.0) {
            ip->erot = erot_dev(sp0, twall_eff, rand_gen);
            ip->evib = evib_dev(sp0, twall_eff, rand_gen);
          } else if (sp0 != ip->ispecies) {
            ip->erot = 0.0;
            ip->evib = 0.0;
          }
          ip->ispecies = sp0;
          cosine_velocity(ip->v, norm, d_e0(m), d_species[sp0].mass, rand_gen);
          rand_pool.free_state(rand_gen);
          return (m + 1);

        } else if (type == PWI_RECOMBINATION) {
          if (sigma_on)   // CPU: sigma_accumulate(deposit_species(isp))
            sigma_acc(isurf, deposit_species_dev(ip->ispecies), pw_inc);
          ip = NULL;
          rand_pool.free_state(rand_gen);
          return (m + 1);

        } else if (type == PWI_TRIM_REFLECT) {
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
          if (twall_eff > 0.0) {
            ip->erot = erot_dev(sp0, twall_eff, rand_gen);
            ip->evib = evib_dev(sp0, twall_eff, rand_gen);
          } else {
            ip->erot = 0.0;
            ip->evib = 0.0;
          }
          ip->ispecies = sp0;
          ip->v[0] = v_out[0];
          ip->v[1] = v_out[1];
          ip->v[2] = v_out[2];
          rand_pool.free_state(rand_gen);
          return (m + 1);

        } else {  // PWI_ABSORB_REEMIT: CPU lottery (atomic / molecular / pump)
          double R_rec = d_Rrec(m);
          if (R_surf_on) {                 // per-surf R_surf custom, clamped
            R_rec = d_R_surf(isurf);
            if (R_rec < 0.0) R_rec = 0.0;
            if (R_rec > 1.0) R_rec = 1.0;
          }
          const double f_mol = d_e1(m);
          const int sp_atom = ip->ispecies;
          const int sp_mol  = d_prod(m);
          const bool has_mol_channel = (sp_mol != sp_atom);
          double p_atom, p_mol;
          if (has_mol_channel) {
            p_atom = R_rec * (1.0 - f_mol);
            p_mol  = R_rec * f_mol * 0.5;   // 2 atoms -> 1 molecule
          } else {
            p_atom = R_rec;
            p_mol  = 0.0;
          }
          const double T_out = (twall_eff > 0.0) ? twall_eff : 300.0;
          const double u = rand_gen.drand();
          if (u < p_atom + p_mol) {
            // atomic (u < p_atom) or molecular re-emission at the wall temperature
            const int sp_out = (u < p_atom) ? sp_atom : sp_mol;
            thermal_flux_velocity(ip->v, norm, T_out, d_species[sp_out].mass,
                                  rand_gen);
            ip->ispecies = sp_out;
            ip->erot = erot_dev(sp_out, T_out, rand_gen);
            ip->evib = evib_dev(sp_out, T_out, rand_gen);
            rand_pool.free_state(rand_gen);
            return (m + 1);
          } else {
            // retained (pumped): deposit into the areal-density ledger, delete
            if (sigma_on)   // CPU: sigma_accumulate(deposit_species(sp_atom))
              sigma_acc(isurf, deposit_species_dev(sp_atom), pw_inc);
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
