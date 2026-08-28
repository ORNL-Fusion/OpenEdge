/* ----------------------------------------------------------------------
   OpenEdge: fix surface/emit/source — Kokkos backend (gate 10).
   See the header for scope. Kernel is a device twin of the CPU
   perform_task() non-perspecies FLOW branch (Thompson / fixed-energy).
------------------------------------------------------------------------- */

#include "fix_surface_emit_source_kokkos.h"

#include <cstdlib>

#include "comm.h"
#include "domain.h"
#include "error.h"
#include "mixture.h"
#include "particle_kokkos.h"
#include "surf_kokkos.h"
#include "sparta.h"
#include "sparta_masks.h"
#include "update.h"
#include "update_kokkos.h"

using namespace SPARTA_NS;

// file-local twins of fix_surface_emit_source.cpp constants
enum{FLOW_LOC,CONSTANT_LOC};                 // npmode
enum{PKEEP_LOC,PINSERT_LOC,PDONE_LOC,PDISCARD_LOC,
     PENTRY_LOC,PEXIT_LOC,PSURF_LOC};        // update.cpp particle flags
static constexpr double EV_TO_J_LOC = 1.602176634e-19;
static constexpr double MY_2PI_EMIT = 6.28318530717958647692;

/* ---------------------------------------------------------------------- */

FixSurfaceEmitSourceKokkos::FixSurfaceEmitSourceKokkos(SPARTA *sparta,
                                                       int narg, char **arg) :
  FixSurfaceEmitSource(sparta, narg, arg),
  // distinct base seed per OpenEdge pool (PWI 22345, coulomb 32345,
  // chem 42345, cd 52345)
  rand_pool(62345 + comm->me
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.init(random);
#endif
  kokkos_flag = 1;
  execution_space = Device;
  datamask_read = PARTICLE_MASK | SPECIES_MASK;
  datamask_modify = PARTICLE_MASK;

  device_ok = 0;
  warned_fallback = 0;
  tasks_uploaded_ = 0;
}

/* ---------------------------------------------------------------------- */

FixSurfaceEmitSourceKokkos::~FixSurfaceEmitSourceKokkos()
{
  if (copymode) return;
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
#endif
}

/* ---------------------------------------------------------------------- */

void FixSurfaceEmitSourceKokkos::init()
{
  FixSurfaceEmitSource::init();

  device_ok = 1;
  const char *why = nullptr;
  if (getenv("OE_EMIT_HOST")) {
    device_ok = 0; why = "OE_EMIT_HOST env override";
  } else if (domain->dimension != 3) {
    device_ok = 0; why = "2D (device emission is 3D-only)";
  } else if (perspecies) {
    device_ok = 0; why = "perspecies emission";
  } else if (region) {
    device_ok = 0; why = "region filter";
  } else if (npmode != FLOW_LOC) {
    device_ok = 0; why = "np constant mode";
  } else if (emit_model != MODEL_THOMPSON &&
             emit_model != MODEL_FIXED_ENERGY) {
    device_ok = 0; why = "thermal / thermal_tsurf model";
  } else if (!nlaunch_total_mode) {
    device_ok = 0; why = "per-fnum / per-task launch mode (device is nlaunch_total only)";
  } else if (!dynamic_cast<UpdateKokkos *>(update)) {
    device_ok = 0; why = "no UpdateKokkos (host run)";
  }
  if (device_ok) {
    // erot/evib are emitted 0 on device: reject polyatomic mixture species
    int *msp = particle->mixture[imix]->species;
    for (int isp = 0; isp < nspecies; isp++) {
      const int sp = msp[isp];
      if (particle->species[sp].rotdof >= 2 ||
          particle->species[sp].vibdof >= 2) {
        device_ok = 0; why = "polyatomic mixture species (erot/evib sampling)";
        break;
      }
    }
  }

  if (!device_ok) {
    if (comm->me == 0 && screen && !warned_fallback)
      fprintf(screen,"fix surface/emit/source/kk: HOST emission (%s)\n",why);
    warned_fallback = 1;
    return;
  }
  if (comm->me == 0 && screen)
    fprintf(screen,"fix surface/emit/source/kk: device emission armed "
            "(activates once the per-task source is static)\n");
  tasks_uploaded_ = 0;
}

/* ---------------------------------------------------------------------- */

void FixSurfaceEmitSourceKokkos::grid_changed()
{
  FixSurfaceEmitSource::grid_changed();
  tasks_uploaded_ = 0;
}

/* ----------------------------------------------------------------------
   flatten the task list onto device views (per task build)
------------------------------------------------------------------------- */

void FixSurfaceEmitSourceKokkos::upload_tasks()
{
  const int nt = ntask;
  ntask_ = nt;

  d_t_pcell  = DAT::t_int_1d(Kokkos::view_alloc("emit:pcell",
                 Kokkos::WithoutInitializing), nt > 0 ? nt : 1);
  d_t_isurf  = DAT::t_int_1d(Kokkos::view_alloc("emit:isurf",
                 Kokkos::WithoutInitializing), nt > 0 ? nt : 1);
  d_t_npoint = DAT::t_int_1d(Kokkos::view_alloc("emit:npoint",
                 Kokkos::WithoutInitializing), nt > 0 ? nt : 1);
  d_t_tan1 = DAT::t_float_2d_lr(Kokkos::view_alloc("emit:tan1",
                 Kokkos::WithoutInitializing), nt > 0 ? nt : 1, 3);
  d_t_tan2 = DAT::t_float_2d_lr(Kokkos::view_alloc("emit:tan2",
                 Kokkos::WithoutInitializing), nt > 0 ? nt : 1, 3);
  d_t_vstream = DAT::t_float_2d_lr(Kokkos::view_alloc("emit:vstream",
                 Kokkos::WithoutInitializing), nt > 0 ? nt : 1, 3);
  d_t_poff = DAT::t_int_1d(Kokkos::view_alloc("emit:poff",
                 Kokkos::WithoutInitializing), nt + 1);

  auto h_pcell  = Kokkos::create_mirror_view(d_t_pcell);
  auto h_isurf  = Kokkos::create_mirror_view(d_t_isurf);
  auto h_npoint = Kokkos::create_mirror_view(d_t_npoint);
  auto h_tan1   = Kokkos::create_mirror_view(d_t_tan1);
  auto h_tan2   = Kokkos::create_mirror_view(d_t_tan2);
  auto h_vs     = Kokkos::create_mirror_view(d_t_vstream);
  auto h_poff   = Kokkos::create_mirror_view(d_t_poff);

  int npts = 0;
  for (int i = 0; i < nt; i++) {
    h_poff(i) = npts;
    npts += tasks[i].npoint;
  }
  h_poff(nt) = npts;

  d_t_path = DAT::t_float_1d(Kokkos::view_alloc("emit:path",
                 Kokkos::WithoutInitializing), npts > 0 ? 3*npts : 1);
  d_t_frac = DAT::t_float_1d(Kokkos::view_alloc("emit:frac",
                 Kokkos::WithoutInitializing), npts > 0 ? npts : 1);
  auto h_path = Kokkos::create_mirror_view(d_t_path);
  auto h_frac = Kokkos::create_mirror_view(d_t_frac);

  for (int i = 0; i < nt; i++) {
    h_pcell(i)  = tasks[i].pcell;
    h_isurf(i)  = (int) tasks[i].isurf;
    h_npoint(i) = tasks[i].npoint;
    for (int c = 0; c < 3; c++) {
      h_tan1(i,c) = tasks[i].tan1[c];
      h_tan2(i,c) = tasks[i].tan2[c];
      h_vs(i,c)   = tasks[i].vstream[c];
    }
    const int off = h_poff(i);
    for (int pnt = 0; pnt < tasks[i].npoint; pnt++)
      for (int c = 0; c < 3; c++)
        h_path(3*(off+pnt)+c) = tasks[i].path[3*pnt+c];
    const int nsub = tasks[i].npoint - 2;
    for (int t = 0; t < nsub; t++)
      h_frac(off + t) = tasks[i].fracarea[t];
  }
  Kokkos::deep_copy(d_t_pcell,h_pcell);
  Kokkos::deep_copy(d_t_isurf,h_isurf);
  Kokkos::deep_copy(d_t_npoint,h_npoint);
  Kokkos::deep_copy(d_t_tan1,h_tan1);
  Kokkos::deep_copy(d_t_tan2,h_tan2);
  Kokkos::deep_copy(d_t_vstream,h_vs);
  Kokkos::deep_copy(d_t_poff,h_poff);
  Kokkos::deep_copy(d_t_path,h_path);
  Kokkos::deep_copy(d_t_frac,h_frac);

  // static per-task source (nlaunch_total mode, frozen upstream)
  d_t_src = DAT::t_float_1d(Kokkos::view_alloc("emit:src",
                Kokkos::WithoutInitializing), nt > 0 ? nt : 1);
  auto h_src = Kokkos::create_mirror_view(d_t_src);
  for (int i = 0; i < nt; i++) h_src(i) = cached_task_source[i];
  Kokkos::deep_copy(d_t_src,h_src);

  // mixture cumulative fractions + species map
  nspecies_mix_ = nspecies;
  d_cumm  = DAT::t_float_1d(Kokkos::view_alloc("emit:cumm",
                Kokkos::WithoutInitializing), nspecies);
  d_mix_sp = DAT::t_int_1d(Kokkos::view_alloc("emit:mixsp",
                Kokkos::WithoutInitializing), nspecies);
  auto h_cumm = Kokkos::create_mirror_view(d_cumm);
  auto h_msp  = Kokkos::create_mirror_view(d_mix_sp);
  int *msp = particle->mixture[imix]->species;
  for (int isp = 0; isp < nspecies; isp++) {
    h_cumm(isp) = cummulative[isp];
    h_msp(isp)  = msp[isp];
  }
  Kokkos::deep_copy(d_cumm,h_cumm);
  Kokkos::deep_copy(d_mix_sp,h_msp);

  tasks_uploaded_ = 1;
}

/* ---------------------------------------------------------------------- */

void FixSurfaceEmitSourceKokkos::perform_task()
{
  // device path requires the static (cached) per-task source: the first
  // step(s) run on the host and populate/freeze the cache
  const bool cache_ready = task_source_cached &&
      (int) cached_task_source.size() == ntask &&
      cached_source_total > 0.0;

  int dev = device_ok;
  const char *why = nullptr;
  if (dev && !cache_ready) { dev = 0; why = nullptr; }  // silent: warmup
  if (dev && update->nsurf_tally > 0) {
    dev = 0; why = "active surf tallies this step";
  }

  if (!dev) {
    if (why && !warned_fallback && comm->me == 0 && screen) {
      fprintf(screen,"fix surface/emit/source/kk: HOST emission (%s)\n",why);
      warned_fallback = 1;
    }
    ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
    particle_kk->sync(Host,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
    particle_kk->modify(Host,PARTICLE_MASK|CUSTOM_MASK);
    FixSurfaceEmitSource::perform_task();
    particle_kk->sync(Device,PARTICLE_MASK|CUSTOM_MASK);
    return;
  }

  if (!tasks_uploaded_) upload_tasks();
  if (ntask_ == 0 && ntask == 0) return;

  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;

  // scalars for the kernel
  dt_eff_ = update->dt * nevery;
  fnum_   = update->fnum;
  src_total_ = cached_source_total;
  nlaunch_total_ = nlaunch_total;
  model_  = emit_model;
  ub_     = model_Ub;
  emax_   = model_Emax;
  cosn_   = model_cos_n;
  efixed_ = model_E_fixed;
  weighted_ = 1;   // nlaunch_total mode is always weighted

  // pre-grow: sum of int(ntarget)+1 over emitting tasks is a hard bound
  {
    long long cap = 0;
    for (int i = 0; i < ntask; i++) {
      const double s = cached_task_source[i];
      if (s <= 0.0) continue;
      cap += (long long)((double) nlaunch_total * s / src_total_) + 1;
    }
    // exact capacity for the in-kernel atomic append
    if (particle->nlocal + cap > particle->maxlocal)
      particle_kk->grow((int)(particle->nlocal + cap - particle->maxlocal));
  }

  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
  d_particles = particle_kk->k_particles.view_device();
  d_species   = particle_kk->k_species.d_view;
  SurfKokkos *surf_kk = (SurfKokkos *) surf;
  surf_kk->sync(Device,ALL_MASK);
  d_tris = surf_kk->k_tris.view_device();

  custom_  = particle_kk->device_custom();
  pw_slot_ = (pweight_ewhich >= 0) ? pweight_ewhich : -1;

  if (!d_new_count.data()) {
    d_new_count = Kokkos::View<int, DeviceType>("emit:newn");
    d_nsingle   = Kokkos::View<int, DeviceType>("emit:nsingle");
  }
  Kokkos::deep_copy(d_new_count, particle->nlocal);
  Kokkos::deep_copy(d_nsingle, 0);

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType,TagFixSurfEmitSource>(0,ntask),*this);
  DeviceType().fence();
  copymode = 0;

  int nnew_total = 0, nsingle_dev = 0;
  Kokkos::deep_copy(nnew_total, d_new_count);
  Kokkos::deep_copy(nsingle_dev, d_nsingle);
  if (nnew_total > particle->nlocal) {
    particle->nlocal = nnew_total;
    particle->sorted = 0;
  }
  nsingle += nsingle_dev;
  particle_kk->modify(Device,PARTICLE_MASK|CUSTOM_MASK);
}

/* ----------------------------------------------------------------------
   one task per thread: stratified ninsert draw + in-kernel creation
   (device twin of the CPU non-perspecies FLOW/nlaunch_total branch)
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixSurfaceEmitSourceKokkos::operator()(TagFixSurfEmitSource,
                                            const int &i) const
{
  const double s = d_t_src(i);
  if (s <= 0.0) return;

  rand_type rand_gen = rand_pool.get_state();

  const double ntarget = (double) nlaunch_total_ * s / src_total_;
  const int ninsert = (int)(ntarget + rand_gen.drand());
  if (ninsert <= 0) { rand_pool.free_state(rand_gen); return; }

  // stratified source conservation (CPU comment applies verbatim)
  const double w_emit = (ntarget >= 1.0)
      ? s / ninsert
      : src_total_ / (double) nlaunch_total_;

  const int isurf = d_t_isurf(i);
  const int pcell = d_t_pcell(i);
  const int off   = d_t_poff(i);
  const int nsub  = d_t_npoint(i) - 2;

  const double *normal = d_tris[isurf].norm;
  double atan[3], btan[3], vstream[3];
  for (int c = 0; c < 3; c++) {
    atan[c]    = d_t_tan1(i,c);
    btan[c]    = d_t_tan2(i,c);
    vstream[c] = d_t_vstream(i,c);
  }
  const double vs_n = vstream[0]*normal[0] + vstream[1]*normal[1] +
                      vstream[2]*normal[2];
  const double vs_a = vstream[0]*atan[0] + vstream[1]*atan[1] +
                      vstream[2]*atan[2];
  const double vs_b = vstream[0]*btan[0] + vstream[1]*btan[1] +
                      vstream[2]*btan[2];

  int nactual = 0;
  for (int m = 0; m < ninsert; m++) {
    // species from the mixture cumulative fractions
    double rn = rand_gen.drand();
    int isp = 0;
    while (isp < nspecies_mix_ - 1 && d_cumm(isp) < rn) isp++;
    const int ispecies = d_mix_sp(isp);

    // position: pick a sub-triangle of the clipped polygon, then a
    // uniform point in it
    rn = rand_gen.drand();
    int n = 0;
    while (n < nsub - 1 && rn >= d_t_frac(off + n)) n++;
    const double *p1 = &d_t_path(3*off);
    const double *p2 = &d_t_path(3*(off+n+1));
    const double *p3 = &d_t_path(3*(off+n+2));
    double alpha = rand_gen.drand();
    double beta  = rand_gen.drand();
    if (alpha + beta > 1.0) { alpha = 1.0 - alpha; beta = 1.0 - beta; }
    double x[3];
    for (int c = 0; c < 3; c++)
      x[c] = p1[c] + alpha*(p2[c]-p1[c]) + beta*(p3[c]-p1[c]);

    // energy + cos^n direction in the surface frame
    double E_eV, cos_n_local = 1.0;
    if (model_ == MODEL_THOMPSON) {
      // Thompson with optional high-E cutoff (CPU sampler, verbatim)
      if (emax_ <= 0.0) {
        double vv = rand_gen.drand();
        if (vv > 1.0 - 1.0e-12) vv = 1.0 - 1.0e-12;
        const double sq = Kokkos::sqrt(vv);
        E_eV = ub_ * sq / (1.0 - sq);
      } else {
        const double rmax = emax_ / (emax_ + ub_);
        const double vmax = rmax * rmax;
        E_eV = 0.5 * ub_;
        for (int it = 0; it < 1000; it++) {
          const double sq = Kokkos::sqrt(rand_gen.drand() * vmax);
          const double E = ub_ * sq / (1.0 - sq);
          const double w = 1.0 - Kokkos::sqrt((E + ub_) / (emax_ + ub_));
          if (rand_gen.drand() < w) { E_eV = E; break; }
        }
      }
      cos_n_local = cosn_;
    } else {              // MODEL_FIXED_ENERGY
      E_eV = efixed_;
    }
    double xi = rand_gen.drand();
    if (xi <= 0.0) xi = 1.0e-12;
    const double cos_th = Kokkos::pow(xi, 1.0 / (cos_n_local + 1.0));
    const double s2 = 1.0 - cos_th*cos_th;
    const double sin_th = (s2 > 0.0) ? Kokkos::sqrt(s2) : 0.0;
    const double phi_az = MY_2PI_EMIT * rand_gen.drand();
    const double mass_sp = d_species[ispecies].mass;
    const double speed = (mass_sp > 0.0)
        ? Kokkos::sqrt(2.0 * E_eV * EV_TO_J_LOC / mass_sp) : 0.0;
    const double vnmag = speed * cos_th + vs_n;
    const double vamag = speed * sin_th * Kokkos::cos(phi_az) + vs_a;
    const double vbmag = speed * sin_th * Kokkos::sin(phi_az) + vs_b;

    double v[3];
    for (int c = 0; c < 3; c++)
      v[c] = vnmag*normal[c] + vamag*atan[c] + vbmag*btan[c];

    const int id = (int)(MAXSMALLINT * rand_gen.drand());
    const int index = Kokkos::atomic_fetch_add(&d_new_count(),1);
    const int rf = ParticleKokkos::add_particle_kokkos(
        d_particles,index,id,ispecies,pcell,x,v,0.0,0.0);
    if (rf) continue;   // cannot happen: capacity pre-grown
    nactual++;

    Particle::OnePart *np = &d_particles[index];
    np->flag = PSURF_LOC + 1 + isurf;
    np->dtremain = dt_eff_ * rand_gen.drand();

    // newborn customs: zero + pweight (PWI-newborn approximation of the
    // CPU's update_custom chain)
    custom_.zero_all(index);
    if (pw_slot_ >= 0)
      custom_.set_dvec(pw_slot_, index, weighted_ ? w_emit : fnum_);
  }
  if (nactual) Kokkos::atomic_add(&d_nsingle(), nactual);

  rand_pool.free_state(rand_gen);
}
