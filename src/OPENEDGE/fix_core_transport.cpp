/* ----------------------------------------------------------------------
   OpenEdge: fix core/transport (see header)
     fix ID core/transport background-ID surf-group mixture-ID psi_n X
         [grid N] [dperp D|file] [vpinch v|file] [nevery M] [nlaunch n]
         [position uniform|entry] [fraction f] [depth L] [dlen L]
         [profiles file] [output file N] [state file] [read_state file] [export file]
         [coefficients effective | physical geometry file]
         [solver substep | split max_dt seconds]
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <queue>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include "fix_core_transport.h"
#include "core_transport_1d.h"
#include "core_transport_exchange.h"
#include "fix_background.h"
#include "process_library.h"
#include "database_paths.h"
#include "openedge_geom.h"
#include "particle.h"
#include "mixture.h"
#include "surf.h"
#include "grid.h"
#include "domain.h"
#include "update.h"
#include "pusher.h"
#include "modify.h"
#include "comm.h"
#include "input.h"
#include "output.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "error.h"

using namespace SPARTA_NS;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};   // several files

static const double QE = 1.602176634e-19;
static const double TWOPI = 6.283185307179586;

static char *copy_arg(const char *s)
{
  char *c = new char[strlen(s)+1];
  strcpy(c,s);
  return c;
}

// two-column text table, '#' comments
static void read_xy(const char *path, std::vector<double> &x, std::vector<double> &y,
                    int ncol, std::vector<std::vector<double>> *extra, Error *error)
{
  std::ifstream in(path);
  if (!in) error->all(FLERR,(std::string("fix core/transport: cannot open ") + path).c_str());
  std::string line;
  while (std::getline(in,line)) {
    size_t h = line.find('#');
    if (h != std::string::npos) line = line.substr(0,h);
    std::istringstream ss(line);
    std::vector<double> v; double t;
    while (ss >> t) v.push_back(t);
    if (v.empty()) continue;
    if ((int) v.size() < ncol)
      error->all(FLERR,(std::string("fix core/transport: short line in ") + path).c_str());
    x.push_back(v[0]); y.push_back(v[1]);
    if (extra) for (int k = 2; k < ncol; k++) (*extra)[k-2].push_back(v[k]);
  }
  if (x.size() < 2) error->all(FLERR,(std::string("fix core/transport: need >= 2 rows in ") + path).c_str());
}

/* ---------------------------------------------------------------------- */

FixCoreTransport::FixCoreTransport(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix core/transport command");

  pd_id = copy_arg(arg[2]);
  pd = nullptr;
  int igroup = surf->find_group(arg[3]);
  if (igroup < 0) error->all(FLERR,"fix core/transport: surf group ID does not exist");
  groupbit = surf->bitmask[igroup];
  imix = particle->find_mixture(arg[4]);
  if (imix < 0) error->all(FLERR,"fix core/transport: mixture ID does not exist");

  psi_n = -1.0; ngrid = 50; dperp = 0.3; vpinch = 0.0; nlaunch = 20;
  position_mode = 0; fraction = 1.0; depth = 0.0; dlen_in = -1.0;
  profiles_file = output_file = state_file = read_state_file = nullptr;
  export_file = nullptr;
  coef_physical = 0; geometry_file = nullptr; geo_axis_rho = -1.0;
  solver_split = 0; split_max_dt = 0.0;
  reflect_groupbit = 0;
  coupling_aux = 0; layer_h = 0.0; marker_weight = 0.0; ex = nullptr; window_index = 0;
  output_every = 0; nevery = 10;

  int iarg = 5;
  while (iarg < narg) {
    const char *k = arg[iarg];
    auto need = [&](int n) { if (iarg+n >= narg) error->all(FLERR,"Illegal fix core/transport command"); };
    if (strcmp(k,"psi_n") == 0) { need(1); psi_n = input->numeric(FLERR,arg[iarg+1]); iarg += 2; }
    else if (strcmp(k,"grid") == 0) { need(1); ngrid = input->inumeric(FLERR,arg[iarg+1]); iarg += 2; }
    else if (strcmp(k,"dperp") == 0) {
      need(1);
      if (path_looks_literal(arg[iarg+1])) read_xy(arg[iarg+1],dprof_x,dprof_y,2,nullptr,error);
      else dperp = input->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(k,"vpinch") == 0) {
      need(1);
      if (path_looks_literal(arg[iarg+1])) read_xy(arg[iarg+1],vprof_x,vprof_y,2,nullptr,error);
      else vpinch = input->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(k,"nevery") == 0) { need(1); nevery = input->inumeric(FLERR,arg[iarg+1]); iarg += 2; }
    else if (strcmp(k,"nlaunch") == 0) { need(1); nlaunch = input->inumeric(FLERR,arg[iarg+1]); iarg += 2; }
    else if (strcmp(k,"position") == 0) {
      need(1);
      if (strcmp(arg[iarg+1],"uniform") == 0) position_mode = 0;
      else if (strcmp(arg[iarg+1],"entry") == 0) position_mode = 1;
      else error->all(FLERR,"fix core/transport position must be uniform or entry");
      iarg += 2;
    } else if (strcmp(k,"fraction") == 0) { need(1); fraction = input->numeric(FLERR,arg[iarg+1]); iarg += 2; }
    else if (strcmp(k,"depth") == 0) { need(1); depth = input->numeric(FLERR,arg[iarg+1]); iarg += 2; }
    else if (strcmp(k,"dlen") == 0) { need(1); dlen_in = input->numeric(FLERR,arg[iarg+1]); iarg += 2; }
    else if (strcmp(k,"profiles") == 0) { need(1); profiles_file = copy_arg(arg[iarg+1]); iarg += 2; }
    else if (strcmp(k,"output") == 0) {
      need(2); output_file = copy_arg(arg[iarg+1]);
      output_every = input->inumeric(FLERR,arg[iarg+2]); iarg += 3;
    } else if (strcmp(k,"state") == 0) { need(1); state_file = copy_arg(arg[iarg+1]); iarg += 2; }
    else if (strcmp(k,"read_state") == 0) { need(1); read_state_file = copy_arg(arg[iarg+1]); iarg += 2; }
    else if (strcmp(k,"export") == 0) { need(1); export_file = copy_arg(arg[iarg+1]); iarg += 2; }
    else if (strcmp(k,"coefficients") == 0) {
      need(1);
      if (strcmp(arg[iarg+1],"effective") == 0) { coef_physical = 0; iarg += 2; }
      else if (strcmp(arg[iarg+1],"physical") == 0) {
        need(3);
        if (strcmp(arg[iarg+2],"geometry") != 0)
          error->all(FLERR,"fix core/transport: coefficients physical needs geometry <file>");
        coef_physical = 1; geometry_file = copy_arg(arg[iarg+3]); iarg += 4;
      } else error->all(FLERR,"fix core/transport coefficients must be effective or physical");
    } else if (strcmp(k,"coupling") == 0) {
      // temporary selector while the auxiliary coupling replaces the legacy one
      need(1);
      if (strcmp(arg[iarg+1],"legacy") == 0) { coupling_aux = 0; iarg += 2; }
      else if (strcmp(arg[iarg+1],"auxiliary") == 0) {
        need(5);
        if (strcmp(arg[iarg+2],"layer") != 0 || strcmp(arg[iarg+4],"marker_weight") != 0)
          error->all(FLERR,"fix core/transport: coupling auxiliary layer <h> marker_weight <w>");
        coupling_aux = 1;
        layer_h = input->numeric(FLERR,arg[iarg+3]);
        marker_weight = input->numeric(FLERR,arg[iarg+5]);
        if (!(layer_h > 0.0) || !(marker_weight > 0.0))
          error->all(FLERR,"fix core/transport: layer and marker_weight must be > 0");
        iarg += 6;
      } else error->all(FLERR,"fix core/transport coupling must be legacy or auxiliary");
    } else if (strcmp(k,"gc_reflect") == 0) {
      // another closed surface that reflects guiding-centre markers (stored GC follows the path)
      need(1);
      int ig = surf->find_group(arg[iarg+1]);
      if (ig < 0) error->all(FLERR,"fix core/transport: gc_reflect surf group does not exist");
      reflect_groupbit |= surf->bitmask[ig];
      iarg += 2;
    } else if (strcmp(k,"solver") == 0) {
      need(1);
      if (strcmp(arg[iarg+1],"substep") == 0) { solver_split = 0; iarg += 2; }
      else if (strcmp(arg[iarg+1],"split") == 0) {
        need(3);
        if (strcmp(arg[iarg+2],"max_dt") != 0)
          error->all(FLERR,"fix core/transport: solver split needs max_dt <seconds>");
        solver_split = 1; split_max_dt = input->numeric(FLERR,arg[iarg+3]); iarg += 4;
        if (!(split_max_dt > 0.0)) error->all(FLERR,"fix core/transport: max_dt must be > 0");
      } else error->all(FLERR,"fix core/transport solver must be substep or split");
    }
    else error->all(FLERR,"Illegal fix core/transport command");
  }
  if (psi_n <= 0.0 || psi_n > 1.0) error->all(FLERR,"fix core/transport: psi_n in (0,1] is required");
  if (ngrid < 4) error->all(FLERR,"fix core/transport: grid must be >= 4");
  if (nevery < 1 || nlaunch < 1) error->all(FLERR,"fix core/transport: nevery and nlaunch must be >= 1");
  if (fraction <= 0.0 || fraction > 1.0) error->all(FLERR,"fix core/transport: fraction in (0,1]");

  // charge-state chain from the mixture: states 0..Zmax, each once
  Mixture *mix = particle->mixture[imix];
  int zmax = -1;
  for (int k = 0; k < mix->nspecies; k++)
    zmax = std::max(zmax,(int) lround(particle->species[mix->species[k]].charge));
  if (zmax < 1) error->all(FLERR,"fix core/transport: mixture must span a charge-state chain");
  nion = zmax + 1;
  spec_of_z.assign(nion,-1);
  for (int k = 0; k < mix->nspecies; k++) {
    int isp = mix->species[k];
    int z = (int) lround(particle->species[isp].charge);
    if (z < 0 || spec_of_z[z] >= 0)
      error->all(FLERR,"fix core/transport: mixture charge states must be distinct and >= 0");
    spec_of_z[z] = isp;
  }
  for (int z = 0; z < nion; z++)
    if (spec_of_z[z] < 0) error->all(FLERR,"fix core/transport: mixture must contain every charge state 0..Zmax");
  element = element_of_ion_name(particle->species[spec_of_z[0]].id);
  std::transform(element.begin(),element.end(),element.begin(),::tolower);
  mass = particle->species[spec_of_z[0]].mass;

  scalar_flag = vector_flag = 1;
  size_vector = 5;
  coupling_every = nevery;
  nevery = 1;  // observe every dt, including changes within a coupling window
  global_freq = 1;

  ct = nullptr; built = 0; rng = nullptr; flip_normal = 0;
  gin_local.assign(nion,0.0); gin.assign(nion,0.0); gout.assign(nion,0.0);
  gin_total = gout_total = emitted_total = emitted_window = 0.0;
  gin_rate = gout_rate = prad = 0.0;
  pool_cap = std::max(256,nlaunch);
  if (position_mode == 1) pool.resize(pool_cap);
  elapsed = last_dt_win = physical_time = 0.0; window_steps = 0; output_fresh = true;
  pweight_index = tag_index = -1;
}

/* ---------------------------------------------------------------------- */

FixCoreTransport::~FixCoreTransport()
{
  if (copy || copymode) return;
  if (update->core_fix == this) {
    update->core_fix = nullptr;
    update->core_groupbit = 0;
    update->core_species = nullptr;
    if (coupling_aux) update->gc_sync_groupbit = 0;
  }
  delete [] pd_id;
  delete [] profiles_file; delete [] output_file; delete [] state_file; delete [] read_state_file;
  delete [] export_file;
  delete [] geometry_file;
  delete ex;
  delete ct;
  delete rng;
}

/* ---------------------------------------------------------------------- */

int FixCoreTransport::setmask()
{
  return START_OF_STEP | END_OF_STEP;
}

/* ---------------------------------------------------------------------- */

void FixCoreTransport::init()
{
  if (sparta->kokkos) error->all(FLERR,"fix core/transport is not ported to the Kokkos mover yet");
  if (!coupling_aux) {
    // legacy absorb/deposit/re-emit: full-orbit markers, hand-off by particle position
    if (update->pusher && update->pusher->pusher_mode != Pusher::PUSHER_BORIS)
      error->all(FLERR,"fix core/transport legacy coupling supports the Boris pusher only (not gca or hybrid)");
  } else {
    // auxiliary coupling exchanges guiding-centre inventory
    if (!update->pusher || update->pusher->pusher_mode == Pusher::PUSHER_BORIS)
      error->all(FLERR,"fix core/transport coupling auxiliary needs the hybrid or gca pusher");
    if (domain->dimension != 2 || !domain->axisymmetric)
      error->all(FLERR,"fix core/transport coupling auxiliary: only 2-D axisymmetric so far");
    if (depth > 0.0 || dlen_in > 0.0 || position_mode != 0)
      error->all(FLERR,"fix core/transport: depth, dlen and position belong to the legacy coupling");
  }
  int ifix = modify->find_fix(pd_id);
  if (ifix < 0) error->all(FLERR,"fix core/transport: background fix ID not found");
  pd = dynamic_cast<FixBackground *>(modify->fix[ifix]);
  if (!pd) error->all(FLERR,"fix core/transport: ID is not a fix background");
  if (!pd->has_equ) error->all(FLERR,"fix core/transport: background has no psi map");
  if (surf->implicit || surf->distributed)
    error->all(FLERR,"fix core/transport needs explicit, non-distributed surfs");
  pweight_index = particle->find_custom((char *) "pweight");
  if (pweight_index < 0) error->all(FLERR,"fix core/transport requires fix particle/weight");

  z_of_spec.assign(particle->nspecies,-1);
  for (int z = 0; z < nion; z++) z_of_spec[spec_of_z[z]] = z;
  if (update->core_fix && update->core_fix != this)
    error->all(FLERR,"only one fix core/transport is allowed");
  update->core_fix = this;
  update->core_groupbit = coupling_aux ? 0 : groupbit;   // auxiliary: no absorption at the cut
  if (coupling_aux) update->gc_sync_groupbit = groupbit | reflect_groupbit;   // guiding centres reflect at the cut
  else if (reflect_groupbit) error->all(FLERR,"fix core/transport: gc_reflect belongs to the auxiliary coupling");
  update->core_species = z_of_spec.data();

  if (!rng) {
    int seed = 0;
    if (comm->me == 0) seed = (int) (update->ranmaster->uniform() * 900000000) + 1;
    MPI_Bcast(&seed,1,MPI_INT,0,world);
    rng = new RanKnuth(seed);
  }
  if (!built) {
    physical_time = update->time + (update->ntimestep-update->time_last_update)*update->dt;
    build_grid();
    build_surface();
    build_profiles();
    build_rates();
    if (coupling_aux) build_layer();
    if (read_state_file) read_state();
    built = 1;
  }
  if (export_file) write_export();
}

/* ----------------------------------------------------------------------
   flux-surface volumes of the closed region psi_N < psi_n around the
   axis (flood fill on the equilibrium grid) -> volume radius grid
------------------------------------------------------------------------- */

double FixCoreTransport::psi_rz(double R, double Z) const
{
  return pd->psi_norm_at(R,Z);
}

void FixCoreTransport::point_rz(const double *x, double &R, double &Z) const
{
  OpenEdge::sparta_to_RZ(x,domain->dimension,domain->axisymmetric != 0,R,Z,
                         pd->column_x0,pd->column_y0);
}

void FixCoreTransport::build_grid()
{
  const int jm = pd->equ_jm, km = pd->equ_km;
  const double denom = pd->psib - pd->psi_axis;
  if (jm < 3 || km < 3 || fabs(denom) < 1e-30)
    error->all(FLERR,"fix core/transport: unusable psi map");
  const int nb = 1000;
  try {
    CoreFluxGeometry geometry(pd->equ_r,pd->equ_z,pd->psirz,
                              pd->psi_axis,pd->psib,psi_n);
    r_axis = geometry.raxis;
    flood_cells.swap(geometry.points);
    vcum.swap(geometry.cumulative);
  } catch (const std::exception &e) { error->all(FLERR,e.what()); }
  const double vtot = vcum[nb];
  rvol_x = sqrt(vtot/(2.0*M_PI*M_PI*r_axis));
  dr = rvol_x/(ngrid - 0.5);

  rr.resize(ngrid); vol.resize(ngrid); psin.resize(ngrid);
  for (int i = 0; i < ngrid; i++) rr[i] = i*dr;
  delete ct;
  ct = new CoreTransport1D(ngrid,nion);
  ct->set_grid(rr.data());
  for (int i = 0; i < ngrid; i++) {
    vol[i] = ct->cell_volume(i)*TWOPI*r_axis;
    double v = 2.0*M_PI*M_PI*r_axis*rr[i]*rr[i];          // volume inside the cell centre
    int b = (int) (std::lower_bound(vcum.begin(),vcum.end(),v) - vcum.begin());
    b = std::min(nb,std::max(1,b));
    double f = (vcum[b] > vcum[b-1]) ? (v - vcum[b-1])/(vcum[b] - vcum[b-1]) : 0.0;
    psin[i] = (b - 1 + f)*psi_n/nb;
  }
  psin[0] = 0.0;
  const size_t nn = (size_t) nion*ngrid;
  n.assign(nn,0.0); src.assign(nn,0.0);
  diff.assign(nn,0.0); conv.assign(nn,0.0); srate.assign(nn,0.0); rrate.assign(nn,0.0);
  depo.assign(ngrid,0.0);
  if (depth <= 0.0) depo[ngrid-1] = 1.0;
  else {
    double s = 0.0;
    for (int i = 0; i < ngrid; i++) { depo[i] = vol[i]*exp(-(rvol_x - rr[i])/depth); s += depo[i]; }
    for (int i = 0; i < ngrid; i++) depo[i] /= s;
  }
  dlen = (dlen_in > 0.0) ? dlen_in : 0.5*dr;
  if (comm->me == 0 && screen)
    fprintf(screen,"fix core/transport: core region psi_N < %g: V = %.4g m3, R_axis = %.3f m, "
            "r_vol = %.4f m, %d cells (dr = %.2f mm)\n",psi_n,vtot,r_axis,rvol_x,ngrid,1e3*dr);
}

/* ----------------------------------------------------------------------
   n_e, T_e, T_i per cell: from a file (rho_N ne Te Ti) or sampled from
   the background over the flood region, extrapolated flat inward where
   the background has no data
------------------------------------------------------------------------- */

double FixCoreTransport::profile_at(const std::vector<double> &x, const std::vector<double> &y,
                                    double r, double fallback) const
{
  if (x.empty()) return fallback;
  if (r <= x.front()) return y.front();
  if (r >= x.back()) return y.back();
  size_t k = std::upper_bound(x.begin(),x.end(),r) - x.begin();
  double f = (r - x[k-1])/(x[k] - x[k-1]);
  return y[k-1] + f*(y[k] - y[k-1]);
}

void FixCoreTransport::build_profiles()
{
  ne.assign(ngrid,0.0); te.assign(ngrid,0.0); ti.assign(ngrid,0.0);
  if (profiles_file) {
    std::vector<double> px, pne; std::vector<std::vector<double>> ex(2);
    read_xy(profiles_file,px,pne,4,&ex,error);
    for (int i = 0; i < ngrid; i++) {
      double rho = sqrt(psin[i]);
      ne[i] = profile_at(px,pne,rho,0.0);
      te[i] = profile_at(px,ex[0],rho,0.0);
      ti[i] = profile_at(px,ex[1],rho,0.0);
    }
  } else {
    const int nb = (int) vcum.size() - 1;
    std::vector<double> w(ngrid,0.0);
    for (const CoreFluxPoint &c : flood_cells) {
      double R = c.R, Z = c.Z;
      double tev = pd->interp2D(pd->temp_e,R,Z), nev = pd->interp2D(pd->dens_e,R,Z);
      if (!(tev > 0.0) || !(nev > 0.0)) continue;
      double tiv = pd->interp2D(pd->temp_i,R,Z);
      double p = c.psi;
      double v = vcum[std::min(nb,std::max(0,(int) (p/psi_n*nb)))];
      double r = sqrt(v/(2.0*M_PI*M_PI*r_axis));
      int i = std::min(ngrid-1,std::max(0,(int) (r/dr + 0.5)));
      double dv = R;                                      // ~ volume weight on a uniform (R,Z) grid
      ne[i] += dv*nev; te[i] += dv*tev; ti[i] += dv*(tiv > 0.0 ? tiv : tev); w[i] += dv;
    }
    int nfill = 0, last = -1;
    for (int i = ngrid-1; i >= 0; i--) {
      if (w[i] > 0.0) { ne[i] /= w[i]; te[i] /= w[i]; ti[i] /= w[i]; last = i; }
      else if (last >= 0) { ne[i] = ne[last]; te[i] = te[last]; ti[i] = ti[last]; nfill++; }
    }
    if (last < 0) {
      // nothing inside the cut: flat profiles from the background just
      // outside the core surface
      const int dim = domain->dimension;
      double sn = 0.0, st = 0.0, si = 0.0; int cnt = 0;
      for (int m : core_elems) {
        double c[3], R, Z;
        for (int d = 0; d < 3; d++) {
          double nd = (dim == 3) ? surf->tris[m].norm[d] : surf->lines[m].norm[d];
          double cd = (dim == 3) ? (surf->tris[m].p1[d]+surf->tris[m].p2[d]+surf->tris[m].p3[d])/3.0
                                 : 0.5*(surf->lines[m].p1[d]+surf->lines[m].p2[d]);
          c[d] = cd + 5.0e-3*(flip_normal ? -nd : nd);
        }
        point_rz(c,R,Z);
        double tev = pd->interp2D(pd->temp_e,R,Z), nev = pd->interp2D(pd->dens_e,R,Z);
        if (!(tev > 0.0) || !(nev > 0.0)) continue;
        double tiv = pd->interp2D(pd->temp_i,R,Z);
        sn += nev; st += tev; si += (tiv > 0.0 ? tiv : tev); cnt++;
      }
      if (!cnt) error->all(FLERR,"fix core/transport: no background plasma at the core cut; give profiles file");
      for (int i = 0; i < ngrid; i++) { ne[i] = sn/cnt; te[i] = st/cnt; ti[i] = si/cnt; }
      nfill = ngrid;
      if (comm->me == 0 && screen)
        fprintf(screen,"fix core/transport: flat core profiles from the core surface: ne %.3g m-3, Te %.4g eV, Ti %.4g eV\n",
                ne[0],te[0],ti[0]);
    }
    if (nfill && comm->me == 0)
      error->warning(FLERR,"fix core/transport: background has no data in the inner cells, "
                     "profiles extrapolated flat inward (give profiles file for real core profiles)");
  }
  for (int i = 0; i < ngrid; i++)
    if (!(ne[i] > 0.0) || !(te[i] > 0.0) || !(ti[i] > 0.0))
      error->all(FLERR,"fix core/transport: non-positive n_e/T_e/T_i on the 1-D grid");
}

/* ----------------------------------------------------------------------
   ADAS ionization/recombination (1/s) and PLT/PRB (W m3) per cell and
   state, D and v per cell
------------------------------------------------------------------------- */

void FixCoreTransport::build_rates()
{
  ProcessLibrary lib;
  std::string path = resolve_processes_file();
  if (path.empty()) error->all(FLERR,"fix core/transport: processes.h5 not found");
  lib.open(path,world,error);
  struct Tab { std::vector<double> c, lt, ln; int nq = 0, nt = 0, nd = 0; bool ok = false; };
  auto load = [&](const char *cls, Tab &t) {
    t.ok = lib.load_rate(cls,element,t.c,t.lt,t.ln,t.nq,t.nt,t.nd);
  };
  Tab scd, acd, pltt, prbt;
  load("scd",scd); load("acd",acd); load("plt",pltt); load("prb",prbt);
  if (!scd.ok || !acd.ok)
    error->all(FLERR,("fix core/transport: no scd/acd tables for " + element + " in processes.h5").c_str());
  if (scd.nq < nion-1 || acd.nq < nion-1)
    error->all(FLERR,"fix core/transport: ADAS tables have fewer charge states than the mixture");
  auto look = [](const Tab &t, int q, double lte, double lne) {
    auto idx = [](const std::vector<double> &ax, double v, double &f) {
      int n = (int) ax.size();
      if (v <= ax[0]) { f = 0.0; return 0; }
      if (v >= ax[n-1]) { f = 1.0; return n-2; }
      int k = (int) (std::upper_bound(ax.begin(),ax.end(),v) - ax.begin()) - 1;
      f = (v - ax[k])/(ax[k+1] - ax[k]);
      return k;
    };
    double ft, fn;
    int it = idx(t.lt,lte,ft), in = idx(t.ln,lne,fn);
    auto at = [&](int a, int b) { return t.c[(size_t) q*t.nt*t.nd + (size_t) a*t.nd + b]; };
    double v = (1-ft)*(1-fn)*at(it,in) + ft*(1-fn)*at(it+1,in)
             + (1-ft)*fn*at(it,in+1) + ft*fn*at(it+1,in+1);
    return pow(10.0,v);
  };
  if (pltt.ok && prbt.ok && pltt.nq >= nion-1 && prbt.nq >= nion-1) {
    plt.assign((size_t) nion*ngrid,0.0); prb.assign((size_t) nion*ngrid,0.0);
  } else { plt.clear(); prb.clear(); }
  for (int i = 0; i < ngrid; i++) {
    double ne_cm3 = ne[i]*1e-6, lte = log10(te[i]), lne = log10(ne_cm3);
    double rho = sqrt(psin[i]);
    for (int z = 0; z < nion; z++) {
      size_t k = (size_t) z*ngrid + i;
      if (z < nion-1) {
        srate[k] = look(scd,z,lte,lne)*ne_cm3;        // z -> z+1
        rrate[k] = look(acd,z,lte,lne)*ne_cm3;        // z+1 -> z
        if (!plt.empty()) prb[k] = look(prbt,z,lte,lne)*1e-6;   // radiation of z+1 recombining
      }
      if (!plt.empty() && z < pltt.nq) plt[k] = look(pltt,z,lte,lne)*1e-6;
    }
  }

  // transport coefficients: as given (effective) or mapped from physical ones
  if (coef_physical && geo_rho.empty()) read_geometry();
  diff_in.assign(ngrid,0.0); conv_in.assign(ngrid,0.0);
  g1cell.assign(ngrid,1.0); g2cell.assign(ngrid,1.0);
  for (int i = 0; i < ngrid; i++) {
    double rho = sqrt(psin[i]);
    diff_in[i] = profile_at(dprof_x,dprof_y,rho,dperp);
    conv_in[i] = profile_at(vprof_x,vprof_y,rho,vpinch);
    if (coef_physical) {
      g1cell[i] = profile_at(geo_rho,geo_g1,rho,1.0);
      g2cell[i] = profile_at(geo_rho,geo_g2,rho,1.0);
    }
    for (int z = 0; z < nion; z++) {
      diff[(size_t) z*ngrid + i] = g2cell[i]*diff_in[i];
      conv[(size_t) z*ngrid + i] = g1cell[i]*conv_in[i];
    }
  }
}

/* ----------------------------------------------------------------------
   metric factors file: rho_N r_vol g1 g2, optional comment
   "# axis_extrapolated_below_rho_N <value>". Must cover the interface.
------------------------------------------------------------------------- */

void FixCoreTransport::read_geometry()
{
  std::ifstream in(geometry_file);
  if (!in) error->all(FLERR,(std::string("fix core/transport: cannot open geometry file ") + geometry_file).c_str());
  std::string line;
  while (std::getline(in,line)) {
    size_t h = line.find('#');
    if (h != std::string::npos) {
      size_t t = line.find("axis_extrapolated_below_rho_N");
      if (t != std::string::npos) geo_axis_rho = atof(line.c_str() + t + 29);
      line = line.substr(0,h);
    }
    std::istringstream ss(line);
    double v[4]; int nv = 0;
    while (nv < 4 && ss >> v[nv]) nv++;
    if (nv == 0) continue;
    if (nv < 4) error->all(FLERR,"fix core/transport: geometry file needs rho_N r_vol g1 g2");
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !(v[2] > 0.0) || !(v[3] > 0.0) ||
        !std::isfinite(v[2]) || !std::isfinite(v[3]) || (!geo_rho.empty() && v[0] <= geo_rho.back()))
      error->all(FLERR,"fix core/transport: invalid geometry file (finite, g > 0, rho_N increasing)");
    geo_rho.push_back(v[0]); geo_rv.push_back(v[1]); geo_g1.push_back(v[2]); geo_g2.push_back(v[3]);
  }
  if (geo_rho.size() < 2) error->all(FLERR,"fix core/transport: geometry file needs >= 2 rows");
  const double rho_x = sqrt(psi_n);
  if (geo_rho.back() < rho_x*(1.0 - 1e-6))
    error->all(FLERR,"fix core/transport: geometry file does not reach the interface psi_n");
  // same equilibrium? compare the file's volume radius at the interface with ours
  double rv = profile_at(geo_rho,geo_rv,rho_x,0.0);
  if (fabs(rv - rvol_x) > 0.02*rvol_x) {
    char msg[200];
    snprintf(msg,200,"fix core/transport: geometry file r_vol %.4f m at psi_n differs from the psi map's %.4f m",rv,rvol_x);
    error->all(FLERR,msg);
  }
  if (comm->me == 0 && screen)
    fprintf(screen,"fix core/transport: physical coefficients, g2 %.4f, g1 %.4f at the interface; "
            "factors below rho_N %.3f are extrapolated\n",
            profile_at(geo_rho,geo_g2,rho_x,1.0),profile_at(geo_rho,geo_g1,rho_x,1.0),
            geo_axis_rho >= 0.0 ? std::max(geo_axis_rho,geo_rho.front()) : geo_rho.front());
}

int FixCoreTransport::split_steps(double dt_win) const
{
  return std::max(1,(int) ceil(dt_win/split_max_dt*(1.0 - 1e-12)));
}

/* ----------------------------------------------------------------------
   core surf elements: area CDF for uniform emission, psi_N consistency,
   normal orientation (must point toward increasing psi_N)
------------------------------------------------------------------------- */

void FixCoreTransport::build_surface()
{
  const int dim = domain->dimension;
  core_elems.clear(); core_cdf.clear();
  double asum = 0.0, psum = 0.0, pmax = 0.0; int npos = 0, nneg = 0;
  for (int m = 0; m < surf->nlocal; m++) {
    int mask = (dim == 3) ? surf->tris[m].mask : surf->lines[m].mask;
    if (!(mask & groupbit)) continue;
    double c[3], a, nrm[3];
    if (dim == 3) {
      Surf::Tri &t = surf->tris[m];
      double e1[3], e2[3], cr[3];
      for (int d = 0; d < 3; d++) { e1[d] = t.p2[d]-t.p1[d]; e2[d] = t.p3[d]-t.p1[d]; c[d] = (t.p1[d]+t.p2[d]+t.p3[d])/3.0; nrm[d] = t.norm[d]; }
      cr[0] = e1[1]*e2[2]-e1[2]*e2[1]; cr[1] = e1[2]*e2[0]-e1[0]*e2[2]; cr[2] = e1[0]*e2[1]-e1[1]*e2[0];
      a = 0.5*sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
    } else {
      Surf::Line &l = surf->lines[m];
      for (int d = 0; d < 3; d++) { c[d] = 0.5*(l.p1[d]+l.p2[d]); nrm[d] = l.norm[d]; }
      a = sqrt(pow(l.p2[0]-l.p1[0],2) + pow(l.p2[1]-l.p1[1],2));
      if (domain->axisymmetric) a *= TWOPI*c[1];
    }
    asum += a;
    core_elems.push_back(m); core_cdf.push_back(asum);
    double R, Z, R2, Z2, x2[3];
    point_rz(c,R,Z);
    double p = psi_rz(R,Z);
    psum += p; pmax = std::max(pmax,fabs(p - psi_n));
    const double eps = 1e-3;
    for (int d = 0; d < 3; d++) x2[d] = c[d] + eps*nrm[d];
    point_rz(x2,R2,Z2);
    if (psi_rz(R2,Z2) > p) npos++; else nneg++;
  }
  if (core_elems.empty()) error->all(FLERR,"fix core/transport: no surf elements in the core group");
  for (double &v : core_cdf) v /= asum;
  double pmean = psum/core_elems.size();
  if (fabs(pmean - psi_n) > 0.03) {
    char msg[200];
    snprintf(msg,200,"fix core/transport: core surface sits at psi_N = %.3f (max dev %.3f), psi_n says %.3f",
             pmean,pmax,psi_n);
    error->all(FLERR,msg);
  }
  flip_normal = (nneg > npos);
  if (comm->me == 0 && screen)
    fprintf(screen,"fix core/transport: %zu core elements, area %.4g m2, psi_N %.3f +- %.3f, normals %s\n",
            core_elems.size(),asum,pmean,pmax,flip_normal ? "flipped" : "outward");
}

/* ---------------------------------------------------------------------- */

void FixCoreTransport::absorb(int i, int isurf)
{
  double pw = particle->edvec[particle->ewhich[pweight_index]][i];
  if (pw == 0.0) pw = update->fnum;
  int z = z_of_spec[particle->particles[i].ispecies];
  gin_local[z] += pw;
  if (!(pw > 0.0) || !std::isfinite(pw))
    error->one(FLERR,"fix core/transport: invalid absorbed weight");
  if (position_mode == 1) {
    uint64_t event = core_entry_hash((uint64_t) particle->particles[i].id) ^
                     core_entry_hash((uint64_t) update->ntimestep);
    // Include the hit point to distinguish coincident particle IDs.
    const double *x = particle->particles[i].x;
    for (int d = 0; d < 3; ++d) {
      uint64_t bits; memcpy(&bits,x+d,sizeof(bits));
      event = core_entry_hash(event^bits);
    }
    core_entry_offer(pool,event,isurf,x,pw);
  }
}

/* ---------------------------------------------------------------------- */

void FixCoreTransport::start_of_step()
{
  elapsed += update->dt;
  physical_time += update->dt;
  ++window_steps;
}

void FixCoreTransport::end_of_step()
{
  if (window_steps >= coupling_every || update->ntimestep == update->laststep)
    advance_window();
}

void FixCoreTransport::post_run()
{
  // Also flush on timeout/early termination. A run 0 only writes state.
  if (elapsed > 0.0) advance_window();
  if (output_file) write_output();
  if (state_file) write_state();
}

void FixCoreTransport::advance_window()
{
  const double dt_win = elapsed;
  if (!(dt_win > 0.0)) return;
  last_dt_win = dt_win;
  MPI_Allreduce(gin_local.data(),gin.data(),nion,MPI_DOUBLE,MPI_SUM,world);
  std::fill(gin_local.begin(),gin_local.end(),0.0);

  if (coupling_aux) { exchange_window(dt_win); return; }

  gin_rate = 0.0;
  for (int z = 0; z < nion; z++) {
    double g = gin[z]/fraction/dt_win;                   // atoms/s, torus
    gin_rate += g;
    for (int i = 0; i < ngrid; i++) src[(size_t) z*ngrid + i] = g*depo[i]/vol[i];
  }
  gin_total += gin_rate*dt_win;

  try {
    if (solver_split)
      ct->advance_split(dt_win,diff.data(),conv.data(),srate.data(),rrate.data(),src.data(),
                        nullptr,dlen,n.data(),split_steps(dt_win));
    else
      ct->advance(dt_win,diff.data(),conv.data(),srate.data(),rrate.data(),nullptr,src.data(),
                  nullptr,dlen,true,n.data());
  } catch (const std::exception &e) { error->all(FLERR,e.what()); }

  gout_rate = 0.0;
  for (int z = 0; z < nion; z++) { gout[z] = ct->outflux(z)*TWOPI*r_axis; gout_rate += gout[z]; }
  gout_total += gout_rate*dt_win;

  prad = 0.0;
  if (!plt.empty())
    for (int i = 0; i < ngrid; i++) {
      double s = 0.0;
      for (int z = 0; z < nion; z++) {
        size_t k = (size_t) z*ngrid + i;
        s += plt[k]*n[k];
        if (z < nion-1) s += prb[k]*n[k + ngrid];
      }
      prad += vol[i]*ne[i]*s;
    }

  emit_window(dt_win);
  elapsed = 0.0; window_steps = 0;

  bigint step = update->ntimestep;
  if (output_file && step != update->laststep && output_every > 0 && step % output_every == 0)
    write_output();
  if (state_file && step != update->laststep && output_every > 0 && step % output_every == 0)
    write_state();
}

/* ----------------------------------------------------------------------
   auxiliary coupling: layer geometry. The marker layer is the shell of
   volume radius [r_x, r_x + h] outside the cut; psi_layer is its outer
   flux surface, from the same refined flux volumes as the core grid.
------------------------------------------------------------------------- */

double FixCoreTransport::psi_of_volume(double v) const
{
  const int nb = (int) vcum_out.size() - 1;
  int b = (int) (std::lower_bound(vcum_out.begin(),vcum_out.end(),v) - vcum_out.begin());
  b = std::min(nb,std::max(1,b));
  double f = (vcum_out[b] > vcum_out[b-1]) ? (v - vcum_out[b-1])/(vcum_out[b] - vcum_out[b-1]) : 0.0;
  return (b - 1 + f)*vcum_out_cut/nb;
}

void FixCoreTransport::build_layer()
{
  layer_cells = std::max(1,(int) lround(layer_h/dr));
  if (layer_cells > ngrid/2) error->all(FLERR,"fix core/transport: layer is wider than half the 1-D grid");
  const double hc = layer_cells*dr, hm = layer_h;
  vol_core_layer = 0.0;
  for (int i = ngrid-layer_cells; i < ngrid; i++) vol_core_layer += vol[i];
  const double vx = 2.0*M_PI*M_PI*r_axis*rvol_x*rvol_x;
  const double vo = 2.0*M_PI*M_PI*r_axis*(rvol_x + hm)*(rvol_x + hm);
  vol_marker_layer = vo - vx;
  vcum_out_cut = psi_n;
  for (int it = 0; it < 12; it++) {
    vcum_out_cut = std::min(0.98,1.3*vcum_out_cut + 0.02);
    try {
      CoreFluxGeometry g(pd->equ_r,pd->equ_z,pd->psirz,pd->psi_axis,pd->psib,vcum_out_cut);
      vcum_out = g.cumulative;
      box_r[0] = box_z[0] = 1e300; box_r[1] = box_z[1] = -1e300;
      for (const CoreFluxPoint &q : g.points) {
        box_r[0] = std::min(box_r[0],q.R); box_r[1] = std::max(box_r[1],q.R);
        box_z[0] = std::min(box_z[0],q.Z); box_z[1] = std::max(box_z[1],q.Z);
      }
    } catch (const std::exception &e) { error->all(FLERR,e.what()); }
    if (vcum_out.back() >= vo) break;
    if (vcum_out_cut >= 0.98) error->all(FLERR,"fix core/transport: marker layer does not fit inside psi_N 0.98");
  }
  psi_layer = psi_of_volume(vo);
  delete ex; ex = new CoreExchange(nion);
  if (comm->me == 0 && screen)
    fprintf(screen,"fix core/transport: auxiliary coupling, core layer %d cells (%.3g m, %.4g m3), "
            "marker layer psi_N %.4f..%.4f (%.3g m, %.4g m3), marker weight %.3g\n",
            layer_cells,hc,vol_core_layer,psi_n,psi_layer,hm,vol_marker_layer,marker_weight);
}

double FixCoreTransport::pending_total() const
{
  double s = 0.0;
  if (ex) for (int z = 0; z < nion; z++) s += ex->pending(z);
  return s;
}

/* ----------------------------------------------------------------------
   one coupling window: closed-boundary 1-D advance, then the two-sided
   exchange. Layer membership from valid stored guiding centres; a chain
   marker near the layer without one (Boris fallback, invalidated state)
   is an error before any inventory moves.
------------------------------------------------------------------------- */

void FixCoreTransport::exchange_window(double dt_win)
{
  Pusher *pu = update->pusher;
  ++window_index;

  // pass 1 (before anything moves): classify the chain markers. In or next to the
  // layer a marker must carry a valid stored guiding centre pushed by the GCA
  // branch, and that guiding centre must be on the marker side of the cut.
  Particle::OnePart *p = particle->particles;
  const int nlocal = particle->nlocal;
  const double *pw = particle->edvec[particle->ewhich[pweight_index]];
  const double width = psi_layer - psi_n;
  const double guard = psi_layer + 0.5*width, inside_tol = 0.01*width;
  std::vector<int> layer_list;
  long bad_loc[5] = {0,0,0,-1,-1}, bad[5];
  for (int i = 0; i < nlocal; i++) {
    if (z_of_spec[p[i].ispecies] < 0) continue;
    Pusher::GuidingCentre g;
    double R, Z;
    if (!pu->guiding_centre(i,nullptr,g) || g.mode != 1) {
      point_rz(p[i].x,R,Z);
      if (psi_rz(R,Z) < guard) { bad_loc[0]++; if (g.mode != 1) bad_loc[1]++; bad_loc[3] = p[i].id; }
      continue;
    }
    point_rz(g.X,R,Z);
    const double ps = psi_rz(R,Z);
    if (ps < psi_n - inside_tol) { bad_loc[2]++; bad_loc[4] = p[i].id; continue; }
    if (ps < psi_layer) layer_list.push_back(i);
  }
  MPI_Allreduce(bad_loc,bad,3,MPI_LONG,MPI_SUM,world);
  MPI_Allreduce(&bad_loc[3],&bad[3],2,MPI_LONG,MPI_MAX,world);
  if (bad[0]) {
    char msg[256];
    snprintf(msg,256,"fix core/transport: %ld chain marker(s) in or next to the coupling layer without a valid "
             "GCA guiding centre (%ld not on the GCA branch, the rest invalidated GCA states); e.g. marker id %ld",
             bad[0],bad[1],bad[3]);
    error->all(FLERR,msg);
  }
  if (bad[2]) {
    char msg[256];
    snprintf(msg,256,"fix core/transport: %ld guiding centre(s) inside the core cut (psi_N < %.6f); e.g. marker id %ld",
             bad[2],psi_n - inside_tol,bad[4]);
    error->all(FLERR,msg);
  }

  try {
    if (solver_split)
      ct->advance_split(dt_win,diff.data(),conv.data(),srate.data(),rrate.data(),nullptr,
                        nullptr,0.0,n.data(),split_steps(dt_win));
    else
      ct->advance(dt_win,diff.data(),conv.data(),srate.data(),rrate.data(),nullptr,nullptr,
                  nullptr,0.0,true,n.data());
  } catch (const std::exception &e) { error->all(FLERR,e.what()); }

  const double hc = layer_cells*dr, ell = 0.5*(hc + layer_h);
  const double area_d = TWOPI*rvol_x*TWOPI*r_axis*diff[(size_t) ngrid - 1];   // V' D_eff at the cut
  try { ex->set_interval(area_d/ell/vol_core_layer,area_d/ell/vol_marker_layer,dt_win); }
  catch (const std::exception &e) { error->all(FLERR,e.what()); }

  std::vector<int> dellist;
  std::vector<double> removed_loc(nion,0.0), removed(nion,0.0);
  for (int i : layer_list)
    if (CoreExchange::uniform((uint64_t) (unsigned) p[i].id + ((uint64_t) p[i].ispecies << 40),
                              (uint64_t) window_index,7) < ex->p_markers_to_core()) {
      removed_loc[z_of_spec[p[i].ispecies]] += (pw[i] == 0.0 ? update->fnum : pw[i])/fraction;
      dellist.push_back(i);
    }
  MPI_Allreduce(removed_loc.data(),removed.data(),nion,MPI_DOUBLE,MPI_SUM,world);

  // pass 2: transfer (torus inventories), update the core layer, delete and create markers
  std::vector<double> layer(nion,0.0), change(nion), wref(nion,marker_weight/fraction);
  std::vector<int> nemit(nion,0);
  for (int z = 0; z < nion; z++)
    for (int i = ngrid-layer_cells; i < ngrid; i++) layer[z] += n[(size_t) z*ngrid + i]*vol[i];
  try { ex->transfer(layer.data(),removed.data(),wref.data(),change.data(),nemit.data()); }
  catch (const std::exception &e) { error->all(FLERR,e.what()); }
  gin_rate = gout_rate = 0.0; emitted_window = 0.0;
  for (int z = 0; z < nion; z++) {
    for (int i = ngrid-layer_cells; i < ngrid; i++) {
      size_t k = (size_t) z*ngrid + i;
      n[k] = n[k]*(1.0 - ex->p_core_to_markers()) + removed[z]/vol_core_layer;
    }
    gin[z] = removed[z]*fraction; gout[z] = nemit[z]*wref[z]/dt_win;
    gin_rate += removed[z]/dt_win; gout_rate += gout[z];
    emitted_window += nemit[z]*marker_weight;
  }
  gin_total += gin_rate*dt_win; gout_total += gout_rate*dt_win; emitted_total += emitted_window;

  if (!dellist.empty()) { particle->compress_reactions((int) dellist.size(),dellist.data()); particle->sorted = 0; }
  const int nlocal0 = particle->nlocal;
  for (int z = 0; z < nion; z++)
    for (int m = 0; m < nemit[z]; m++) {
      // uniform in the layer volume: R with density ~ R in the box, Z uniform, accept inside the shell
      double R, Z, u[5];
      for (int it = 0; ; it++) {
        if (it > 100000) error->all(FLERR,"fix core/transport: cannot place a marker in the coupling layer");
        R = sqrt(box_r[0]*box_r[0] + rng->uniform()*(box_r[1]*box_r[1] - box_r[0]*box_r[0]));
        Z = box_z[0] + rng->uniform()*(box_z[1] - box_z[0]);
        const double ps = psi_rz(R,Z);
        if (ps >= psi_n && ps < psi_layer) break;
      }
      for (int k = 0; k < 5; k++) u[k] = rng->uniform();
      create_gc_marker(z,R,Z,u,(int) (MAXSMALLINT*rng->uniform()));
    }
  bigint made_loc = particle->nlocal - nlocal0, made;
  MPI_Allreduce(&made_loc,&made,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
  bigint want = 0; for (int z = 0; z < nion; z++) want += nemit[z];
  if (made != want) error->all(FLERR,"fix core/transport: created marker count differs from the exchange's");
  if (made_loc) particle->sorted = 0;

  prad = 0.0;
  if (!plt.empty())
    for (int i = 0; i < ngrid; i++) {
      double s = 0.0;
      for (int z = 0; z < nion; z++) {
        size_t k = (size_t) z*ngrid + i;
        s += plt[k]*n[k];
        if (z < nion-1) s += prb[k]*n[k + ngrid];
      }
      prad += vol[i]*ne[i]*s;
    }
  elapsed = 0.0; window_steps = 0;
  bigint step = update->ntimestep;
  if (output_file && step != update->laststep && output_every > 0 && step % output_every == 0) write_output();
  if (state_file && step != update->laststep && output_every > 0 && step % output_every == 0) write_state();
}

// every rank calls this with the same arguments; only the owner of the cell creates the marker
void FixCoreTransport::create_gc_marker(int z, double R, double Z, double u[5], int id)
{
  double x[3] = {Z,R,0.0}, v[3] = {0.0,0.0,0.0};          // axisymmetric slots: x = Z, y = R
  int icell = grid->id_find_child(0,0,domain->boxlo,domain->boxhi,x);
  if (icell < 0 || icell >= grid->nlocal) return;
  if (grid->cells[icell].nsplit > 1) icell = update->split2d(icell,x);
  if (icell < 0) error->one(FLERR,"fix core/transport: no cell for a marker in the coupling layer");
  particle->add_particle(id,spec_of_z[z],icell,x,v,0.0,0.0);
  const int i = particle->nlocal - 1;
  GCAPusher::GCAFields F;
  if (!update->pusher->sample_gca_fields(x,icell,i,F) || !(F.Bmag > 0.0))
    error->one(FLERR,"fix core/transport: no magnetic field for a marker in the coupling layer");
  double vpar, mu;
  GCAPusher::sample_thermal_gc(ti[ngrid-1]*QE,mass,F.Bmag,u[0],u[1],u[2],vpar,mu);
  if (!update->pusher->set_guiding_centre(i,x,vpar,mu,F.B,u[3]))
    error->one(FLERR,"fix core/transport: the pusher refused the guiding-centre state of a created marker");
  particle->edvec[particle->ewhich[pweight_index]][i] = marker_weight;
  if (update->tag_index >= 0) particle->eivec[particle->ewhich[update->tag_index]][i] = 1;
}

/* ----------------------------------------------------------------------
   re-emit the window's outflux per state as markers of equal weight,
   same random sequence on every rank: a rank creates only the markers
   that land in its own cells
------------------------------------------------------------------------- */

void FixCoreTransport::emit_window(double dt_win)
{
  emitted_window = 0.0;
  double ptot = 0.0;
  std::vector<double> p(nion);
  for (int z = 0; z < nion; z++) { p[z] = gout[z]*fraction*dt_win; ptot += p[z]; }
  if (ptot <= 0.0) return;
  const double w0 = ptot/nlaunch;

  std::vector<double> gpool;
  int ng = 0;
  if (position_mode == 1) {
    struct KeyRank { double key; int rank; };
    std::vector<KeyRank> local(pool_cap), winner(pool_cap);
    for (int k = 0; k < pool_cap; ++k) {
      local[k].key = pool[k].key; local[k].rank = comm->me;
    }
    MPI_Allreduce(local.data(),winner.data(),pool_cap,MPI_DOUBLE_INT,MPI_MINLOC,world);
    if (std::isfinite(winner[0].key)) {
      std::vector<double> data((size_t) 5*pool_cap,0.0);
      gpool.resize(data.size());
      for (int k = 0; k < pool_cap; ++k) if (winner[k].rank == comm->me) {
        data[5*k] = pool[k].surf;
        for (int d = 0; d < 3; ++d) data[5*k+1+d] = pool[k].x[d];
        data[5*k+4] = pool[k].weight;
      }
      MPI_Allreduce(data.data(),gpool.data(),(int) data.size(),MPI_DOUBLE,MPI_SUM,world);
      ng = pool_cap;
    }
  }

  int ncreated = 0;
  const int nlocal0 = particle->nlocal;
  for (int z = 0; z < nion; z++) {
    if (p[z] <= 0.0) continue;
    int nz = (int) (p[z]/w0);
    if (rng->uniform() < p[z]/w0 - nz) nz++;
    for (int m = 0; m < nz; m++) {
      int isurf; double xs[3], nrm[3];
      if (ng > 0) {
        int k = std::min(ng-1,(int) (rng->uniform()*ng));
        isurf = (int) gpool[5*k]; xs[0] = gpool[5*k+1]; xs[1] = gpool[5*k+2]; xs[2] = gpool[5*k+3];
      } else {
        double u = rng->uniform();
        int k = (int) (std::lower_bound(core_cdf.begin(),core_cdf.end(),u) - core_cdf.begin());
        isurf = core_elems[std::min((int) core_elems.size()-1,k)];
        surf_point(isurf,xs,nrm);
      }
      emit_marker(z,isurf,xs,w0);
      emitted_window += w0;
    }
  }
  ncreated = particle->nlocal - nlocal0;
  if (ncreated) particle->sorted = 0;
  double local_created = 0.0, global_created;
  const double *pw = particle->edvec[particle->ewhich[pweight_index]];
  for (int i = nlocal0; i < particle->nlocal; ++i) local_created += pw[i];
  MPI_Allreduce(&local_created,&global_created,1,MPI_DOUBLE,MPI_SUM,world);
  if (fabs(global_created-emitted_window) > 1e-10*emitted_window)
    error->all(FLERR,"fix core/transport: created weight differs from attempted emission");
  emitted_window = global_created;
  emitted_total += emitted_window;
}

void FixCoreTransport::surf_point(int isurf, double *x, double *nrm)
{
  if (domain->dimension == 3) {
    Surf::Tri &t = surf->tris[isurf];
    double s = sqrt(rng->uniform()), b = rng->uniform()*s, a = 1.0 - s;
    for (int d = 0; d < 3; d++) { x[d] = t.p1[d] + a*(t.p2[d]-t.p1[d]) + b*(t.p3[d]-t.p1[d]); nrm[d] = t.norm[d]; }
  } else {
    Surf::Line &l = surf->lines[isurf];
    double u = rng->uniform();
    for (int d = 0; d < 3; d++) { x[d] = l.p1[d] + u*(l.p2[d]-l.p1[d]); nrm[d] = l.norm[d]; }
    x[2] = 0.0;
  }
}

void FixCoreTransport::emit_marker(int z, int isurf, const double *xs, double weight)
{
  const int dim = domain->dimension;
  double nrm[3];
  const double *sn = (dim == 3) ? surf->tris[isurf].norm : surf->lines[isurf].norm;
  for (int d = 0; d < 3; d++) nrm[d] = flip_normal ? -sn[d] : sn[d];
  double x[3];
  for (int d = 0; d < 3; d++) x[d] = xs[d] + 1e-7*nrm[d];
  if (dim == 2) x[2] = 0.0;

  // half-Maxwellian flux at T_i of the outer cell, outward along the normal
  const double kT = ti[ngrid-1]*QE, sig = sqrt(kT/mass);
  double t1[3], t2[3];
  if (fabs(nrm[0]) < 0.9) { t1[0] = 0.0; t1[1] = nrm[2]; t1[2] = -nrm[1]; }
  else { t1[0] = -nrm[2]; t1[1] = 0.0; t1[2] = nrm[0]; }
  double l = sqrt(t1[0]*t1[0]+t1[1]*t1[1]+t1[2]*t1[2]);
  for (int d = 0; d < 3; d++) t1[d] /= l;
  t2[0] = nrm[1]*t1[2]-nrm[2]*t1[1]; t2[1] = nrm[2]*t1[0]-nrm[0]*t1[2]; t2[2] = nrm[0]*t1[1]-nrm[1]*t1[0];
  double u1 = std::max(1e-300,rng->uniform()), u2 = rng->uniform(), u3 = std::max(1e-300,rng->uniform());
  double vn = sqrt(-2.0*kT/mass*log(u1));
  double g = sqrt(-2.0*log(u3));
  double va = sig*g*cos(TWOPI*u2), vb = sig*g*sin(TWOPI*u2);
  double v[3];
  for (int d = 0; d < 3; d++) v[d] = vn*nrm[d] + va*t1[d] + vb*t2[d];
  int id = (int) (MAXSMALLINT*rng->uniform());
  // every rank has drawn the same numbers up to here

  int icell = grid->id_find_child(0,0,domain->boxlo,domain->boxhi,x);
  if (icell < 0 || icell >= grid->nlocal) return;
  if (grid->cells[icell].nsplit > 1)
    icell = (dim == 3) ? update->split3d(icell,x) : update->split2d(icell,x);
  if (icell < 0 || icell >= grid->nlocal || grid->cinfo[icell].volume <= 0.0) return;

  particle->add_particle(id,spec_of_z[z],icell,x,v,0.0,0.0);
  int inew = particle->nlocal - 1;
  Particle::OnePart *pp = &particle->particles[inew];
  pp->flag = PKEEP;
  pp->weight = grid->cinfo[icell].weight;
  pp->dtremain = 0.0;
  if (modify->n_update_custom) {
    double temp_K = ti[ngrid-1]*QE/update->boltz;
    double vstream[3] = {0.0,0.0,0.0};
    modify->update_custom(inew,temp_K,temp_K,temp_K,vstream);
  }
  particle->edvec[particle->ewhich[pweight_index]][inew] = weight;
  if (update->tag_index >= 0)
    particle->eivec[particle->ewhich[update->tag_index]][inew] = 1;
}

/* ---------------------------------------------------------------------- */

double FixCoreTransport::compute_scalar()
{
  if (!ct) return 0.0;
  double s = 0.0;
  for (int z = 0; z < nion; z++) s += ct->content(z,n.data());
  return s*TWOPI*r_axis;
}

double FixCoreTransport::compute_vector(int i)
{
  if (i == 0) return gin_rate;
  if (i == 1) return gout_rate;
  if (i == 2) return prad;
  if (i == 3) return emitted_window;
  return pending_total();                    // torus atoms owed to the markers, not yet emitted
}

/* ---------------------------------------------------------------------- */

void FixCoreTransport::write_output()
{
  if (comm->me != 0) return;
  FILE *fp = fopen(output_file,(output_fresh && !read_state_file) ? "w" : "a");
  output_fresh = false;
  if (!fp) error->one(FLERR,"fix core/transport: cannot write output file");
  fprintf(fp,"# step %ld time %.6e Ncore %.6e Gin %.6e Gout %.6e Prad %.6e emitted %.6e Gin_total %.6e Gout_total %.6e emitted_total %.6e dt_window %.17g substeps %d pending %.6e\n",
          (long) update->ntimestep,physical_time,compute_scalar(),gin_rate,gout_rate,prad,
          emitted_window,gin_total,gout_total,emitted_total,last_dt_win,ct->substeps(),pending_total());
  fprintf(fp,"# i rvol psiN ne Te Ti prad_density n_0 ... n_%d\n",nion-1);
  for (int i = 0; i < ngrid; i++) {
    double s = 0.0;
    if (!plt.empty())
      for (int z = 0; z < nion; z++) {
        size_t k = (size_t) z*ngrid + i;
        s += plt[k]*n[k] + ((z < nion-1) ? prb[k]*n[k + ngrid] : 0.0);
      }
    fprintf(fp,"%d %.6e %.5f %.4e %.4e %.4e %.4e",i,rr[i],psin[i],ne[i],te[i],ti[i],ne[i]*s);
    for (int z = 0; z < nion; z++) fprintf(fp," %.4e",n[(size_t) z*ngrid + i]);
    fprintf(fp,"\n");
  }
  fprintf(fp,"# Z Gin Gout\n");
  for (int z = 0; z < nion; z++)
    fprintf(fp,"Z %d %.5e %.5e\n",z,(last_dt_win > 0.0 ? gin[z]/fraction/last_dt_win : 0.0),gout[z]);
  fclose(fp);
}

// Export the actual solver inputs, not a second reconstruction of them.
// Row-major state arrays use [charge][radial cell]. All numbers are SI.
void FixCoreTransport::write_export()
{
  if (comm->me != 0) return;
  FILE *fp = fopen(export_file,"w");
  if (!fp) error->one(FLERR,"fix core/transport: cannot write export file");
  fprintf(fp,"{\n  \"format\": \"openedge-core-transport-v2\",\n"
          "  \"coefficients\": \"%s\", \"geometry_file\": \"%s\",\n"
          "  \"geometry_axis_extrapolated_below_rho\": %.17g,\n"
          "  \"solver\": \"%s\", \"split_max_dt\": %.17g,\n"
          "  \"ngrid\": %d, \"nstates\": %d,\n"
          "  \"psi_n\": %.17g, \"r_axis\": %.17g, \"rvol_edge\": %.17g,\n"
          "  \"dlen\": %.17g, \"depth\": %.17g, \"fraction\": %.17g",
          coef_physical ? "physical" : "effective",geometry_file ? geometry_file : "",
          coef_physical ? (geo_axis_rho >= 0.0 ? std::max(geo_axis_rho,geo_rho.front()) : geo_rho.front()) : -1.0,
          solver_split ? "split" : "substep",split_max_dt,
          ngrid,nion,psi_n,r_axis,rvol_x,dlen,depth,fraction);
  auto array = [&](const char *name, const std::vector<double> &v) {
    fprintf(fp,",\n  \"%s\": [",name);
    for (size_t k = 0; k < v.size(); ++k) fprintf(fp,"%s%.17g",k ? ", " : "",v[k]);
    fprintf(fp,"]");
  };
  std::vector<double> per_length(ngrid);
  for (int i = 0; i < ngrid; ++i) per_length[i] = ct->cell_volume(i);
  array("rr",rr); array("psiN",psin);
  array("vol_per_length",per_length); array("vol_torus",vol);
  array("ne",ne); array("te",te); array("ti",ti);
  array("diff_input",diff_in); array("conv_input",conv_in);
  array("g1",g1cell); array("g2",g2cell);
  array("diff",diff); array("conv",conv);
  array("srate",srate); array("rrate",rrate); array("depo",depo);
  fprintf(fp,"\n}\n");
  int failed = ferror(fp);
  if (fclose(fp) || failed) error->one(FLERR,"fix core/transport: error writing export file");
}

void FixCoreTransport::write_state()
{
  if (comm->me != 0) return;
  FILE *fp = fopen(state_file,"w");
  if (!fp) error->one(FLERR,"fix core/transport: cannot write state file");
  fprintf(fp,"%d %d %.17g %.17g %.17g %.17g\n",ngrid,nion,rvol_x,gin_total,gout_total,emitted_total);
  for (int z = 0; z < nion; z++) {
    for (int i = 0; i < ngrid; i++) fprintf(fp," %.17g",n[(size_t) z*ngrid + i]);
    fprintf(fp,"\n");
  }
  fclose(fp);
}

void FixCoreTransport::read_state()
{
  std::ifstream in(read_state_file);
  if (!in) error->all(FLERR,"fix core/transport: cannot open read_state file");
  int ng, ni; double rx;
  in >> ng >> ni >> rx >> gin_total >> gout_total >> emitted_total;
  if (ng != ngrid || ni != nion || fabs(rx - rvol_x) > 1e-6*rvol_x)
    error->all(FLERR,"fix core/transport: read_state file does not match the grid or the chain");
  for (size_t k = 0; k < n.size(); k++) in >> n[k];
  if (!in) error->all(FLERR,"fix core/transport: short read_state file");
}
