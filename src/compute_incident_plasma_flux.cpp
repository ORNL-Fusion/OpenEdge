/* ----------------------------------------------------------------------
   OpenEdge compute incident/flux
   Interpolate incident plasma flux from plasma.h5 onto surface elements.
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_incident_plasma_flux.h"
#include "surf.h"
#include "domain.h"
#include "comm.h"
#include "openedge_geom.h"
#include "update.h"
#include "input.h"
#include "memory.h"
#include "error.h"

#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace SPARTA_NS;

ComputeIncidentPlasmaFlux::ComputeIncidentPlasmaFlux(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal compute incident/plasma/flux command");

  int igroup = surf->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute incident/plasma/flux surf group ID does not exist");
  groupbit = surf->bitmask[igroup];

  if (strcmp(arg[3],"file") != 0)
    error->all(FLERR,"compute incident/plasma/flux syntax: compute ID incident/plasma/flux surfgroup file plasma.h5 nflux_incident");
  plasma_path = std::string(arg[4]);

  std::vector<int> which_tmp;
  std::vector<int> which_species_tmp;
  int iarg = 5;
  auto push_request = [&](int kind, int species_slot) {
    which_tmp.push_back(kind);
    which_species_tmp.push_back(species_slot);
  };
  while (iarg < narg) {
    if (strcmp(arg[iarg],"nflux_incident") == 0) {
      push_request(NFLUX_INCIDENT,0);
      iarg++;
    } else if (strcmp(arg[iarg],"nflux_normal") == 0) {
      push_request(NFLUX_NORMAL,0);
      iarg++;
    } else if (strcmp(arg[iarg],"nflux_species") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"nflux_species needs species slot (1..N)");
      std::string tok(arg[iarg+1]);
      if (tok == "all" || tok == "[all]") {
        const int nall = peek_nspec_from_plasma();
        for (int s = 1; s <= nall; s++) push_request(NFLUX_SPECIES,s);
      } else {
        if (tok.size() > 2 && tok.front() == '[' && tok.back() == ']')
          tok = tok.substr(1,tok.size()-2);
        int slot = 0;
        try {
          slot = std::stoi(tok);
        } catch (...) {
          error->all(FLERR,"nflux_species needs numeric slot or 'all'");
        }
        if (slot <= 0) error->all(FLERR,"nflux_species slot must be >= 1");
        push_request(NFLUX_SPECIES,slot);
      }
      iarg += 2;
    } else error->all(FLERR,"Invalid compute incident/plasma/flux value");
  }
  nvalue = static_cast<int>(which_tmp.size());
  which = new int[nvalue];
  which_species = new int[nvalue];
  for (int i = 0; i < nvalue; i++) {
    which[i] = which_tmp[i];
    which_species[i] = which_species_tmp[i];
  }

  per_surf_flag = 1;
  if (nvalue == 1) size_per_surf_cols = 0;
  else size_per_surf_cols = nvalue;
  surf_tally_flag = 0;
  timeflag = 0;

  dimension = domain->dimension;
  firstflag = 1;
  distributed = 0;
  nsown = 0;
  vector_surf = nullptr;
  array_surf = nullptr;

  nr = nz = nspec = 0;
  has_multi_ion = 0;
  has_bfield = 0;
}

int ComputeIncidentPlasmaFlux::peek_nspec_from_plasma() const
{
  try {
    H5::H5File file(plasma_path, H5F_ACC_RDONLY);
    if (H5Lexists(file.getId(), "ions/dens", H5P_DEFAULT) > 0) {
      H5::DataSet ds = file.openDataSet("ions/dens");
      H5::DataSpace sp = ds.getSpace();
      hsize_t dims[3] = {0,0,0};
      sp.getSimpleExtentDims(dims);
      if (dims[0] > 0) return static_cast<int>(dims[0]);
    }
  } catch (...) {
    // fall through to single-ion default
  }
  return 1;
}

ComputeIncidentPlasmaFlux::~ComputeIncidentPlasmaFlux()
{
  if (copymode) return;
  delete [] which;
  delete [] which_species;
  memory->destroy(vector_surf);
  memory->destroy(array_surf);
}

void ComputeIncidentPlasmaFlux::load_plasma()
{
  H5::H5File file(plasma_path, H5F_ACC_RDONLY);

  auto hasDataset = [&](const std::string& name) -> bool {
    return H5Lexists(file.getId(), name.c_str(), H5P_DEFAULT) > 0;
  };
  auto read1D = [&](const std::string& name, std::vector<double>& out) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t d0;
    sp.getSimpleExtentDims(&d0);
    out.resize(static_cast<size_t>(d0));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };
  auto read2D = [&](const std::string& name, std::vector<double>& out) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[2];
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[0]) != nz || static_cast<int>(dims[1]) != nr)
      throw std::runtime_error("compute incident/plasma/flux: 2D dataset shape mismatch");
    out.resize(static_cast<size_t>(dims[0]*dims[1]));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };
  auto read3D = [&](const std::string& name, std::vector<double>& out, int& ns_out) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[3];
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[1]) != nz || static_cast<int>(dims[2]) != nr)
      throw std::runtime_error("compute incident/plasma/flux: 3D dataset shape mismatch");
    ns_out = static_cast<int>(dims[0]);
    out.resize(static_cast<size_t>(dims[0]*dims[1]*dims[2]));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };

  read1D("r", rvals);
  read1D("z", zvals);
  nr = static_cast<int>(rvals.size());
  nz = static_cast<int>(zvals.size());
  if (nr < 2 || nz < 2)
    throw std::runtime_error("compute incident/plasma/flux: plasma r/z must have size >= 2");

  read2D("dens_i", dens_i);
  read2D("parr_flow", parr_flow);
  if (hasDataset("parr_flow_r")) read2D("parr_flow_r", parr_flow_r);
  if (hasDataset("parr_flow_t")) read2D("parr_flow_t", parr_flow_t);
  if (hasDataset("parr_flow_z")) read2D("parr_flow_z", parr_flow_z);
  if (hasDataset("br")) read2D("br", br);
  if (hasDataset("bt")) read2D("bt", bt);
  if (hasDataset("bz")) read2D("bz", bz);
  has_bfield = (!br.empty() && !bz.empty());

  has_multi_ion = 0;
  if (hasDataset("ions/dens") && hasDataset("ions/parr_flow")) {
    int ns1 = 0, ns2 = 0;
    read3D("ions/dens", ions_dens, ns1);
    read3D("ions/parr_flow", ions_parr_flow, ns2);
    int ns3 = 0, ns4 = 0, ns5 = 0;
    if (hasDataset("ions/parr_flow_r") && hasDataset("ions/parr_flow_z")) {
      read3D("ions/parr_flow_r", ions_parr_flow_r, ns3);
      read3D("ions/parr_flow_z", ions_parr_flow_z, ns4);
      if (hasDataset("ions/parr_flow_t"))
        read3D("ions/parr_flow_t", ions_parr_flow_t, ns5);
    }
    if (ns1 != ns2) throw std::runtime_error("compute incident/plasma/flux: ions/dens vs ions/parr_flow nspec mismatch");
    if (!ions_parr_flow_r.empty() && ns1 != ns3)
      throw std::runtime_error("compute incident/plasma/flux: ions/parr_flow_r nspec mismatch");
    if (!ions_parr_flow_z.empty() && ns1 != ns4)
      throw std::runtime_error("compute incident/plasma/flux: ions/parr_flow_z nspec mismatch");
    if (!ions_parr_flow_t.empty() && ns1 != ns5)
      throw std::runtime_error("compute incident/plasma/flux: ions/parr_flow_t nspec mismatch");
    nspec = ns1;
    has_multi_ion = (nspec > 0);
  }
  if (hasDataset("ion_species/spec_index")) {
    H5::DataSet ds = file.openDataSet("ion_species/spec_index");
    H5::DataSpace sp = ds.getSpace();
    hsize_t d0;
    sp.getSimpleExtentDims(&d0);
    ion_spec_index.resize(static_cast<size_t>(d0));
    ds.read(ion_spec_index.data(), H5::PredType::NATIVE_INT);
  }
}

void ComputeIncidentPlasmaFlux::init()
{
  if (!surf->exist)
    error->all(FLERR,"Cannot use compute incident/plasma/flux when surfs do not exist");
  if (surf->implicit)
    error->all(FLERR,"Cannot use compute incident/plasma/flux with implicit surfs");

  // each rank reads plasma.h5 once (simpler + robust)
  try {
    load_plasma();
  } catch (const std::exception& e) {
    error->all(FLERR, e.what());
  } catch (...) {
    error->all(FLERR, "compute incident/plasma/flux failed reading plasma.h5");
  }

  distributed = surf->distributed;
  if (!firstflag) return;
  firstflag = 0;

  // Follow SPARTA per-surf convention: per-surf vectors/arrays are sized
  // by nown, with local index i mapped to element m depending on layout.
  nsown = surf->nown;
  if (nvalue == 1)
    memory->create(vector_surf,nsown,"incident/flux:vector_surf");
  else
    memory->create(array_surf,nsown,nvalue,"incident/flux:array_surf");
}

double ComputeIncidentPlasmaFlux::interp2D(const std::vector<double> &f, double r, double z) const
{
  if (f.empty()) return 0.0;
  if (r < rvals.front() || r > rvals.back() ||
      z < zvals.front() || z > zvals.back()) return 0.0;
  const double rc = r;
  const double zc = z;
  auto itR = std::lower_bound(rvals.begin(), rvals.end(), rc);
  auto itZ = std::lower_bound(zvals.begin(), zvals.end(), zc);
  int ir2 = static_cast<int>(itR - rvals.begin());
  int iz2 = static_cast<int>(itZ - zvals.begin());
  if (ir2 <= 0) ir2 = 1;
  if (iz2 <= 0) iz2 = 1;
  if (ir2 >= nr) ir2 = nr-1;
  if (iz2 >= nz) iz2 = nz-1;
  const int ir1 = ir2-1;
  const int iz1 = iz2-1;

  const double r1 = rvals[ir1], r2 = rvals[ir2];
  const double z1 = zvals[iz1], z2 = zvals[iz2];
  const double tr = (r2 != r1) ? (rc-r1)/(r2-r1) : 0.0;
  const double tz = (z2 != z1) ? (zc-z1)/(z2-z1) : 0.0;
  const double w11 = (1.0-tr)*(1.0-tz);
  const double w21 = tr*(1.0-tz);
  const double w12 = (1.0-tr)*tz;
  const double w22 = tr*tz;

  auto at2 = [&](int iz, int ir)->double { return f[static_cast<size_t>(iz)*nr + ir]; };
  return w11*at2(iz1,ir1) + w21*at2(iz1,ir2) + w12*at2(iz2,ir1) + w22*at2(iz2,ir2);
}

double ComputeIncidentPlasmaFlux::interp3D(const std::vector<double> &f, int ispec, double r, double z) const
{
  if (f.empty() || ispec < 0 || ispec >= nspec) return 0.0;
  if (r < rvals.front() || r > rvals.back() ||
      z < zvals.front() || z > zvals.back()) return 0.0;
  const double rc = r;
  const double zc = z;
  auto itR = std::lower_bound(rvals.begin(), rvals.end(), rc);
  auto itZ = std::lower_bound(zvals.begin(), zvals.end(), zc);
  int ir2 = static_cast<int>(itR - rvals.begin());
  int iz2 = static_cast<int>(itZ - zvals.begin());
  if (ir2 <= 0) ir2 = 1;
  if (iz2 <= 0) iz2 = 1;
  if (ir2 >= nr) ir2 = nr-1;
  if (iz2 >= nz) iz2 = nz-1;
  const int ir1 = ir2-1;
  const int iz1 = iz2-1;

  const double r1 = rvals[ir1], r2 = rvals[ir2];
  const double z1 = zvals[iz1], z2 = zvals[iz2];
  const double tr = (r2 != r1) ? (rc-r1)/(r2-r1) : 0.0;
  const double tz = (z2 != z1) ? (zc-z1)/(z2-z1) : 0.0;
  const double w11 = (1.0-tr)*(1.0-tz);
  const double w21 = tr*(1.0-tz);
  const double w12 = (1.0-tr)*tz;
  const double w22 = tr*tz;

  const size_t off = static_cast<size_t>(ispec) * nz * nr;
  auto at3 = [&](int iz, int ir)->double { return f[off + static_cast<size_t>(iz)*nr + ir]; };
  return w11*at3(iz1,ir1) + w21*at3(iz1,ir2) + w12*at3(iz2,ir1) + w22*at3(iz2,ir2);
}

void ComputeIncidentPlasmaFlux::compute_per_surf()
{
  invoked_per_surf = update->ntimestep;
  if (nvalue == 1) {
    for (int i = 0; i < nsown; i++) vector_surf[i] = 0.0;
  } else {
    for (int i = 0; i < nsown; i++)
      for (int j = 0; j < nvalue; j++) array_surf[i][j] = 0.0;
  }

  Surf::Line *lines = distributed ? surf->mylines : surf->lines;
  Surf::Tri *tris = distributed ? surf->mytris : surf->tris;
  const int me = comm->me;
  const int nprocs = comm->nprocs;
  for (int i = 0; i < nsown; i++) {
    int m;
    if (distributed) m = i;
    else m = me + i*nprocs;
    const int in_group = (dimension == 2) ? (lines[m].mask & groupbit) : (tris[m].mask & groupbit);
    if (!in_group) continue;

    double r = 0.0, z = 0.0;
    if (dimension == 2) {
      const double mid[3] = {0.5*(lines[m].p1[0] + lines[m].p2[0]),
                             0.5*(lines[m].p1[1] + lines[m].p2[1]), 0.0};
      OpenEdge::sparta_to_RZ(mid, dimension, domain->axisymmetric, r, z);
    } else {
      const double mid[3] = {(tris[m].p1[0] + tris[m].p2[0] + tris[m].p3[0]) / 3.0,
                             (tris[m].p1[1] + tris[m].p2[1] + tris[m].p3[1]) / 3.0,
                             (tris[m].p1[2] + tris[m].p2[2] + tris[m].p3[2]) / 3.0};
      OpenEdge::sparta_to_RZ(mid, dimension, domain->axisymmetric, r, z);
    }

    // Surface normal in (R,Z) plane for 2D.
    double nr_surf = 0.0, nz_surf = 0.0;
    if (dimension == 2) {
      double nphi_unused;
      OpenEdge::sparta_v_to_RZphi(lines[m].norm, dimension,
                                   domain->axisymmetric, 0.0,
                                   nr_surf, nz_surf, nphi_unused);
    } else {
      // For 3D Cartesian tri normals, project to local cylindrical R/Z.
      const double xc = (tris[m].p1[0] + tris[m].p2[0] + tris[m].p3[0]) / 3.0;
      const double yc = (tris[m].p1[1] + tris[m].p2[1] + tris[m].p3[1]) / 3.0;
      const double rr = std::sqrt(xc*xc + yc*yc);
      const double cr = (rr > 0.0) ? xc/rr : 1.0;
      const double sr = (rr > 0.0) ? yc/rr : 0.0;
      nr_surf = tris[m].norm[0]*cr + tris[m].norm[1]*sr;
      nz_surf = tris[m].norm[2];
    }

    // Sample slightly off the wall. Probe both normal directions and keep
    // the side with larger plasma flux (robust to normal orientation).
    const double dr = (nr > 1) ? std::fabs(rvals[1] - rvals[0]) : 0.0;
    const double dz = (nz > 1) ? std::fabs(zvals[1] - zvals[0]) : 0.0;
    const double dprobe = 0.5 * ((dr > 0.0 && dz > 0.0) ? std::min(dr,dz) : (dr + dz));
    auto species_velocity = [&](int s, double rp, double zp,
                                double &ni, double &vpar,
                                double &vr, double &vt, double &vz) {
      ni = 0.0; vpar = 0.0; vr = vt = vz = 0.0;
      if (has_multi_ion) {
        ni = interp3D(ions_dens, s, rp, zp);
        vpar = interp3D(ions_parr_flow, s, rp, zp);
        if (!ions_parr_flow_r.empty() && !ions_parr_flow_z.empty()) {
          vr = interp3D(ions_parr_flow_r, s, rp, zp);
          vz = interp3D(ions_parr_flow_z, s, rp, zp);
          if (!ions_parr_flow_t.empty()) vt = interp3D(ions_parr_flow_t, s, rp, zp);
        } else if (has_bfield) {
          const double brr = interp2D(br, rp, zp);
          const double btt = bt.empty() ? 0.0 : interp2D(bt, rp, zp);
          const double bzz = interp2D(bz, rp, zp);
          const double bmag = std::sqrt(brr*brr + btt*btt + bzz*bzz);
          if (bmag > 0.0) {
            vr = vpar * brr / bmag;
            vt = vpar * btt / bmag;
            vz = vpar * bzz / bmag;
          }
        }
      } else {
        ni = interp2D(dens_i, rp, zp);
        vpar = interp2D(parr_flow, rp, zp);
        if (!parr_flow_r.empty() && !parr_flow_z.empty()) {
          vr = interp2D(parr_flow_r, rp, zp);
          vz = interp2D(parr_flow_z, rp, zp);
          if (!parr_flow_t.empty()) vt = interp2D(parr_flow_t, rp, zp);
        } else if (has_bfield) {
          const double brr = interp2D(br, rp, zp);
          const double btt = bt.empty() ? 0.0 : interp2D(bt, rp, zp);
          const double bzz = interp2D(bz, rp, zp);
          const double bmag = std::sqrt(brr*brr + btt*btt + bzz*bzz);
          if (bmag > 0.0) {
            vr = vpar * brr / bmag;
            vt = vpar * btt / bmag;
            vz = vpar * bzz / bmag;
          }
        }
      }
    };

    auto eval_total = [&](double rp, double zp, double &gtot, double &gnorm) {
      gtot = 0.0;
      gnorm = 0.0;
      if (has_multi_ion) {
        for (int s = 0; s < nspec; s++) {
          double ni,vp,vr,vt,vz;
          species_velocity(s,rp,zp,ni,vp,vr,vt,vz);
          gtot += std::fabs(ni * vp);
          const double vin = -(vr*nr_surf + vz*nz_surf);
          if (vin > 0.0) gnorm += ni * vin;
        }
      } else {
        double ni,vp,vr,vt,vz;
        species_velocity(0,rp,zp,ni,vp,vr,vt,vz);
        gtot = std::fabs(ni * vp);
        const double vin = -(vr*nr_surf + vz*nz_surf);
        if (vin > 0.0) gnorm = ni * vin;
      }
    };

    const double r_minus = r - dprobe * nr_surf;
    const double z_minus = z - dprobe * nz_surf;
    const double r_plus  = r + dprobe * nr_surf;
    const double z_plus  = z + dprobe * nz_surf;

    double gtot_minus = 0.0, gnorm_minus = 0.0;
    double gtot_plus = 0.0, gnorm_plus = 0.0;
    eval_total(r_minus, z_minus, gtot_minus, gnorm_minus);
    eval_total(r_plus, z_plus, gtot_plus, gnorm_plus);

    if (gtot_plus > gtot_minus) {
      r = r_plus; z = z_plus;
    } else {
      r = r_minus; z = z_minus;
    }

    double gamma_total = 0.0;
    double gamma_n_total = 0.0;
    eval_total(r, z, gamma_total, gamma_n_total);

    auto clean = [](double x) {
      return (std::fabs(x) < 1.0e-200 || !std::isfinite(x)) ? 0.0 : x;
    };

    if (nvalue == 1) {
      double g0 = gamma_total;
      if (which[0] == NFLUX_NORMAL) g0 = gamma_n_total;
      else if (which[0] == NFLUX_SPECIES) {
        const int slot = which_species[0] - 1; // user uses 1-based slot
        g0 = 0.0;
        if (has_multi_ion && slot >= 0 && slot < nspec) {
          double ni,vp,vr,vt,vz;
          species_velocity(slot,r,z,ni,vp,vr,vt,vz);
          const double vin = -(vr*nr_surf + vz*nz_surf);
          if (vin > 0.0) g0 = ni * vin;
        }
      }
      vector_surf[i] = clean(g0);
    } else {
      for (int j = 0; j < nvalue; j++) {
        if (which[j] == NFLUX_INCIDENT) array_surf[i][j] = clean(gamma_total);
        else if (which[j] == NFLUX_NORMAL) array_surf[i][j] = clean(gamma_n_total);
        else if (which[j] == NFLUX_SPECIES) {
          const int slot = which_species[j] - 1; // user uses 1-based slot
          double g = 0.0;
          if (has_multi_ion && slot >= 0 && slot < nspec) {
            double ni,vp,vr,vt,vz;
            species_velocity(slot,r,z,ni,vp,vr,vt,vz);
            const double vin = -(vr*nr_surf + vz*nz_surf);
            if (vin > 0.0) g = ni * vin;
          }
          array_surf[i][j] = clean(g);
        }
      }
    }
  }
}

bigint ComputeIncidentPlasmaFlux::memory_usage()
{
  return static_cast<bigint>(nsown) * nvalue * sizeof(double);
}
