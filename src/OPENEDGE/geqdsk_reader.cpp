/* ----------------------------------------------------------------------
    OpenEdge: G-EQDSK equilibrium file reader
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - Austin Nichols (ORNL, nicholsa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    See geqdsk_reader.h for documentation.
------------------------------------------------------------------------- */

#include "geqdsk_reader.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstring>

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

GeqdskReader::GeqdskReader()
{
}

GeqdskReader::~GeqdskReader()
{
}

/* ----------------------------------------------------------------------
   Read a G-EQDSK file
   Format: fixed-width Fortran output (5 values per line, 16 chars each)
------------------------------------------------------------------------- */

int GeqdskReader::read(const std::string &path)
{
  std::ifstream fin(path);
  if (!fin.is_open()) {
    err_msg = "Cannot open G-EQDSK file: " + path;
    return -1;
  }

  // Line 1: case description + idum, nw, nh
  std::string header;
  std::getline(fin, header);
  geq.case_description = header;

  // Parse nw, nh from the end of the header line
  // Format: "description  idum  nw  nh"
  // The last 3 integers on the line
  {
    std::istringstream iss(header);
    std::vector<std::string> tokens;
    std::string tok;
    while (iss >> tok) tokens.push_back(tok);

    if (tokens.size() < 3) {
      err_msg = "G-EQDSK: cannot parse header dimensions";
      return -1;
    }

    try {
      geq.nh = std::stoi(tokens.back());
      geq.nw = std::stoi(tokens[tokens.size() - 2]);
    } catch (...) {
      err_msg = "G-EQDSK: cannot parse nw, nh from header";
      return -1;
    }
  }

  if (geq.nw <= 0 || geq.nh <= 0) {
    err_msg = "G-EQDSK: invalid dimensions";
    return -1;
  }

  // Lines 2-3: scalars (5 per line in 16-char fields)
  // Line 2: rdim, zdim, rcentr, rleft, zmid
  // Line 3: rmaxis, zmaxis, simag, sibry, bcentr
  // Line 4: current, simag(dup), xdum, rmaxis(dup), xdum
  // Line 5: zmaxis(dup), xdum, sibry(dup), xdum, xdum

  auto read_scalars = [&fin](int n) -> std::vector<double> {
    std::vector<double> vals;
    while (static_cast<int>(vals.size()) < n) {
      std::string line;
      if (!std::getline(fin, line)) break;
      // Parse 16-character fields
      for (size_t pos = 0; pos < line.size() && static_cast<int>(vals.size()) < n; pos += 16) {
        std::string field = line.substr(pos, std::min(size_t(16), line.size() - pos));
        try {
          vals.push_back(std::stod(field));
        } catch (...) {
          // Try smaller field width
          std::istringstream iss(field);
          double v;
          if (iss >> v) vals.push_back(v);
        }
      }
    }
    return vals;
  };

  std::vector<double> scalars = read_scalars(20);
  if (scalars.size() < 20) {
    // Try alternative: whitespace-separated format
    fin.clear();
    fin.seekg(0);
    std::string dummy;
    std::getline(fin, dummy); // skip header

    scalars.clear();
    while (scalars.size() < 20) {
      double v;
      if (!(fin >> v)) break;
      scalars.push_back(v);
    }
  }

  if (scalars.size() < 11) {
    err_msg = "G-EQDSK: could not read enough scalar values";
    return -1;
  }

  geq.rdim   = scalars[0];
  geq.zdim   = scalars[1];
  geq.rcentr = scalars[2];
  geq.rleft  = scalars[3];
  geq.zmid   = scalars[4];
  geq.rmaxis = scalars[5];
  geq.zmaxis = scalars[6];
  geq.simag  = scalars[7];
  geq.sibry  = scalars[8];
  geq.bcentr = scalars[9];
  geq.current = scalars[10];

  // Read 1D profile arrays
  geq.fpol   = read_1d_array(fin, geq.nw);
  geq.pres   = read_1d_array(fin, geq.nw);
  geq.ffprim = read_1d_array(fin, geq.nw);
  geq.pprime = read_1d_array(fin, geq.nw);

  // Read 2D psirz array
  geq.psirz = read_2d_array(fin, geq.nw, geq.nh);

  // Read q profile
  geq.qpsi = read_1d_array(fin, geq.nw);

  // Read boundary and limiter
  int nbdry = 0, nlim = 0;
  fin >> nbdry >> nlim;
  geq.nbdry = nbdry;
  geq.nlim = nlim;

  if (nbdry > 0) {
    geq.rbdry.resize(nbdry);
    geq.zbdry.resize(nbdry);
    for (int i = 0; i < nbdry; i++)
      fin >> geq.rbdry[i] >> geq.zbdry[i];
  }

  if (nlim > 0) {
    geq.rlim.resize(nlim);
    geq.zlim.resize(nlim);
    for (int i = 0; i < nlim; i++)
      fin >> geq.rlim[i] >> geq.zlim[i];
  }

  // Build R, Z coordinate arrays
  geq.r.resize(geq.nw);
  geq.z.resize(geq.nh);
  double dr = geq.rdim / (geq.nw - 1);
  double dz = geq.zdim / (geq.nh - 1);
  for (int i = 0; i < geq.nw; i++)
    geq.r[i] = geq.rleft + i * dr;
  for (int j = 0; j < geq.nh; j++)
    geq.z[j] = (geq.zmid - geq.zdim / 2.0) + j * dz;

  return 0;
}

/* ---------------------------------------------------------------------- */

std::vector<double> GeqdskReader::read_1d_array(std::istream &in, int n)
{
  std::vector<double> arr(n);
  for (int i = 0; i < n; i++) {
    if (!(in >> arr[i])) {
      arr[i] = 0.0;
    }
  }
  return arr;
}

/* ---------------------------------------------------------------------- */

std::vector<double> GeqdskReader::read_2d_array(std::istream &in, int nw, int nh)
{
  int total = nw * nh;
  std::vector<double> arr(total);
  for (int i = 0; i < total; i++) {
    if (!(in >> arr[i])) {
      arr[i] = 0.0;
    }
  }
  return arr;
}

/* ----------------------------------------------------------------------
   Compute B-field on the native (R,Z) grid from psi(R,Z) and F(psi)
------------------------------------------------------------------------- */

void GeqdskReader::compute_bfield(
    std::vector<double> &br,
    std::vector<double> &bt,
    std::vector<double> &bz) const
{
  const int nw = geq.nw;
  const int nh = geq.nh;
  const int ntot = nw * nh;

  br.resize(ntot, 0.0);
  bt.resize(ntot, 0.0);
  bz.resize(ntot, 0.0);

  if (geq.r.empty() || geq.z.empty() || geq.psirz.empty()) return;

  const double dr = geq.r[1] - geq.r[0];
  const double dz = geq.z[1] - geq.z[0];

  for (int j = 0; j < nh; j++) {
    for (int i = 0; i < nw; i++) {
      int idx = j * nw + i;
      double R = geq.r[i];
      if (R < 1.0e-10) R = 1.0e-10;

      // dpsi/dR (central differences)
      double dpsi_dR;
      if (i == 0)
        dpsi_dR = (geq.psirz[(j) * nw + (i+1)] - geq.psirz[idx]) / dr;
      else if (i == nw - 1)
        dpsi_dR = (geq.psirz[idx] - geq.psirz[(j) * nw + (i-1)]) / dr;
      else
        dpsi_dR = (geq.psirz[(j) * nw + (i+1)] - geq.psirz[(j) * nw + (i-1)]) / (2.0 * dr);

      // dpsi/dZ (central differences)
      double dpsi_dZ;
      if (j == 0)
        dpsi_dZ = (geq.psirz[(j+1) * nw + i] - geq.psirz[idx]) / dz;
      else if (j == nh - 1)
        dpsi_dZ = (geq.psirz[idx] - geq.psirz[(j-1) * nw + i]) / dz;
      else
        dpsi_dZ = (geq.psirz[(j+1) * nw + i] - geq.psirz[(j-1) * nw + i]) / (2.0 * dz);

      // B_R = -(1/R) * dpsi/dZ
      br[idx] = -dpsi_dZ / R;

      // B_Z = (1/R) * dpsi/dR
      bz[idx] = dpsi_dR / R;

      // B_t = F(psi) / R
      double psi_val = geq.psirz[idx];
      double F = get_fpol(psi_val);
      bt[idx] = F / R;
    }
  }
}

/* ---------------------------------------------------------------------- */

void GeqdskReader::compute_bfield_at_point(
    double R, double Z,
    double &br_out, double &bt_out, double &bz_out) const
{
  br_out = bt_out = bz_out = 0.0;

  if (geq.r.empty() || geq.z.empty() || geq.psirz.empty()) return;

  const int nw = geq.nw;
  const int nh = geq.nh;

  // Clamp to grid
  double R_c = std::min(std::max(R, geq.r.front()), geq.r.back());
  double Z_c = std::min(std::max(Z, geq.z.front()), geq.z.back());

  double dr = geq.r[1] - geq.r[0];
  double dz = geq.z[1] - geq.z[0];

  // Find grid cell
  int i = static_cast<int>((R_c - geq.r[0]) / dr);
  int j = static_cast<int>((Z_c - geq.z[0]) / dz);
  i = std::min(std::max(i, 0), nw - 2);
  j = std::min(std::max(j, 0), nh - 2);

  // Bilinear interpolation weights
  double t = (R_c - geq.r[i]) / dr;
  double u = (Z_c - geq.z[j]) / dz;
  t = std::min(std::max(t, 0.0), 1.0);
  u = std::min(std::max(u, 0.0), 1.0);

  // Interpolate psi
  auto psi_at = [&](int ii, int jj) -> double {
    return geq.psirz[jj * nw + ii];
  };

  // Compute dpsi/dR and dpsi/dZ via finite differences on interpolated values
  // Use central differences around the interpolation point
  double h = dr * 0.01;  // small perturbation
  double hz = dz * 0.01;

  // psi at current point
  double psi00 = (1-t)*(1-u)*psi_at(i,j) + t*(1-u)*psi_at(i+1,j)
               + (1-t)*u*psi_at(i,j+1) + t*u*psi_at(i+1,j+1);

  // dpsi/dR
  double dpR = 0.0;
  if (i > 0 && i < nw - 2) {
    double psi_r_minus = (1-t)*(1-u)*psi_at(i-1,j) + t*(1-u)*psi_at(i,j)
                       + (1-t)*u*psi_at(i-1,j+1) + t*u*psi_at(i,j+1);
    double psi_r_plus  = (1-t)*(1-u)*psi_at(i+1,j) + t*(1-u)*psi_at(i+2>nw-1?nw-1:i+2,j)
                       + (1-t)*u*psi_at(i+1,j+1) + t*u*psi_at(i+2>nw-1?nw-1:i+2,j+1);
    dpR = (psi_r_plus - psi_r_minus) / (2.0 * dr);
  } else {
    // Forward/backward difference
    double psi_p = (1-t)*(1-u)*psi_at(std::min(i+1,nw-1),j) + t*(1-u)*psi_at(std::min(i+2,nw-1),j)
                 + (1-t)*u*psi_at(std::min(i+1,nw-1),j+1) + t*u*psi_at(std::min(i+2,nw-1),j+1);
    dpR = (psi_p - psi00) / dr;
  }

  // dpsi/dZ
  double dpZ = 0.0;
  if (j > 0 && j < nh - 2) {
    double psi_z_minus = (1-t)*(1-u)*psi_at(i,j-1) + t*(1-u)*psi_at(i+1,j-1)
                       + (1-t)*u*psi_at(i,j) + t*u*psi_at(i+1,j);
    double psi_z_plus  = (1-t)*(1-u)*psi_at(i,j+1) + t*(1-u)*psi_at(i+1,j+1)
                       + (1-t)*u*psi_at(i,std::min(j+2,nh-1)) + t*u*psi_at(i+1,std::min(j+2,nh-1));
    dpZ = (psi_z_plus - psi_z_minus) / (2.0 * dz);
  } else {
    double psi_p = (1-t)*(1-u)*psi_at(i,std::min(j+1,nh-1)) + t*(1-u)*psi_at(i+1,std::min(j+1,nh-1))
                 + (1-t)*u*psi_at(i,std::min(j+2,nh-1)) + t*u*psi_at(i+1,std::min(j+2,nh-1));
    dpZ = (psi_p - psi00) / dz;
  }

  if (R_c < 1.0e-10) R_c = 1.0e-10;

  br_out = -dpZ / R_c;
  bz_out = dpR / R_c;

  double F = get_fpol(psi00);
  bt_out = F / R_c;
}

/* ---------------------------------------------------------------------- */

double GeqdskReader::get_fpol(double psi) const
{
  if (geq.fpol.empty()) return geq.bcentr * geq.rcentr;

  const int nw = geq.nw;
  double psi_norm = (psi - geq.simag) / (geq.sibry - geq.simag);
  psi_norm = std::min(std::max(psi_norm, 0.0), 1.0);

  double idx_d = psi_norm * (nw - 1);
  int i0 = static_cast<int>(idx_d);
  i0 = std::min(std::max(i0, 0), nw - 2);
  double frac = idx_d - i0;

  return geq.fpol[i0] * (1.0 - frac) + geq.fpol[i0 + 1] * frac;
}
