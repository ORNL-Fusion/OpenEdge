/* ----------------------------------------------------------------------
   OpenEdge IEAD lookup table.
   Loads database/iead/iead_database.h5 and provides the convolution

       <Y>(tau, psi, Z, Te) = sum_{Et, theta}
                              Y(Et * Z * Te, theta) f(Et, theta; tau, psi, Z)

   used by compute surface/physical/sputter to fold the ion-energy/angle
   distribution into the Eckstein yield.

   Energy axis is Etilde = E / (Z * Te) per
   docs/iead_normalization.tex section 4 (projectile-sonic
   normalisation). All Z slots share the same Etilde grid; the consumer
   recovers physical energy via E_eV = Etilde * Z * Te_local at runtime.
------------------------------------------------------------------------- */

#ifndef SPARTA_OPENEDGE_IEAD_TABLE_H
#define SPARTA_OPENEDGE_IEAD_TABLE_H

#include <functional>
#include <string>
#include <vector>

namespace SPARTA_NS {

struct IeadTable {
  std::vector<double> tau_grid;          // (n_tau,)
  std::vector<double> psi_grid_deg;      // (n_psi,)
  std::vector<int>    Z_grid;            // (n_Z,)
  std::vector<double> Etilde_edges;      // (n_E + 1,) = E/(Z*Te) bin edges
  std::vector<double> theta_edges_deg;   // (n_theta + 1,)
  std::vector<float>  f;                 // (n_tau, n_psi, n_Z, n_E, n_theta), C-order

  // Sweep conditions (HDF5 attrs)
  double Te_eV_sweep = 0.0;
  double mi_bg_amu = 0.0;

  int n_tau = 0, n_psi = 0, n_Z = 0, n_E = 0, n_theta = 0;

  // Load from HDF5 file. Throws std::runtime_error on failure.
  void load(const std::string &path);

  // Index of `Z` in Z_grid, or -1 if absent.
  int find_z_slot(int Z) const;

  // Index of (tau, psi) cell to use for bilinear interp + the 4 weights.
  // `tau` and `psi_deg` are clamped to the grid range. Returns indices
  // i0, i1 for tau and j0, j1 for psi plus weights wt0/wt1, wp0/wp1.
  void locate_bilinear(double tau, double psi_deg,
                       int &i0, int &i1, double &wt0, double &wt1,
                       int &j0, int &j1, double &wp0, double &wp1) const;

  // Convolve f(Etilde, theta; tau, psi, Z) against an arbitrary yield
  // functor Y(E_eV, theta_deg) using bin-center quadrature. Bin centers
  // for Etilde and theta are constructed from the edges. PDF is already
  // normalised to sum=1 per cell, so the result is < Y > directly.
  // Returns 0 if Z is outside the grid (caller falls back to mean-impact).
  double convolve_yield(double tau, double psi_deg, int Z, double Te_local,
                        const std::function<double(double, double)> &Y) const;
};

}

#endif
