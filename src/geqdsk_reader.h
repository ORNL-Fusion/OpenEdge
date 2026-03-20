/* ----------------------------------------------------------------------
    OpenEdge: G-EQDSK equilibrium file reader
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - Austin Nichols (ORNL, nicholsa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    Reads standard G-EQDSK (Grad-Shafranov equilibrium) format files
    and computes magnetic field components from the poloidal flux
    function psi(R,Z) and the toroidal field function F(psi).

    B_R = -(1/R) * dpsi/dZ
    B_Z =  (1/R) * dpsi/dR
    B_t =  F(psi) / R

    The resulting B-field can be output to the BfieldSnapshot struct
    (compatible with compute_bfield_timedep) or written to HDF5.

    G-EQDSK format reference:
      https://fusion.gat.com/theory/Efitgeqdsk
      Fixed-format Fortran-style file with:
        - Header line with case description, dimensions
        - Scalars: rdim, zdim, rcentr, rleft, zmid, rmaxis, zmaxis,
                   simag, sibry, bcentr, current, xdum
        - 1D profiles: fpol, pres, ffprim, pprime, qpsi (all nw points)
        - 2D array: psirz (nw x nh)
        - Boundary/limiter point arrays
------------------------------------------------------------------------- */

#ifndef SPARTA_GEQDSK_READER_H
#define SPARTA_GEQDSK_READER_H

#include <string>
#include <vector>

namespace SPARTA_NS {

struct GEQDSKData {
  // Header
  std::string case_description;
  int nw;             // number of horizontal R grid points
  int nh;             // number of vertical Z grid points

  // Scalars
  double rdim;        // horizontal dimension in meter of computational box
  double zdim;        // vertical dimension
  double rcentr;      // R in meter of vacuum toroidal magnetic field BCENTR
  double rleft;       // minimum R in meter of computational box
  double zmid;        // Z of center of computational box
  double rmaxis;      // R of magnetic axis
  double zmaxis;      // Z of magnetic axis
  double simag;       // poloidal flux at magnetic axis (Weber/rad)
  double sibry;       // poloidal flux at plasma boundary
  double bcentr;      // vacuum toroidal field at RCENTR
  double current;     // plasma current (Amps)

  // 1D profiles (nw points, on uniform psi grid from simag to sibry)
  std::vector<double> fpol;     // F(psi) = R * B_t  (poloidal flux function)
  std::vector<double> pres;     // pressure
  std::vector<double> ffprim;   // FF'(psi)
  std::vector<double> pprime;   // P'(psi)
  std::vector<double> qpsi;     // safety factor q(psi)

  // 2D poloidal flux array: psirz[j * nw + i] where j=Z index, i=R index
  std::vector<double> psirz;    // psi(R,Z) on rectangular grid

  // Grid coordinates (derived)
  std::vector<double> r;        // R grid [nw]
  std::vector<double> z;        // Z grid [nh]

  // Boundary and limiter
  int nbdry;
  std::vector<double> rbdry, zbdry;
  int nlim;
  std::vector<double> rlim, zlim;
};

class GeqdskReader {
 public:
  GeqdskReader();
  ~GeqdskReader();

  // Read a G-EQDSK file
  // Returns 0 on success, -1 on failure
  int read(const std::string &path);

  // Access parsed data
  const GEQDSKData &data() const { return geq; }

  // Compute B-field on the native (R,Z) grid
  // Returns (br, bt, bz) each of size [nh * nw] in row-major [Z][R]
  void compute_bfield(std::vector<double> &br,
                      std::vector<double> &bt,
                      std::vector<double> &bz) const;

  // Compute B-field at a single (R,Z) point via bilinear interpolation
  void compute_bfield_at_point(double R, double Z,
                                double &br, double &bt, double &bz) const;

  // Get F(psi) at a given psi value (linear interpolation)
  double get_fpol(double psi) const;

  // Compute psi(R,Z) at a single point via bilinear interpolation
  double psi_at_point(double R, double Z) const;

  // Compute normalized psi at a point: psi_n = (psi - simag) / (sibry - simag)
  // Returns 0 at magnetic axis, 1 at separatrix, >1 in SOL
  double psi_norm_at_point(double R, double Z) const;

  // Get error message
  const std::string &error_msg() const { return err_msg; }

 private:
  GEQDSKData geq;
  std::string err_msg;

  // Parse helpers for fixed-format Fortran output
  static std::vector<double> read_1d_array(std::istream &in, int n);
  static std::vector<double> read_2d_array(std::istream &in, int nw, int nh);
};

}

#endif
