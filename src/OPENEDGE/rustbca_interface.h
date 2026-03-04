/* ----------------------------------------------------------------------
    OpenEdge: C FFI wrapper for RustBCA shared library
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - Austin Nichols (ORNL, nicholsa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    Provides a C++ interface to the RustBCA binary-collision approximation
    code via its C FFI (librustbca.so).  The library is loaded dynamically
    at runtime using dlopen/dlsym.

    RustBCA C FFI functions (from rustbca/src/lib.rs):
      compound_bca_list_c()  — main entry point for layered targets

    This wrapper manages:
      - Dynamic library loading
      - Input/output data marshalling
      - Species Z/M lookup
      - Error handling
------------------------------------------------------------------------- */

#ifndef SPARTA_RUSTBCA_INTERFACE_H
#define SPARTA_RUSTBCA_INTERFACE_H

#include <string>
#include <vector>

namespace SPARTA_NS {

// Result from a single BCA simulation
struct BCAResult {
  // Reflected particles
  struct Particle {
    int Z;            // atomic number
    double M;         // mass (amu)
    double E;         // energy (eV)
    double ux, uy, uz;  // direction cosines
  };

  std::vector<Particle> reflected;
  std::vector<Particle> sputtered;

  // Implant information
  double implant_depth;  // Angstroms, 0 if particle was reflected
  bool was_implanted;    // true if projectile stopped in target
};

// Layer description for compound targets
struct BCALayer {
  double thickness;                  // Angstroms
  double density;                    // atoms/cm^3
  std::vector<int> Z;               // atomic numbers of constituents
  std::vector<double> M;            // masses (amu)
  std::vector<double> fraction;     // fractional composition
  std::vector<double> Eb;           // bulk binding energy (eV)
  std::vector<double> Es;           // surface binding energy (eV)
  std::vector<double> Ec;           // cutoff energy (eV)
};

class RustBCAInterface {
 public:
  RustBCAInterface();
  ~RustBCAInterface();

  // Load the RustBCA shared library
  // Returns 0 on success, -1 on failure
  int load_library(const std::string &lib_path);

  // Run a single BCA simulation
  // projectile: Z, M (amu), E (eV), direction (ux,uy,uz)
  // target: vector of layers from surface to bulk
  BCAResult run_bca(int proj_Z, double proj_M, double proj_E,
                    double ux, double uy, double uz,
                    const std::vector<BCALayer> &target);

  // Check if library is loaded
  bool is_loaded() const { return lib_handle != nullptr; }

  // Get error message from last operation
  const std::string &last_error() const { return error_msg; }

 private:
  void *lib_handle;
  std::string error_msg;

  // Function pointer type for RustBCA compound_bca_list_c
  // This matches the C FFI signature from RustBCA
  typedef void (*compound_bca_list_fn)(
      double,              // ux
      double,              // uy
      double,              // uz
      double,              // E (eV)
      double,              // Z1 (projectile)
      double,              // m1 (amu)
      double,              // Ec1 (cutoff energy)
      double,              // Es1 (surface binding)
      int,                 // num_species_target
      double *,            // Z2 array
      double *,            // m2 array
      double *,            // n2 array (number densities)
      double *,            // Ec2 array
      double *,            // Es2 array
      double *,            // Eb2 array
      int,                 // num_layers
      double *,            // layer_thicknesses
      // Output arrays (pre-allocated):
      int *,               // num_reflected
      double *,            // reflected_data [max_particles * 9]
      int *,               // num_sputtered
      double *,            // sputtered_data [max_particles * 9]
      double *             // implant_depth
  );

  compound_bca_list_fn bca_fn;

  static const int MAX_OUTPUT_PARTICLES = 100;
};

}

#endif
