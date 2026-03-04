/* ----------------------------------------------------------------------
    OpenEdge: C FFI wrapper for RustBCA shared library
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - Austin Nichols (ORNL, nicholsa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    See rustbca_interface.h for documentation.
------------------------------------------------------------------------- */

#include "rustbca_interface.h"

#include <dlfcn.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

RustBCAInterface::RustBCAInterface()
  : lib_handle(nullptr), bca_fn(nullptr)
{
}

/* ---------------------------------------------------------------------- */

RustBCAInterface::~RustBCAInterface()
{
  if (lib_handle) {
    dlclose(lib_handle);
    lib_handle = nullptr;
  }
}

/* ---------------------------------------------------------------------- */

int RustBCAInterface::load_library(const std::string &lib_path)
{
  lib_handle = dlopen(lib_path.c_str(), RTLD_NOW);
  if (!lib_handle) {
    error_msg = std::string("Cannot load RustBCA library: ") + dlerror();
    return -1;
  }

  // Clear any existing error
  dlerror();

  // Load the compound BCA function
  bca_fn = (compound_bca_list_fn) dlsym(lib_handle,
                                          "compound_bca_list_c");
  const char *err = dlerror();
  if (err) {
    error_msg = std::string("Cannot find compound_bca_list_c: ") + err;
    dlclose(lib_handle);
    lib_handle = nullptr;
    bca_fn = nullptr;
    return -1;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

BCAResult RustBCAInterface::run_bca(
    int proj_Z, double proj_M, double proj_E,
    double ux, double uy, double uz,
    const std::vector<BCALayer> &target)
{
  BCAResult result;
  result.implant_depth = 0.0;
  result.was_implanted = false;

  if (!bca_fn) {
    error_msg = "RustBCA library not loaded";
    return result;
  }

  if (target.empty()) {
    error_msg = "No target layers specified";
    return result;
  }

  // Flatten target layers into arrays for the C FFI
  // Count total unique species across all layers
  // For compound_bca_list_c, we need a flat representation

  // Collect all unique species across layers
  int num_layers = static_cast<int>(target.size());

  // For each layer, get the total number of species
  // Use the first layer's species count as representative
  int num_species = 0;
  for (const auto &lyr : target) {
    num_species = std::max(num_species, static_cast<int>(lyr.Z.size()));
  }

  // Build flat arrays for the C interface
  std::vector<double> Z2(num_species, 0.0);
  std::vector<double> m2(num_species, 0.0);
  std::vector<double> n2(num_species, 0.0);
  std::vector<double> Ec2(num_species, 1.0);
  std::vector<double> Es2(num_species, 2.0);
  std::vector<double> Eb2(num_species, 3.0);
  std::vector<double> layer_thicknesses(num_layers, 0.0);

  // Use average composition across layers weighted by thickness
  double total_thick = 0.0;
  for (int il = 0; il < num_layers; il++) {
    layer_thicknesses[il] = target[il].thickness;
    total_thick += target[il].thickness;
  }

  // Fill species arrays from first (surface) layer
  const BCALayer &surf = target[0];
  int ns = std::min(num_species, static_cast<int>(surf.Z.size()));
  for (int is = 0; is < ns; is++) {
    Z2[is] = static_cast<double>(surf.Z[is]);
    m2[is] = surf.M[is];
    n2[is] = surf.density * surf.fraction[is];  // partial number density
    Ec2[is] = (is < static_cast<int>(surf.Ec.size())) ? surf.Ec[is] : 1.0;
    Es2[is] = (is < static_cast<int>(surf.Es.size())) ? surf.Es[is] : 2.0;
    Eb2[is] = (is < static_cast<int>(surf.Eb.size())) ? surf.Eb[is] : 3.0;
  }

  // Projectile parameters
  double Ec1 = 1.0;   // cutoff energy for projectile (eV)
  double Es1 = 1.0;   // surface binding for projectile (eV)

  // Output buffers
  int num_reflected = 0;
  int num_sputtered = 0;
  std::vector<double> reflected_data(MAX_OUTPUT_PARTICLES * 9, 0.0);
  std::vector<double> sputtered_data(MAX_OUTPUT_PARTICLES * 9, 0.0);
  double implant_depth = 0.0;

  // Call RustBCA
  bca_fn(
      ux, uy, uz,
      proj_E,
      static_cast<double>(proj_Z),
      proj_M,
      Ec1, Es1,
      num_species,
      Z2.data(), m2.data(), n2.data(),
      Ec2.data(), Es2.data(), Eb2.data(),
      num_layers,
      layer_thicknesses.data(),
      &num_reflected, reflected_data.data(),
      &num_sputtered, sputtered_data.data(),
      &implant_depth
  );

  // Parse reflected particles
  // Each particle: Z, M, E, x, y, z, ux, uy, uz (9 values)
  int nref = std::min(num_reflected, MAX_OUTPUT_PARTICLES);
  for (int i = 0; i < nref; i++) {
    BCAResult::Particle p;
    int base = i * 9;
    p.Z = static_cast<int>(reflected_data[base + 0]);
    p.M = reflected_data[base + 1];
    p.E = reflected_data[base + 2];
    // positions at base+3,4,5 (not used)
    p.ux = reflected_data[base + 6];
    p.uy = reflected_data[base + 7];
    p.uz = reflected_data[base + 8];
    result.reflected.push_back(p);
  }

  // Parse sputtered particles
  int nsput = std::min(num_sputtered, MAX_OUTPUT_PARTICLES);
  for (int i = 0; i < nsput; i++) {
    BCAResult::Particle p;
    int base = i * 9;
    p.Z = static_cast<int>(sputtered_data[base + 0]);
    p.M = sputtered_data[base + 1];
    p.E = sputtered_data[base + 2];
    p.ux = sputtered_data[base + 6];
    p.uy = sputtered_data[base + 7];
    p.uz = sputtered_data[base + 8];
    result.sputtered.push_back(p);
  }

  result.implant_depth = implant_depth;
  result.was_implanted = (num_reflected == 0 && implant_depth > 0.0);

  return result;
}
