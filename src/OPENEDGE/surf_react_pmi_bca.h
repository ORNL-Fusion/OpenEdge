/* ----------------------------------------------------------------------
    OpenEdge: BCA-coupled PMI surface reactions
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - Austin Nichols (ORNL, nicholsa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    Extends surf_react_pmi with RustBCA binary-collision approximation
    for ion-surface interactions, coupled with multi-layer surface state
    tracking for dynamic codeposit formation modeling.

    Instead of tabulated RN/RE/Y, calls RustBCA for each incident
    particle to get the full collision cascade result: reflected
    particles, sputtered atoms, and implantation depth.  The surface
    composition evolves self-consistently through erosion and deposition.

    Input script syntax:
      surf_react ID pmi/bca file reaction_file.surf \
          bca librustbca.so nlayers 20 layer_thickness 5.0 \
          substrate W density 6.33e22

    Options:
      bca <path>           - path to librustbca.so shared library
      nlayers <N>          - max number of layers per element (default 20)
      layer_thickness <A>  - initial substrate layer thickness (Angstroms)
      substrate <species>  - substrate species name
      density <n>          - substrate number density (atoms/cm^3)
      surrogate <path>     - optional surrogate model weights file
      surrogate_threshold <val> - uncertainty threshold for BCA fallback
------------------------------------------------------------------------- */

#ifdef SURF_REACT_CLASS

SurfReactStyle(pmi/bca,SurfReactPMIBCA)

#else

#ifndef SPARTA_SURF_REACT_PMI_BCA_H
#define SPARTA_SURF_REACT_PMI_BCA_H

#include "surf_react_pmi.h"
#include "rustbca_interface.h"
#include "surf_state_multilayer.h"
#include <string>
#include <vector>
#include <memory>

namespace SPARTA_NS {

class SurfReactPMIBCA : public SurfReactPMI {
 public:
  SurfReactPMIBCA(class SPARTA *, int, char **);
  ~SurfReactPMIBCA();
  void init();
  int react(Particle::OnePart *&, int, double *, Particle::OnePart *&, int &);

 private:
  // RustBCA interface
  RustBCAInterface bca;
  std::string bca_lib_path;

  // Multi-layer surface state
  std::unique_ptr<SurfStateMultilayer> surf_state;

  // Configuration
  int max_nlayers;
  double init_layer_thickness;   // Angstroms
  double substrate_density;      // atoms/cm^3
  int substrate_species_idx;     // index in SPARTA species list
  std::string substrate_name;

  // BCA species mapping
  struct SpeciesInfo {
    int Z;                  // atomic number
    double M;               // mass (amu)
    double Eb;              // bulk binding energy (eV)
    double Es;              // surface binding energy (eV)
    double Ec;              // cutoff energy (eV)
  };
  std::vector<SpeciesInfo> species_info;

  // Optional surrogate model
  int use_surrogate;
  std::string surrogate_path;
  double surrogate_threshold;

  // Statistics
  bigint n_bca_calls;
  bigint n_surrogate_calls;
  bigint n_implanted;
  bigint n_reflected_bca;
  bigint n_sputtered_bca;

  // Internal methods
  void setup_species_info();
  int find_sparta_species(int Z, double M) const;
  std::vector<BCALayer> build_target_layers(int isurf) const;
  void apply_bca_result(int isurf, const BCAResult &result,
                        int proj_species);

  // Species database (Z, M, Es, Eb, Ec for common fusion-relevant species)
  static SpeciesInfo lookup_species(const std::string &name);
};

}

#endif
#endif
