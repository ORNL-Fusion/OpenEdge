/* ----------------------------------------------------------------------
    OpenEdge: Multi-layer surface state tracking
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - Austin Nichols (ORNL, nicholsa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    Tracks composition depth profiles on surface elements as a stack
    of thin layers.  Each layer has a thickness, bulk density, and
    per-species fractional composition.  Used by surf_react_surface_pwi to
    update the surface state after BCA ion-surface interactions.

    Key operations:
      - add_implanted(): deposit atoms at a given depth
      - compact_layers(): merge thin layers, cap at max_layers
      - get_surface_composition(): depth-averaged composition query
------------------------------------------------------------------------- */

#ifndef SPARTA_SURF_STATE_MULTILAYER_H
#define SPARTA_SURF_STATE_MULTILAYER_H

#include <vector>
#include <string>

namespace SPARTA_NS {

// Maximum number of layers per surface element (bounds memory)
static const int MAX_LAYERS_DEFAULT = 50;

// Minimum layer thickness before merging [m] (0.5 Angstrom)
static const double MIN_LAYER_THICKNESS = 0.5e-10;

struct SurfaceLayer {
  double thickness;                  // [m]
  double density;                    // [atoms/m^3]
  std::vector<double> composition;   // fractional per species, sum = 1.0

  SurfaceLayer() : thickness(0.0), density(0.0) {}
  SurfaceLayer(double thick, double dens, int nspecies)
    : thickness(thick), density(dens), composition(nspecies, 0.0) {}
};

struct SurfaceElementState {
  std::vector<SurfaceLayer> layers;  // surface (index 0) -> bulk ordering
  int nspecies;                      // number of tracked species
  int max_layers;                    // maximum allowed layers
  double total_fluence;              // cumulative incident fluence [ions/m^2]
  // minimum stratum thickness before compaction merges it [m].
  // Default 0.5 A (atomic scale); compressed-timescale TEST configs with
  // artificial sub-Angstrom layers must lower it (deck: strata K minthick X)
  double min_thickness;

  SurfaceElementState()
    : nspecies(0), max_layers(MAX_LAYERS_DEFAULT), total_fluence(0.0),
      min_thickness(MIN_LAYER_THICKNESS) {}

  SurfaceElementState(int nspec, int maxlyr = MAX_LAYERS_DEFAULT)
    : nspecies(nspec), max_layers(maxlyr), total_fluence(0.0),
      min_thickness(MIN_LAYER_THICKNESS) {}

  // Initialize with a uniform substrate
  void init_substrate(double thickness, double density,
                      int substrate_species);

  // Grow the surface with newly deposited material (redeposition /
  // boronization credits). ALL SI: amount [atoms/m^2]; density of the
  // new stratum [atoms/m^3] (<=0 -> reuse surface density, else solid
  // default). Unlike add_implanted(), thickness INCREASES.
  void deposit(int species, double amount, double density = 0.0);

  // Preferential sputtering: remove `amount` [atoms/m^2] of one
  // species from the top of the stack. Returns the amount actually
  // removed (bounded by surface availability -- never goes negative,
  // which retires the one-sync overshoot at the stack level).
  double erode_species(int species, double amount);

  // Add implanted atoms at a given depth (SI)
  // species: species index (0-based)
  // depth: implantation depth [m] from surface
  // amount: atoms per unit area [atoms/m^2]
  void add_implanted(int species, double depth, double amount);

  // Erode material from the surface
  // thickness: amount to remove [m]

  // Merge thin layers and cap at max_layers
  void compact_layers();

  // Get depth-averaged composition over a sampling depth [m]
  // Returns vector of fractional composition per species
  std::vector<double> get_surface_composition(double depth) const;

  // Get total thickness of all layers

  // Get number of atoms in top layer (for sputtering decisions)
  double surface_density() const;

  // Serialize/deserialize for MPI communication and restart
  // Pack into a flat double array, return number of doubles written
  int pack(double *buf) const;

  // Unpack from flat double array, return number of doubles read
  int unpack(const double *buf);

  // Return the buffer size needed for pack()
};


}

#endif
