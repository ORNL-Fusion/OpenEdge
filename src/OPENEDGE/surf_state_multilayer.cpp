/* ----------------------------------------------------------------------
    OpenEdge: Multi-layer surface state tracking
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - Austin Nichols (ORNL, nicholsa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    See surf_state_multilayer.h for documentation.
------------------------------------------------------------------------- */

#include "surf_state_multilayer.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

using namespace SPARTA_NS;

/* ======================================================================
   SurfaceElementState methods
   ====================================================================== */

void SurfaceElementState::init_substrate(double thickness, double density,
                                          int substrate_species)
{
  layers.clear();
  SurfaceLayer lyr(thickness, density, nspecies);
  if (substrate_species >= 0 && substrate_species < nspecies)
    lyr.composition[substrate_species] = 1.0;
  layers.push_back(lyr);
  total_fluence = 0.0;
}

/* ---------------------------------------------------------------------- */

void SurfaceElementState::add_implanted(int species, double depth, double amount)
{
  if (species < 0 || species >= nspecies) return;
  if (amount <= 0.0) return;

  // Find the layer at the given depth
  double cumulative = 0.0;
  int target_layer = -1;

  for (int i = 0; i < static_cast<int>(layers.size()); i++) {
    cumulative += layers[i].thickness;
    if (depth <= cumulative) {
      target_layer = i;
      break;
    }
  }

  // If depth exceeds total thickness, add to deepest layer
  if (target_layer < 0) {
    if (layers.empty()) {
      // Create a new layer at the implant location
      SurfaceLayer lyr(std::max(depth, 1.0), 6.33e22, nspecies);
      lyr.composition[species] = 1.0;
      layers.push_back(lyr);
      return;
    }
    target_layer = static_cast<int>(layers.size()) - 1;
  }

  SurfaceLayer &lyr = layers[target_layer];

  // Mix the implanted species into the layer composition
  // amount is atoms/cm^2, layer has density * thickness atoms/cm^2
  double layer_areal = lyr.density * lyr.thickness * 1.0e-8;  // atoms/cm^2
  double total_areal = layer_areal + amount;

  if (total_areal > 0.0) {
    for (int s = 0; s < nspecies; s++) {
      lyr.composition[s] = lyr.composition[s] * layer_areal / total_areal;
    }
    lyr.composition[species] += amount / total_areal;

    // Normalize
    double sum = 0.0;
    for (int s = 0; s < nspecies; s++) sum += lyr.composition[s];
    if (sum > 0.0) {
      for (int s = 0; s < nspecies; s++) lyr.composition[s] /= sum;
    }
  }
}

/* ---------------------------------------------------------------------- */

void SurfaceElementState::erode_surface(double thickness)
{
  if (thickness <= 0.0) return;

  double remaining = thickness;
  while (remaining > 0.0 && !layers.empty()) {
    SurfaceLayer &top = layers.front();
    if (top.thickness <= remaining) {
      remaining -= top.thickness;
      layers.erase(layers.begin());
    } else {
      top.thickness -= remaining;
      remaining = 0.0;
    }
  }
}

/* ---------------------------------------------------------------------- */

void SurfaceElementState::compact_layers()
{
  // Merge thin layers
  int i = 0;
  while (i < static_cast<int>(layers.size()) - 1) {
    if (layers[i].thickness < MIN_LAYER_THICKNESS) {
      SurfaceLayer &a = layers[i];
      SurfaceLayer &b = layers[i + 1];

      double area_a = a.density * a.thickness;
      double area_b = b.density * b.thickness;
      double total = area_a + area_b;

      if (total > 0.0) {
        for (int s = 0; s < nspecies; s++) {
          b.composition[s] = (a.composition[s] * area_a +
                              b.composition[s] * area_b) / total;
        }
        b.thickness = a.thickness + b.thickness;
        b.density = total / b.thickness;
      }

      layers.erase(layers.begin() + i);
    } else {
      i++;
    }
  }

  // Cap at max_layers by merging from the bottom
  while (static_cast<int>(layers.size()) > max_layers && layers.size() > 1) {
    int last = static_cast<int>(layers.size()) - 1;
    SurfaceLayer &a = layers[last - 1];
    SurfaceLayer &b = layers[last];

    double area_a = a.density * a.thickness;
    double area_b = b.density * b.thickness;
    double total = area_a + area_b;

    if (total > 0.0) {
      for (int s = 0; s < nspecies; s++) {
        a.composition[s] = (a.composition[s] * area_a +
                            b.composition[s] * area_b) / total;
      }
      a.thickness += b.thickness;
      a.density = total / a.thickness;
    }

    layers.pop_back();
  }
}

/* ---------------------------------------------------------------------- */

std::vector<double> SurfaceElementState::get_surface_composition(double depth) const
{
  std::vector<double> comp(nspecies, 0.0);
  if (layers.empty()) return comp;

  double cumulative = 0.0;
  double total_weight = 0.0;

  for (const auto &lyr : layers) {
    double lo = cumulative;
    double hi = cumulative + lyr.thickness;
    cumulative = hi;

    if (lo >= depth) break;

    double overlap = std::min(hi, depth) - lo;
    if (overlap <= 0.0) break;

    double weight = overlap * lyr.density;
    for (int s = 0; s < nspecies; s++) {
      comp[s] += lyr.composition[s] * weight;
    }
    total_weight += weight;
  }

  if (total_weight > 0.0) {
    for (int s = 0; s < nspecies; s++) comp[s] /= total_weight;
  }

  return comp;
}

/* ---------------------------------------------------------------------- */

double SurfaceElementState::total_thickness() const
{
  double t = 0.0;
  for (const auto &lyr : layers) t += lyr.thickness;
  return t;
}

/* ---------------------------------------------------------------------- */

double SurfaceElementState::surface_density() const
{
  if (layers.empty()) return 0.0;
  return layers.front().density;
}

/* ---------------------------------------------------------------------- */

int SurfaceElementState::pack_size() const
{
  // nspecies, max_layers, total_fluence, nlayers
  // per layer: thickness, density, composition[nspecies]
  int nlayers = static_cast<int>(layers.size());
  return 4 + nlayers * (2 + nspecies);
}

int SurfaceElementState::pack(double *buf) const
{
  int m = 0;
  buf[m++] = static_cast<double>(nspecies);
  buf[m++] = static_cast<double>(max_layers);
  buf[m++] = total_fluence;
  int nlayers = static_cast<int>(layers.size());
  buf[m++] = static_cast<double>(nlayers);

  for (const auto &lyr : layers) {
    buf[m++] = lyr.thickness;
    buf[m++] = lyr.density;
    for (int s = 0; s < nspecies; s++)
      buf[m++] = lyr.composition[s];
  }

  return m;
}

int SurfaceElementState::unpack(const double *buf)
{
  int m = 0;
  nspecies = static_cast<int>(buf[m++]);
  max_layers = static_cast<int>(buf[m++]);
  total_fluence = buf[m++];
  int nlayers = static_cast<int>(buf[m++]);

  layers.resize(nlayers);
  for (int i = 0; i < nlayers; i++) {
    layers[i].thickness = buf[m++];
    layers[i].density = buf[m++];
    layers[i].composition.resize(nspecies);
    for (int s = 0; s < nspecies; s++)
      layers[i].composition[s] = buf[m++];
  }

  return m;
}

/* ======================================================================
   SurfStateMultilayer methods
   ====================================================================== */

SurfStateMultilayer::SurfStateMultilayer(int nspec, int maxlyr)
  : nspecies(nspec), max_layers(maxlyr)
{
}

SurfStateMultilayer::~SurfStateMultilayer()
{
}

void SurfStateMultilayer::allocate(int nsurf)
{
  states.resize(nsurf, SurfaceElementState(nspecies, max_layers));
}

void SurfStateMultilayer::init_substrate(double thickness, double density,
                                          int substrate_species)
{
  for (auto &st : states) {
    st.init_substrate(thickness, density, substrate_species);
  }
}

void SurfStateMultilayer::set_species_names(const std::vector<std::string> &names)
{
  species_names = names;
}

void SurfStateMultilayer::compact_all()
{
  for (auto &st : states) {
    st.compact_layers();
  }
}
