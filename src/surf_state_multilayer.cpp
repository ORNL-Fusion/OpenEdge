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
      SurfaceLayer lyr(std::max(depth, 1.0e-10), 6.33e28, nspecies);
      lyr.composition[species] = 1.0;
      layers.push_back(lyr);
      return;
    }
    target_layer = static_cast<int>(layers.size()) - 1;
  }

  SurfaceLayer &lyr = layers[target_layer];

  // Mix the implanted species into the layer composition
  // SI: areal = density * thickness, no conversion factors
  double layer_areal = lyr.density * lyr.thickness;  // atoms/m^2
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

void SurfaceElementState::deposit(int species, double amount, double density)
{
  if (species < 0 || species >= nspecies || amount <= 0.0) return;
  if (density <= 0.0)
    density = layers.empty() ? 6.33e28 : layers.front().density;

  double thick = amount / density;            // [m]

  // merge into the top stratum if it is thin or already dominated by
  // this species; otherwise push a new stratum (compositional memory)
  if (!layers.empty()) {
    SurfaceLayer &top = layers.front();
    int dom = static_cast<int>(
        std::max_element(top.composition.begin(), top.composition.end()) -
        top.composition.begin());
    if (top.thickness < min_thickness || dom == species) {
      double a_top = top.density * top.thickness;   // atoms/m^2
      double total = a_top + amount;
      if (total > 0.0) {
        for (int s = 0; s < nspecies; s++)
          top.composition[s] *= a_top / total;
        top.composition[species] += amount / total;
        top.thickness += thick;
        top.density = total / top.thickness;
      }
      return;
    }
  }
  SurfaceLayer lyr(thick, density, nspecies);
  lyr.composition[species] = 1.0;
  layers.insert(layers.begin(), lyr);
}

/* ---------------------------------------------------------------------- */

double SurfaceElementState::erode_species(int species, double amount)
{
  if (species < 0 || species >= nspecies || amount <= 0.0) return 0.0;

  double removed = 0.0;
  while (removed < amount && !layers.empty()) {
    SurfaceLayer &top = layers.front();
    double a_layer = top.density * top.thickness;   // atoms/m^2
    if (a_layer <= 0.0) { layers.erase(layers.begin()); continue; }
    double a_sp = a_layer * top.composition[species];
    if (a_sp <= 0.0) break;      // species not exposed at the surface

    double take = std::min(a_sp, amount - removed);
    double a_new = a_layer - take;
    if (a_new <= 1.0e-30) {      // stratum fully consumed
      removed += take;
      layers.erase(layers.begin());
      continue;
    }
    // remove `take` of `species`; other species' areal content unchanged
    for (int s = 0; s < nspecies; s++)
      top.composition[s] *= a_layer / a_new;
    top.composition[species] = (a_sp - take) / a_new;
    top.thickness = a_new / top.density;
    removed += take;
  }
  return removed;
}

/* ---------------------------------------------------------------------- */

void SurfaceElementState::compact_layers()
{
  // Merge thin BURIED layers only (start at 1): the TOP stratum is the
  // growing surface and must accumulate from zero toward min_thickness --
  // compacting it dissolves continuous fine deposition into the substrate
  // and no stratum can ever nucleate.
  int i = 1;
  while (i < static_cast<int>(layers.size()) - 1) {
    if (layers[i].thickness < min_thickness) {
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

double SurfaceElementState::surface_density() const
{
  if (layers.empty()) return 0.0;
  return layers.front().density;
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

