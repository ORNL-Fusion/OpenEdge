/* ----------------------------------------------------------------------
   OpenEdge: fix core/transport — 1-D flux-surface-averaged impurity
   transport behind the core cut. Markers of the charge-state chain that
   hit the core surf group are absorbed and tallied per state; every
   nevery steps the 1-D solver (CoreTransport1D, ADAS ionization and
   recombination, D and v profiles) is advanced and the flux leaving the
   1-D domain is re-emitted per state from the same surf group, with the
   particle/tag transit flag set.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(core/transport,FixCoreTransport)

#else

#ifndef SPARTA_FIX_CORE_TRANSPORT_H
#define SPARTA_FIX_CORE_TRANSPORT_H

#include <string>
#include <vector>
#include "fix.h"
#include "core_transport_geometry.h"
#include "core_transport_reservoir.h"

namespace SPARTA_NS {

class FixCoreTransport : public Fix {
 public:
  FixCoreTransport(class SPARTA *, int, char **);
  ~FixCoreTransport();
  int setmask();
  void init();
  void start_of_step();
  void end_of_step();
  void post_run();
  double compute_scalar();        // core W content (atoms, full torus)
  double compute_vector(int);     // 0 Gamma_in 1 Gamma_out (atoms/s, torus) 2 Prad (W) 3 emitted (atoms, this window)
                                  // 4 pending exchange inventory (atoms, torus)

  // called by Update::move for a chain marker hitting the core group
  void absorb(int ipart, int isurf);

  int groupbit;                   // core surf group

 private:
  char *pd_id;
  class FixBackground *pd;
  int imix, nion;
  std::vector<int> spec_of_z;     // species index per charge state
  std::vector<int> z_of_spec;     // charge state per species, -1 = not in chain
  std::string element;
  double mass;                    // kg

  double psi_n;                   // interface psi_N (1-D domain edge)
  int ngrid;                      // 1-D cells
  double dperp, vpinch;           // constants (m2/s, m/s) unless profiles given
  std::vector<double> dprof_x, dprof_y, vprof_x, vprof_y;   // (rho_N, value)
  // coefficients effective (default): dperp/vpinch are the r_vol operator's
  // coefficients. physical: D_perp and the pinch speed along the surface
  // normal, mapped with g2 = <|grad r_vol|^2> and g1 = <|grad r_vol|>
  int coef_physical;
  char *geometry_file;
  std::vector<double> geo_rho, geo_rv, geo_g1, geo_g2;
  double geo_axis_rho;            // factors below this rho_N are extrapolated (-1 unknown)
  std::vector<double> diff_in, conv_in, g1cell, g2cell;   // per cell, before mapping
  int solver_split;               // 0 substep (reference), 1 implicit split
  double split_max_dt;            // largest internal step of the split solver (s)
  int split_steps(double dt_win) const;

  // auxiliary-layer coupling (replaces absorb/deposit/re-emit once validated):
  // closed 1-D boundary, a core layer of width layer_h and a guiding-centre
  // marker layer of the same width outside the cut exchange through CoreExchange
  int coupling_aux;
  int reflect_groupbit;            // extra surf groups that reflect guiding centres
  double layer_h, marker_weight;   // m (r_vol); weight of created markers
  int marker_auto;                 // 1 = created markers take the mean weight of the layer markers
  std::vector<double> wmark;       // auto: per state, last mean weight seen in the layer (0 = none yet)
  double wmark_all;                // auto: the same over all states
  int layer_cells;                 // core cells in the core layer
  double psi_layer, vol_core_layer, vol_marker_layer;   // outer psi_N, torus volumes
  double box_r[2], box_z[2];       // (R,Z) box enclosing the marker layer
  std::vector<double> vcum_out; double vcum_out_cut;
  class CoreExchange *ex;
  bigint window_index;
  double pending_total() const;
  void build_layer();
  double psi_of_volume(double v) const;
  void exchange_window(double dt_win);
  void create_gc_marker(int z, const double *x, double u[5], int id, double weight);
  bool in_core_region(double R, double Z) const;
  bool inside_cut(const double *x) const;        // SPARTA position, 2-D or 3-D
  // 3-D cut: triangles of a surface of revolution, whole torus or a wedge
  // between periodic caps (the cap model sets the angles)
  int cut_full;                    // 1 = full torus
  double cut_phi0, cut_dphi;       // wedge start and angle (rad)
  std::vector<std::vector<int> > cut_bins;       // cut triangles per toroidal bin
  double cut_angle(const double *x) const;       // angle from cut_phi0, in [0,cut_dphi) for points in the wedge
  void build_cut3d(double &enclosed, double &sliver);
  void read_geometry();
  int nlaunch;                    // markers per window (all states)
  int position_mode;              // 0 uniform over the surface, 1 entry positions
  double fraction;                // torus fraction of the MC domain
  double depth;                   // deposition depth of the influx (m in rvol), 0 = outer cell
  double dlen_in;                 // outer decay length (m), <= 0 = half a cell
  char *profiles_file, *output_file, *state_file, *read_state_file, *export_file;
  int output_every, coupling_every, window_steps;
  double elapsed, last_dt_win, physical_time;
  bool output_fresh;

  // 1-D state (SI, torus volumes)
  class CoreTransport1D *ct;
  int built;
  double r_axis, rvol_x, dr, dlen;
  std::vector<double> rr, vol, psin, ne, te, ti;     // per cell
  std::vector<double> diff, conv, srate, rrate, n, src;    // [nion][ir]
  std::vector<double> plt, prb;                     // [nion][ir] W m3, may be empty
  std::vector<double> depo;                         // deposition weight per cell
  std::vector<double> vcum;                         // cumulative volume vs psi_N bins
  std::vector<CoreFluxPoint> flood_cells;                     // equilibrium cells inside the contour

  // tallies
  std::vector<double> gin_local, gin, gout;         // atoms (window), atoms/s torus
  double gin_total, gout_total, emitted_total, emitted_window;
  double gin_rate, gout_rate, prad;

  // Persistent independent weighted samples of all absorbed entries.
  std::vector<CoreEntry> pool;
  int pool_cap;

  // emission geometry
  std::vector<int> core_elems;
  std::vector<double> core_cdf;
  int flip_normal;
  class RanKnuth *rng;            // same seed on every rank: identical draws

  int pweight_index, tag_index;

  void advance_window();
  void build_grid();
  void build_profiles();
  void build_rates();
  void build_surface();
  void read_state();
  void write_state();
  void write_export();
  void write_output();
  void emit_window(double dt_win);
  void emit_marker(int z, int isurf, const double *xs, double weight);
  void surf_point(int isurf, double *x, double *nrm);
  double profile_at(const std::vector<double> &x, const std::vector<double> &y,
                    double rho, double fallback) const;
  double psi_rz(double R, double Z) const;
  void point_rz(const double *x, double &R, double &Z) const;
};

}

#endif
#endif
