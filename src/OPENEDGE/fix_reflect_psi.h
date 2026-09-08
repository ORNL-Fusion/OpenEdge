/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix reflect/psi: Reflect particles that cross a psi_norm boundary.

    At END_OF_STEP, evaluates normalized poloidal flux psi_n(R,Z) at
    each particle position using an equilibrium file.  If
    psi_n < psi_threshold (particle has entered the core), the particle
    is reflected back to its previous position and its velocity is
    reversed along the radial direction.

    psi_n = (psi - psi_axis) / (psib - psi_axis)
      0 at magnetic axis, 1 at separatrix, >1 in SOL

    Supports both G-EQDSK and SOLPS .equ equilibrium formats.

    Syntax:
      fix ID reflect/psi {equ PATH | background FIXID} [psi_norm VALUE]
          [action reflect|absorb] [mixture MIXTURE_ID]

    Data source:
      - equ PATH          : read psi map from a SOLPS .equ file (GEQDSK-style)
      - background FIXID : read psi map from a fix background (from
                            plasma.h5 /psi + /psicore + /psisep). The
                            default threshold is 0, i.e. anything on the
                            core side of /psicore triggers the action.

    The check runs every move step inside the particle mover; there is no
    Nevery throttle (it's a boundary condition, not a diagnostic).

    Examples:
      fix fcore reflect/psi equ input/g174310.03500_153.X4.equ psi_norm 0.926
      fix fcore reflect/psi background pd action absorb
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(reflect/psi,FixReflectPsi)

#else

#ifndef SPARTA_FIX_REFLECT_PSI_H
#define SPARTA_FIX_REFLECT_PSI_H

#include "fix.h"

#include <string>
#include <vector>

namespace SPARTA_NS {

class FixReflectPsi : public Fix {
 public:
  FixReflectPsi(class SPARTA *, int, char **);
  ~FixReflectPsi();
  int  setmask();
  void init();
  double compute_vector(int) override;

  // Called by the particle mover immediately before an absorbed particle is
  // discarded. The tally is species resolved and uses physical marker weight.
  void tally_absorb(int ispecies, int iparticle);

  // Geometry helpers used by the mover for exact contour reflection.
  double psi_norm_at_sparta(const double xyz[3]) const;
  bool segment_crossing(const double x0[3], const double x1[3],
                        double &fraction, double normal[3]) const;

  enum { PSI_ACTION_REFLECT, PSI_ACTION_ABSORB };

  // Device mover (Kokkos) access: read-only psi map and a bulk absorption
  // tally folded from per-species device counters after each move pass.
  const std::vector<double> &psi_r_grid() const { return r_grid_; }
  const std::vector<double> &psi_z_grid() const { return z_grid_; }
  const std::vector<double> &psi_map() const { return psirz_; }
  double psi_axis_value() const { return psi_axis_; }
  double psi_boundary_value() const { return psib_; }
  void tally_absorb_bulk(int ispecies, double nevents, double weight);

 protected:
  int action_;                 // PSI_ACTION_REFLECT or PSI_ACTION_ABSORB
  double psi_threshold_;     // normalized psi boundary
  int imix_;                  // mixture restriction, -1 = every species

  // Global absorption ledger. For every selected species/group the vector
  // exposes: simulation events, cumulative physical particles, and mean
  // physical removal rate [s^-1] since this fix was initialized.
  int nrows_;
  int pweight_index_, pweight_ewhich_;
  bigint start_step_, reduced_step_;
  std::vector<double> absorbed_events_local_;
  std::vector<double> absorbed_physical_local_;
  std::vector<double> absorbed_events_global_;
  std::vector<double> absorbed_physical_global_;

  // Equilibrium data
  int nw_, nh_;
  double psi_axis_, psib_;
  std::vector<double> r_grid_, z_grid_;
  std::vector<double> psirz_;   // [nh * nw], row-major [z][r]

  // Interpolation
  double psi_norm_at_point(double R, double Z) const;
  double psi_norm_gradient(double R, double Z,
                           double &grad_r, double &grad_z) const;

  // File readers
  void read_equ_file(const std::string &path);
  void load_from_background(const std::string &fix_id);
  int row_for_species(int ispecies) const;
  void reduce_tallies();
};

}  // namespace SPARTA_NS

#endif
#endif
