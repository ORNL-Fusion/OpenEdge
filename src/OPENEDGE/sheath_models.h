/* ----------------------------------------------------------------------
   Borodkina sheath model, Russian Phys. J. 58 (2015) 438-445 and
   Contrib. Plasma Phys. 56 (2016) 640-645.
   Alpha is measured from the wall normal. The 2015 closure is recovered
   at Ti=Te and k=2; OpenEdge uses its Ti/Te generalization and k=3.
------------------------------------------------------------------------- */

#ifndef OPENEDGE_SHEATH_MODELS_H
#define OPENEDGE_SHEATH_MODELS_H

namespace SPARTA_NS {
namespace SheathModels {

struct ChoduraMetrics {
  double bdotn = 0.0;
  double alpha_deg = 90.0;
  double mach_par = 0.0;
  double mach_n = 0.0;
  double u_n = 0.0;
};

ChoduraMetrics chodura_metrics(double upar_ms, double cs_ms,
                               const double b[3], const double n[3]);

struct SheathEmagCoeffs {
  double lambdaD_m = 0.0;
  double rho_i_m = 0.0;
  double lmps_m = 0.0;
  double mps_end_m = 0.0;
  double phi_total_eV = 0.0;
  double phi_ds_eV = 0.0;
  double phi_mps_eV = 0.0;
  double te_eV = 0.0;
  double lambda_w = 0.0;
  double lambda_mps = 0.0;
  double a_ds = 0.0;
  double q_ds = 0.0;
  double xi_mps = 0.0;
  double inv_lambdaD = 0.0;
  double inv_mps = 0.0;
  double fd = 0.0;
  double alpha_star_deg = 0.0;
  int pure_mps = 0;
};

SheathEmagCoeffs sheath_prepare(double te_eV, double ti_eV, double ne_m3,
                                double bmag_T, double alpha_deg, double mD_amu,
                                double pot_mult = 0.0);
SheathEmagCoeffs sheath_prepare_borodkina(double te_eV, double ti_eV,
                                          double ne_m3, double bmag_T,
                                          double alpha_normal_deg,
                                          double mD_amu,
                                          double pot_mult = 0.0);

double sheath_emag_at_distance(const SheathEmagCoeffs &c, double dist_m);
double sheath_phi_at_distance(const SheathEmagCoeffs &c, double dist_m);

struct SheathProfile {
  double esheath_eV = 0.0;
  double emag_vpm = 0.0;
  double lambdaD_m = 0.0;
  double rho_i_m = 0.0;
  double extent_m = 0.0;
  double phi_total_eV = 0.0;
  double phi_ds_eV = 0.0;
};

SheathProfile sheath_at_distance(double dist_m, double te_eV, double ti_eV,
                                 double ne_m3, double bmag_T, double alpha_deg,
                                 double mD_amu, double pot_mult = 0.0);

}  // namespace SheathModels
}  // namespace SPARTA_NS

#endif
