/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef SURF_REACT_CLASS

SurfReactStyle(pmi,SurfReactPMI)

#else

#ifndef SPARTA_SURF_REACT_PMI_H
#define SPARTA_SURF_REACT_PMI_H

#include "surf_react.h"
#include "eckstein_sputter.h"
#include "eirene_trim.h"
#include <cstdio>
#include <map>
#include <vector>
#include <string>

namespace SPARTA_NS {

// Raw storage for one EIRENE TRIM reflection table.  Owns the memory;
// pointer snapshots (TrimView) live in eirene_trim.h.
struct TrimTableData {
  std::string name;
  double Z1, M1, Z2, M2;
  std::vector<double> E_grid;        // [nE]
  std::vector<double> theta_grid;    // [nW]
  std::vector<double> raar;          // [nR]
  std::vector<double> R_N;           // [nE*nW]
  std::vector<double> Eout_q;        // [nE*nW*nR]
  std::vector<double> Eout_min;      // [nE*nW]
  std::vector<double> Eout_max;      // [nE*nW]
  std::vector<double> cos_polar_q;   // [nE*nW*nR*nR]
  std::vector<double> cos_azim_q;    // [nE*nW*nR*nR*nR]

  EireneTrim::TrimView view() const {
    EireneTrim::TrimView v;
    v.E           = E_grid.data();
    v.theta_deg   = theta_grid.data();
    v.raar        = raar.data();
    v.R_N         = R_N.data();
    v.Eout_q      = Eout_q.data();
    v.Eout_min    = Eout_min.data();
    v.Eout_max    = Eout_max.data();
    v.cos_polar_q = cos_polar_q.data();
    v.cos_azim_q  = cos_azim_q.data();
    return v;
  }
};

class SurfReactPMI : public SurfReact {
 public:
  SurfReactPMI(class SPARTA *, int, char **);
  SurfReactPMI(class SPARTA *sparta) : SurfReact(sparta) {} // needed for Kokkos
  virtual ~SurfReactPMI();
  virtual void init();
  int react(Particle::OnePart *&, int, double *, Particle::OnePart *&, int &);
  char *reactionID(int);
  double reaction_coeff(int);
  int match_reactant(char *, int);
  int match_product(char *, int);

  // reaction info, as read from file

  struct OneReaction {
    int active;                    // 1 if reaction is active
    int type;                      // reaction type = REFLECT, SPUTTER, ABSORB
    int style;                     // reaction style = SIMPLE
    int ncoeff;                    // # of numerical coeffs
    int nreactant,nproduct;        // # of reactants and products
    char **id_reactants,**id_products;  // species IDs of reactants/products
    int *reactants,*products;      // species indices of reactants/products
    double *coeff;                 // numerical coeffs for reaction
    char *id;                      // reaction ID (formula)
  };

  int last_outcome;                // 1=reflect, 2=sputter, 3=absorb

 protected:
  class RanKnuth *random;     // RNG for reaction probabilities

  int mode;                        // 0=constant, 1=file

  // constant-mode parameters
  double const_RN, const_RE, const_Y, const_Ebind;

  // file-mode HDF5 tables
  int nE, nA;
  std::vector<double> E_axis, A_axis;
  std::vector<double> RN_table, RE_table, spyld_table;
  double Ebind;

  // optional tabulated CDF distributions
  int nb_cdf;
  int has_cdf_R, has_cdf_Y;
  std::vector<double> edist_R, adist_R, edist_Y, adist_Y;

  // reaction list
  OneReaction *rlist;
  int nlist_prob;                  // # of reactions read from file
  int maxlist_prob;                // max # of reactions in rlist

  // per-species reaction role lookup (replaces generic reaction list)
  struct SpeciesReactions {
    int ireflect;                  // index into rlist, or -1
    int isputter;                  // index into rlist, or -1
    int iabsorb;                   // index into rlist, or -1
  };
  SpeciesReactions *species_reactions;

  std::string h5_path;

  // EIRENE TRIM reflection table storage (name-indexed).  Populated at
  // parse time when a reaction is declared with style ECKSTEIN_TRIM.
  std::string trim_dir;
  std::vector<TrimTableData> trim_tables;
  std::map<std::string,int> trim_index;

  // load trim table from trim_dir/<name>.h5 if not already cached.
  // Returns index into trim_tables, or -1 on failure.
  int load_or_get_trim_table(const char *name);

  // impact logging
  int logflag;
  FILE *logfp;
  char *logfile_base;
  int me;
  void open_logfile();
  void close_logfile();
  void log_impact(Particle::OnePart *ip, int isurf, double *norm,
                  double E_eV, double theta_deg, int outcome,
                  double E_out_eV);

  virtual void init_reactions();
  void load_surface_file();
  double interp_table(const std::vector<double> &table,
                      double e_eV, double a_deg) const;
  double sample_thompson(double Eb, double Emax);
  void sample_cosine_direction(double *norm, double *dir, double vmag);
  double sample_from_cdf(const std::vector<double> &cdf, int iE, int iA);
  void readfile(char *);
  int readone(char *, char *, int &, int &);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

*/
