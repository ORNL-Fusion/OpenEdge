
#ifdef FIX_CLASS

FixStyle(chem/adas, FixChemAdas)

#else

#ifndef SPARTA_FIX_CHEM_ADAS_H
#define SPARTA_FIX_CHEM_ADAS_H

#include <string>
#include <H5Cpp.h>
#include <map>
#include <unordered_map>
#include <tuple>
#include <stdio.h>
#include "fix.h"
#include "update.h"

namespace SPARTA_NS {

// Enum for reaction type
enum class ReactionType { Ionization, Recombination };
enum SrcKind { SRC_NONE, SRC_VAR, SRC_COMP };
struct GridSrc {
  SrcKind kind = SRC_NONE;

  // variable case (grid-style variable)
  int  varid   = -1;        // resolved in init()
  char *vname  = nullptr;   // optional: if you store names

  // compute case
  int   icompute = -1;      // index in modify->compute[]
  int   col      = 0;       // 1-based column c_ID[col]
  char *cid      = nullptr; // compute ID (owned)

  // per-step cache (for compute case)
  double **arr_cache = nullptr;
  int      src_index = -1;
  int      cache_ts  = -1;
};


// Struct for the interpolation cache key
struct InterpolationKey {
    int icell;
    int charge;
    int atomic_number;
    ReactionType reactionType;

    // Equality operator
    bool operator==(const InterpolationKey& other) const {
        return icell == other.icell &&
               charge == other.charge &&
               atomic_number == other.atomic_number &&
               reactionType == other.reactionType;
    }
};

// Local hasher (avoid polluting std::hash)
struct InterpolationKeyHasher {
   std::size_t operator()(const InterpolationKey& k) const {
       std::size_t h1 = std::hash<int>{}(k.icell);
       std::size_t h2 = std::hash<int>{}(k.charge);
       std::size_t h3 = std::hash<int>{}(k.atomic_number);
       std::size_t h4 = std::hash<int>{}(static_cast<int>(k.reactionType));
       return ((h1 ^ (h2 << 1)) ^ (h3 << 2)) ^ (h4 << 3);
   }
};


class FixChemAdas : public Fix {
public:
    FixChemAdas(class SPARTA*, int, char**);
    virtual ~FixChemAdas();
    int setmask();
    void init();
    void end_of_step();
    double memory_usage();


int    use_grid_plasma = 0;
char  *tstr = NULL, *nstr = NULL;
int    tvar = -1,   nvar = -1;
double **array_grid = NULL;
GridSrc srcTe, srcNe;
    inline void compute_plasma_grid();
    inline double fetch_compute_cell_value(const GridSrc& S, int icell);

inline double read_cell(const GridSrc &S, int icell, int var_col /*0=Te,1=ne*/) {
  if (S.kind == SRC_COMP)
    return (S.arr_cache && S.src_index >= 0) ? S.arr_cache[icell][S.src_index] : 0.0;
  // VAR path
  return array_grid ? array_grid[icell][var_col] : 0.0;
}



protected:
    FILE* fp;
    int nlist;
    int atomic_number;
    bigint* tally_reactions, * tally_reactions_all;
    int tally_flag;
    int maxgrid;
    int maxgrid_plasma;
    int icompute;
    virtual void end_of_step_no_average();

    // PMI
    struct RateData {
        std::vector<double> Atomic_Number;
        std::vector<std::vector<std::vector<double>>> IonizationRateCoeff, RecombinationRateCoeff;
        std::vector<std::vector<double>> gridChargeState_Ionization, gridChargeState_Recombination;
        std::vector<double> gridDensity_Ionization, gridDensity_Recombination, gridTemperature_Ionization, gridTemperature_Recombination;
    };

    std::map<std::string, RateData> rateDataCache;
    std::unordered_map<int, RateData> materials_rate_data;
    std::unordered_map<InterpolationKey, double, InterpolationKeyHasher> rateCache;

    RateData rate_data;
    void readRateData(const std::string& filePath, RateData& data);
    double computeReactionProbability(double rate, double dt, double ne);
    bool setupInterpolation(ReactionType reactionType, int atomic_number, size_t charge_idx,
                    double te, double ne, double& x0, double& x1, double& y0, double& y1,
                    double& f00, double& f01, double& f10, double& f11);
    void interpolateRateData(int atomic_number, double charge, int icell, double te, double ne,
                        double& rate_final, ReactionType reactionType);
    void readRateDataParallel(const std::string& filePath, RateData& rateData);
    void broadcastRateData(RateData& rateData);
    double computeReactionProbability_(double rate,
                                               double dt,
                                               double ne,
                                               bool rate_is_log10 /*=true*/,
                                               double k_units_scale /*=1.0*/);

    struct OneReaction {
        int active;           // 1 if reaction is active
        int initflag;         // 1 if reaction params have been init
        int type;             // reaction type = DISSOCIATION, etc
        int style;            // reaction style = ARRHENIUS, etc
        int ncoeff;           // # of numerical coeffs
        int nreactant, nproduct;  // # of reactants and products
        char** id_reactants, ** id_products;  // species IDs
        int* reactants, * products;           // species indices
        double* coeff;        // numerical coeffs
        char* id;             // reaction ID (formula)
    };

    OneReaction* rlist;       // list of all reactions
    int maxlist;              // max # of reactions

    struct ReactionIJ {
        int* list;            // N-length list of rlist indices
        int n;                // # of reactions in list
    };

    ReactionIJ* reactions;
    int* list_ij;

    int attempt(Particle::OnePart* ip, double Te_eV, double ne_m3);
    void readfile(char*);
    int readone(char*, char*, int&, int&);
    void check_duplicate();
    void print_reaction(char*, char*);
    void print_reaction(OneReaction*);
      void   refresh_compute_src(GridSrc &S);   // <— declaration only

};

} // namespace SPARTA_NS

#endif
#endif
