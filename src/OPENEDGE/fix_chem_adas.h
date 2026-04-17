
/* ----------------------------------------------------------------------
    OpenEdge: ADAS ionization/recombination chemistry fix
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - 42d
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(chem/adas, FixChemAdas)

#else

#ifndef SPARTA_FIX_CHEM_ADAS_H
#define SPARTA_FIX_CHEM_ADAS_H

#include <string>
#include <H5Cpp.h>
#include <map>
#include <stdio.h>
#include "fix.h"
#include "update.h"

namespace SPARTA_NS {

enum class ReactionType { Ionization, Recombination, ChargeExchange };
enum SrcKind { SRC_NONE, SRC_VAR, SRC_COMP };
struct GridSrc {
  SrcKind kind = SRC_NONE;
  int  varid   = -1;
  char *vname  = nullptr;
  int   icompute = -1;
  int   col      = 0;
  char *cid      = nullptr;
  double **arr_cache = nullptr;
  double  *vec_cache = nullptr;
  int      src_index = -1;
  int      cache_ts  = -1;
};

class RanKnuth;
class ComputePlasmaFields;


class FixChemAdas : public Fix {
public:
    FixChemAdas(class SPARTA*, int, char**);
    virtual ~FixChemAdas();
    int setmask();
    void init();
    void end_of_step();
    void post_run();
    double memory_usage();
    bigint gas_react_one() const override { return nreact_one; }
    bigint gas_react_running() const override { return nreact_running; }

    // True iff any reaction is charge-exchange (type 2 = EXCHANGE in the
    // file-local enum). Used by Update::init() to skip PCACHE_TI/VPAR/BFIELD
    // cache writes when no CX channel is loaded — those fields are only
    // consumed by attempt() when a CX reaction fires.
    bool needs_cx_fields() const {
      for (int i = 0; i < nlist; i++) if (rlist[i].type == 2) return true;
      return false;
    }

int    use_grid_plasma = 0;
char  *tstr = NULL, *nstr = NULL;
int    tvar = -1,   nvar = -1;
GridSrc srcTe, srcNe;
    inline void compute_plasma_grid();
double read_cell(const GridSrc &S, int icell, int var_col /*0=Te,1=ne*/);

// Per-cell source-term output for Gkeyll / external coupling.
// Columns: 0=ionization, 1=recombination, 2=charge exchange, 3=dissociation.
// Units: cumulative event count per cell since fix start (stored as double).
// Exposed via the inherited base-class array_grid with size_per_grid_cols = 4.
// Allocated lazily when grid->nlocal changes (see ensure_src_alloc).
int maxgrid_src = 0;

// Internal 2-column buffer used to cache Te/ne from variables/computes during
// attempt() when use_grid_plasma is set. Distinct from the output array_grid.
double **plasma_cache_2d = NULL;



protected:
    FILE* fp;
    int nlist;
    int atomic_number;
    bigint* tally_reactions, * tally_reactions_all;
    bigint nreact_one, nreact_running;
    int tally_flag;
    int maxgrid;
    int maxgrid_plasma;
    int icompute;
    virtual void end_of_step_no_average();

    struct RateData {
        std::vector<double> Atomic_Number;
        // Flat contiguous rate tables: data[q * nT * nD + iT * nD + iD]
        std::vector<double> ion_coeff, rec_coeff, cx_coeff;
        int ion_nQ, ion_nT, ion_nD;
        int rec_nQ, rec_nT, rec_nD;
        int cx_nQ, cx_nT, cx_nD;
        std::vector<double> gridT_ion, gridD_ion;
        std::vector<double> gridT_rec, gridD_rec;
        std::vector<double> gridT_cx, gridD_cx;

        inline double ion_at(int q, int it, int id) const {
            return ion_coeff[q * ion_nT * ion_nD + it * ion_nD + id];
        }
        inline double rec_at(int q, int it, int id) const {
            return rec_coeff[q * rec_nT * rec_nD + it * rec_nD + id];
        }
        inline double cx_at(int q, int it, int id) const {
            return cx_coeff[q * cx_nT * cx_nD + it * cx_nD + id];
        }
    };

    std::map<int, RateData> materials_rate_data;

    RateData rate_data;
    RanKnuth *rng_adas;
    ComputePlasmaFields *cp_plasma_cached_;
    void readRateData(const std::string& filePath, RateData& data);
    double computeReactionLambda(double rate_log10_cm3s, double dt, double ne_m3);
    bool setupInterpolation(ReactionType reactionType, int atomic_number, size_t charge_idx,
                    double te, double ne, double& x0, double& x1, double& y0, double& y1,
                    double& f00, double& f01, double& f10, double& f11);
    void interpolateRateData(int atomic_number, double charge, int icell, double te, double ne,
                        double& rate_final, ReactionType reactionType);
    void readRateDataParallel(const std::string& filePath, RateData& rateData);
    void broadcastRateData(RateData& rateData);

    struct OneReaction {
        int active;
        int initflag;
        int type;
        int style;
        int ncoeff;
        int nreactant, nproduct;
        char** id_reactants, ** id_products;
        int* reactants, * products;
        double* coeff;
        char* id;
    };

    OneReaction* rlist;
    int maxlist;

    struct ReactionIJ {
        int* list;
        int n;
    };

    ReactionIJ* reactions;
    int* list_ij;

    int attempt(Particle::OnePart* ip, double Te_eV, double ne_m3,
                double Ti_eV = 0.0, double vpar = 0.0,
                double bx = 0.0, double by = 0.0, double bz = 0.0);
    void readfile(char*);
    int readone(char*, char*, int&, int&);
    void check_duplicate();
    void print_reaction(char*, char*);
    void print_reaction(OneReaction*);
    void refresh_compute_src(GridSrc &S);

    // Deferred particle creation for dissociation (avoids invalidation during iteration)
    struct DeferredParticle {
      double x[3], v[3];
      int species, icell;
    };
    std::vector<DeferredParticle> deferred_particles;

};

} // namespace SPARTA_NS

#endif
#endif
