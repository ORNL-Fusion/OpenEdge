
/* ----------------------------------------------------------------------
    OpenEdge: ADAS ionization/recombination chemistry fix
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - 42d
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(volume/chem/adas, FixVolumeChemAdas)

#else

#ifndef SPARTA_FIX_VOLUME_CHEM_ADAS_H
#define SPARTA_FIX_VOLUME_CHEM_ADAS_H

#include <string>
#include <H5Cpp.h>
#include <map>
#include <stdio.h>
#include "fix.h"
#include "update.h"

namespace SPARTA_NS {

enum class ReactionType { Ionization, Recombination, ChargeExchange,
                          LineRadiation, RecombRadiation };

class RanKnuth;


class FixVolumeChemAdas : public Fix {
public:
    FixVolumeChemAdas(class SPARTA*, int, char**);
    virtual ~FixVolumeChemAdas();
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

    // Zero the 20-column per-cell source tally (array_grid). Called between
    // outer coupling iterations so each Gkeyll/SOLPS handoff sees a fresh
    // accumulation from the new puff, not a sum over previous iterations.
    // Leaves per-type reaction counters and exhaust state untouched: those
    // are diagnostics, not coupled state.
    void reset_tally();

// Per-cell source-term output for Gkeyll / external coupling.
// Columns: 0=ionization, 1=recombination, 2=charge exchange, 3=dissociation.
// Units: cumulative event count per cell since fix start (stored as double).
// Exposed via the inherited base-class array_grid with size_per_grid_cols = 4.
// Allocated lazily when grid->nlocal changes (see ensure_src_alloc).
int maxgrid_src = 0;

// Output units for the 20-column per-cell source tally (array_grid):
//   TALLY_COUNTS: raw cumulative event totals since the fix was created.
//                 count col = events, m*v col = [kg m/s], 0.5 m v^2 col = [J].
//   TALLY_RATE:   instantaneous rate over the last `nevery` window, in SI.
//                 count col = particle source [m^-3 s^-1],
//                 m*v col   = momentum source [kg m^-2 s^-2] = [N/m^3],
//                 0.5 m v^2 col = energy source [W/m^3].
// In RATE mode the tally is zeroed at the start of every end_of_step and the
// accumulated totals are normalized by (vol * window_s / fnum) at the end,
// so the array_grid always holds a window-average and not a running total.
// This plays nicely with `fix ave/grid ... f_fchem[*] ave running` for
// smoothed rates ready to hand to Gkeyll / SOLPS.
// TALLY_BATCH: batch-weighted rate (EIRENE-style Monte Carlo). A one-shot
// puff launches N_batch trajectories representing a physical emission rate
// R_puff [atoms/s]. Each event carries a weight w = R_puff / N_batch
// (physical events/s per trajectory); per-cell contributions are divided by
// cell volume so array_grid stores the steady-state source rate directly.
// Tally is cumulative across the run and *does not* reset per window -- the
// final value after all trajectories have terminated is the steady-state
// Gkeyll/SOLPS source (what EIRENE returns to S3X/SOLPS each coupling call).
//
// TALLY_BATCH_FIX: same semantics but N_batch is pulled at tally time from
// the cumulative emit count of a paired emit fix (e.g. fix surface/emit/puff).
// Useful when you don't want to hand-match the N in `units batch` with the
// `stop_at_np` cap on the emit fix -- the chem fix reads it directly. Value
// is cached at the start of every chem nevery step, so scaling tracks the
// ramp-up phase. Post-ramp (latched), N is constant and the scaling is exact.
enum { TALLY_COUNTS, TALLY_RATE, TALLY_BATCH, TALLY_BATCH_FIX };
int    tally_units   = TALLY_COUNTS;

// Output layout of the per-cell tally (array_grid):
//   OUT_SUMMARY  (default): 6 cols = {Sp, Sm_x, Sm_y, Sm_z, Qe, Qi}, signs per
//                 Sec.4 of docs/neutral_plasma_coupling/main.tex -- signed
//                 plasma-frame source moments, ready to feed a fluid solver.
//   OUT_DETAILED (diagnostic): 20 cols = quantity x reaction cross-product
//                 {count, m vx, m vy, m vz, 0.5 m v^2} x
//                 {ioniz, recomb, CX, dissoc}, unsigned raw tally (legacy
//                 layout, kept for per-reaction debugging). No Eeff folding,
//                 no Qe/Qi split; each cell is just summed events.
enum { OUT_SUMMARY = 0, OUT_DETAILED = 1 };
int    output_mode = OUT_SUMMARY;

double batch_R_puff  = 0.0;    // [atoms/s] physical emission rate
int    batch_N       = 0;      // fixed batch N (TALLY_BATCH)
char  *batch_fix_id  = nullptr;// emit fix ID (TALLY_BATCH_FIX)
int    batch_fix_idx = -1;     // resolved index in modify->fix[]
bigint batch_N_cached = 0;     // cached N from emit fix this chem step

// Mode A (EIRENE-semantics) state.
//   eirene_mode     = 1 -> delete particle on IONIZATION instead of relabeling
//   stop_on_exhaust = 1 -> stop the run when source_species population hits 0
//                          (redundant with SPARTA's built-in nglobal==0 check
//                          for pure-neutral runs, but needed when kinetic
//                          impurities remain alongside vanishing neutrals).
//   source_species  = resolved species indices matching src_species_names.
int    eirene_mode     = 0;
int    stop_on_exhaust = 0;
// Channel toggles (default: all on). Each reaction's type is matched
// against these flags at init; disabled reactions are flagged inactive
// the same way as when a reactant/product species is missing, so the
// runtime doesn't need per-tick gating. Useful for ablation studies
// (e.g. `cx no` to isolate ionization-only physics).
int    chan_ionization  = 1;
int    chan_recombination = 1;
int    chan_cx          = 1;
int    chan_dissociation = 1;
// Allow early termination when only a small fat-tail population remains.
// stop_on_exhaust triggers when alive_global <= exhaust_threshold (default 0,
// i.e. wait for the population to literally hit zero). Setting a nonzero
// threshold (e.g. 10% of batch) skips the slow-converging tail where per-step
// cost stays high but source-moment contributions are negligible. Closer to
// what EIRENE gets for free by tracing trajectories independently.
// exhaust_armed is set to 1 the first time alive_global > exhaust_threshold
// (batch ramp-up is detectable -- prevents spurious exit at step 1 when
// population is still empty before the puff has loaded).
bigint exhaust_threshold = 0;
int    exhaust_armed     = 0;
int    nsrc_species    = 0;
char **src_species_names = nullptr;
std::vector<int> source_species;
std::vector<int> dellist;

protected:
    FILE* fp;
    int nlist;
    int atomic_number;
    bigint* tally_reactions, * tally_reactions_all;
    bigint nreact_one, nreact_running;
    int tally_flag;
    int maxgrid;
    int icompute;
    virtual void end_of_step_no_average();

    struct RateData {
        std::vector<double> Atomic_Number;
        // Flat contiguous rate tables: data[q * nT * nD + iT * nD + iD].
        // Rate coefficients stored as log10(sigma_v [cm^3/s]); radiation power
        // coefficients as log10(P [W cm^3]).  Consumers convert to SI at lookup.
        std::vector<double> ion_coeff, rec_coeff, cx_coeff;
        std::vector<double> plt_coeff, prb_coeff;         // line rad, recomb+brems
        int ion_nQ = 0, ion_nT = 0, ion_nD = 0;
        int rec_nQ = 0, rec_nT = 0, rec_nD = 0;
        int cx_nQ  = 0, cx_nT  = 0, cx_nD  = 0;
        int plt_nQ = 0, plt_nT = 0, plt_nD = 0;
        int prb_nQ = 0, prb_nT = 0, prb_nD = 0;
        std::vector<double> gridT_ion, gridD_ion;
        std::vector<double> gridT_rec, gridD_rec;
        std::vector<double> gridT_cx,  gridD_cx;
        std::vector<double> gridT_plt, gridD_plt;
        std::vector<double> gridT_prb, gridD_prb;
        // Ionization potential per charge state q (q=0 -> neutral -> 1+, etc.), eV.
        // Empty if the ADAS rate file didn't ship one.
        std::vector<double> ion_potential;

        inline double ion_at(int q, int it, int id) const {
            return ion_coeff[q * ion_nT * ion_nD + it * ion_nD + id];
        }
        inline double rec_at(int q, int it, int id) const {
            return rec_coeff[q * rec_nT * rec_nD + it * rec_nD + id];
        }
        inline double cx_at(int q, int it, int id) const {
            return cx_coeff[q * cx_nT * cx_nD + it * cx_nD + id];
        }
        inline double plt_at(int q, int it, int id) const {
            return plt_coeff[q * plt_nT * plt_nD + it * plt_nD + id];
        }
        inline double prb_at(int q, int it, int id) const {
            return prb_coeff[q * prb_nT * prb_nD + it * prb_nD + id];
        }
    };

    std::map<int, RateData> materials_rate_data;

    RateData rate_data;
    RanKnuth *rng_adas;
    double computeReactionLambda(double rate_log10_cm3s, double dt, double ne_m3);
    bool setupInterpolation(ReactionType reactionType, int atomic_number, size_t charge_idx,
                    double te, double ne, double& x0, double& x1, double& y0, double& y1,
                    double& f00, double& f01, double& f10, double& f11);
    void interpolateRateData(int atomic_number, double charge, int icell, double te, double ne,
                        double& rate_final, ReactionType reactionType);

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

    int attempt(Particle::OnePart* ip, int ip_index,
                double Te_eV, double ne_m3,
                double Ti_eV = 0.0, double vpar = 0.0,
                double bx = 0.0, double by = 0.0, double bz = 0.0);
    void readfile(char*);
    int readone(char*, char*, int&, int&);
    void check_duplicate();
    void print_reaction(char*, char*);
    void print_reaction(OneReaction*);

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
