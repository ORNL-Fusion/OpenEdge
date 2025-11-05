
#ifdef FIX_CLASS

FixStyle(coll/background,FixCollBackground)

#else

#ifndef SPARTA_FIX_COLL_BACKGROUND_H
#define SPARTA_FIX_COLL_BACKGROUND_H

#include <string>
#include <H5Cpp.h>
#include <map>
#include <unordered_map>
#include <tuple>
#include <stdio.h>
#include "fix.h"

namespace SPARTA_NS {
    class RanKnuth;
    
enum CollSrcKind { COLL_SRC_NONE, COLL_SRC_VAR, COLL_SRC_COMP };

struct CollGridSrc {
  CollSrcKind kind = COLL_SRC_NONE;
  // VAR path
  char *vname = nullptr;  int varid = -1;
  // COMP path
  char *cid   = nullptr;  int icompute = -1;  int col = 0; // 1-based

  // --- cache for per-timestep fast access ---
  double **arr_cache = nullptr; // c->array_grid
  int      src_index = -1;      // mapped column index
  int      cache_ts  = -1;      // update->ntimestep when filled
};


class FixCollBackground : public Fix {
public:
    FixCollBackground(class SPARTA*, int, char**);
    virtual ~FixCollBackground();
    int setmask();
    void init();
    void end_of_step();
    double memory_usage();



protected:

    RanKnuth *rng = nullptr;                // now points to SPARTA_NS::RanKnuth

    FILE* fp;
    int nlist;
    int atomic_number;
    bigint* tally_reactions, * tally_reactions_all;
    int tally_flag;
    int maxgrid;
    virtual void end_of_step_no_average();

    // PMI
   
    // void backgroundCollisions(double *v, double mass, double charge) ;
    int use_grid_plasma = 0, use_grid_bfield = 0;
    // plasma sources: Te, Ti, Ni, v_par
    CollGridSrc srcTe, srcTi, srcNi, srcVpar;
    CollGridSrc srcBr, srcBt, srcBz;

        // scratch for grid-variable path
    double **plasma_grid = nullptr;   // [nlocal][4] : Te,Ti,Ni,Vpar
    double **b_grid      = nullptr;   // [nlocal][3] : Br,Bt,Bz
    int maxgrid_plasma = 0, maxgrid_b = 0;
    double A_background = 2.0; // Coulomb logarithm for background collisions
    double Z_background = 1.0; // Charge state for background collisions

    void compute_plasma_grid();
    void compute_bfield_grid();
    double fetch_compute_cell_value(const CollGridSrc& S, int icell);
    inline void refresh_compute_src(CollGridSrc &S);

    void backgroundCollisions(Particle::OnePart *ip);;
    void getSlowDownDirections2(double parallel_direction[], double perp_direction1[], 
        double perp_direction2[], double vx, double vy, double vz) ;
    void getSlowDownFrequencies(double& nu_friction, double& nu_deflection, 
            double& nu_parallel, double& nu_energy, 
            double *v, double charge, double amu, double te_eV, double ti_eV, double density);
};

} // namespace SPARTA_NS

#endif
#endif


