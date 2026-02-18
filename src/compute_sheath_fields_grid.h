/* ----------------------------------------------------------------------
   OpenEdge: sheath electric field per grid cell from plasma + wall geometry
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(sheath/fields/grid,ComputeSheathFieldsGrid)

#else

#ifndef SPARTA_COMPUTE_SHEATH_FIELDS_GRID_H
#define SPARTA_COMPUTE_SHEATH_FIELDS_GRID_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeSheathFieldsGrid : public Compute {
 public:
  ComputeSheathFieldsGrid(class SPARTA *, int, char **);
  ~ComputeSheathFieldsGrid();
  void init();
  void compute_per_grid();
  void reallocate();
  bigint memory_usage();

 protected:
  int nglocal;
  int groupbit;
  int nvalue;
  int *value;

  int iplasma;
  int igeom;

  double dmax;
  double mach_min;
  double mach_max;
  double gamma_see;
  double cur_A_m2;
  double mD_amu;
  double emax_vpm;
  double borodkina_pot_mult;
  int sheath_model;

  enum {
    EX, EY, EZ, ESHEATH, MACH_PAR, MACH_N, ALPHA, ACTIVE, BDOTN, DIST, SURFID
  };
  enum {
    MODEL_EIRENE = 0,
    MODEL_BORODKINA = 1
  };
};

}

#endif
#endif
