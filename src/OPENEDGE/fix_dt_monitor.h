/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(dt/monitor,FixDtMonitor)

#else

#ifndef SPARTA_FIX_DT_MONITOR_H
#define SPARTA_FIX_DT_MONITOR_H

#include "fix.h"
#include <vector>

namespace SPARTA_NS {

class FixDtMonitor : public Fix {
 public:
  FixDtMonitor(class SPARTA *, int, char **);
  ~FixDtMonitor();
  int setmask();
  void init();
  void start_of_step();
  void end_of_step();
  void post_run();
  double compute_vector(int);

 private:
  int nsp;                       // species count the tallies are sized for
  int me;
  bigint win_first,run_first;    // first step of the reporting window / run
  bigint last_report;
  int warned_gyro,warned_chem;
  double vec[4];                 // last window: max and fraction, gyro then chem
  std::vector<class FixVolumeChemAdas *> chem;

  // global sums of the run, per species: 0..3 gyro (exposure, above, max,
  // guiding-centre exposure), 4..6 chem (weight, above, max)
  std::vector<double> run;
  double run_bad[4];             // non-finite gyro, non-finite chem, capped chem, unused

  void allocate();
  void report();
};

}

#endif
#endif
