
#ifdef FIX_CLASS

FixStyle(evap,FixEvap)

#else

#ifndef SPARTA_FIX_EVAP_H
#define SPARTA_FIX_EVAP_H

#include <string>
#include <H5Cpp.h>
#include <map>
#include <unordered_map>
#include <tuple>
#include <stdio.h>
#include "fix.h"
#include "update.h"

namespace SPARTA_NS {


class FixEvap : public Fix {
public:
    FixEvap(class SPARTA*, int, char**);
    virtual ~FixEvap();
    int setmask();
    void init();
    void end_of_step();
    double memory_usage();

protected:
    FILE* fp;
    int nlist;
    int atomic_number;
    bigint* tally_reactions, * tally_reactions_all;
    int tally_flag;
    int maxgrid;
    virtual void end_of_step_no_average();

    // PMI
   
    // void backgroundCollisions(double *v, double mass, double charge) ;
    void droplet_evaporation_model(Particle::OnePart *ip);

};

} // namespace SPARTA_NS

#endif
#endif
