
#ifdef FIX_CLASS

FixStyle(evap,FixEvap)

#else

#ifndef SPARTA_FIX_EVAP_H
#define SPARTA_FIX_EVAP_H

#include <H5Cpp.h>
#include <map>
#include <unordered_map>
#include <tuple>
#include <stdio.h>
#include "fix.h"
#include "update.h"
 #include <string>
#include <vector>
#include <algorithm>


namespace SPARTA_NS {

    struct HeatFluxData{
    std::vector<double> r;   
    std::vector<double> z; 
    std::vector<std::vector<double>> q_mag;
    };

    struct HeatFluxParams {
    double r;
    double z;
    double q_mag;
    };


class FixEvap : public Fix {
public:
    FixEvap(class SPARTA*, int, char**);
    virtual ~FixEvap();
    int setmask();
    void init();
    void end_of_step();
    double memory_usage();

      HeatFluxData heat_flux_data;
      double      Qs_const = 0.0;     // when HF_CONST

protected:
    FILE* fp;
    int nlist;
    int atomic_number;
    bigint* tally_reactions, * tally_reactions_all;
    int tally_flag;
    int maxgrid;
    virtual void end_of_step_no_average();

    // PMI
    std::string heatfluxFilename;
    void droplet_evaporation_model(Particle::OnePart *);
    double set_mass = -1.0;
    double set_temp = -1.0;    // EXPECTED IN KELVIN
    double set_radius = -1.0;
    int force_override = 0;    // 0 = only set when current value <= 0, 1 = always

    // HeatFluxData  heat_flux_data;
    void broadcastHeatFluxData(HeatFluxData& );
    HeatFluxParams interpHeatFluxAt(int icell, const HeatFluxData& data) const;
    mutable std::unordered_map<int, HeatFluxParams> flux_cache;
    HeatFluxData readHeatFlux(const std::string& filePath);
    void initializeHeatFluxData();

    
};

} // namespace SPARTA_NS

#endif
#endif
