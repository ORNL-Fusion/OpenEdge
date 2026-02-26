#ifdef FIX_CLASS

FixStyle(evaporation,FixEvap)

#else

#ifndef SPARTA_FIX_EVAP_H
#define SPARTA_FIX_EVAP_H

#include <H5Cpp.h>
#include "fix.h"
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


// array_grid columns (0-based index; 1-based in SPARTA script references f_ID[N]):
//   [0]  dm_kg:    mass lost per cell per full step [kg]      (sum: all droplets + both half-kicks)
//   [1]  dn_atoms: atoms lost per cell per full step          (= dm_kg / AM,  AM=1.53e-26 kg)
//   [2]  heat_J:   heat absorbed from plasma per cell [J]    (= Qs*4πR_new²*dt_half, summed)

class FixEvap : public Fix {
public:
    FixEvap(class SPARTA*, int, char**);
    virtual ~FixEvap();
    int setmask();
    void init();
    double memory_usage();

      HeatFluxData heat_flux_data;
      double      Qs_const = 0.0;     // when HF_CONST

protected:
    int maxgrid;
    int imix;
    void end_of_step();
    void start_of_step() override;

    std::string heatfluxFilename;
    void droplet_evaporation_model(Particle::OnePart *ip,
                                        const double dt_half,
                                        const int icell);
    double set_mass = -1.0;
    double set_temp = -1.0;
    double set_radius = -1.0;

    void broadcastHeatFluxData(HeatFluxData& );
    HeatFluxParams interpHeatFluxAtPos(double r, double z, const HeatFluxData& data) const;
    HeatFluxData readHeatFlux(const std::string& filePath);
    void initializeHeatFluxData();
    void evap_half(double dt_half);

};

} // namespace SPARTA_NS

#endif
#endif
