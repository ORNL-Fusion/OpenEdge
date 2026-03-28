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

enum HeatfluxMode { HF_NONE=0, HF_FILE, HF_CONST };

    struct HeatFluxData{
    std::vector<double> r;
    std::vector<double> z;
    std::vector<std::vector<double>> q_mag;
    std::vector<std::vector<double>> grad_te_r;
    std::vector<std::vector<double>> grad_te_z;
    };

    struct HeatFluxParams {
    double r;
    double z;
    double q_mag;
    double grad_te_r;
    double grad_te_z;
    };


class FixEvap : public Fix {
public:
    FixEvap(class SPARTA*, int, char**);
    ~FixEvap() override;
    int setmask() override;
    void init() override;
    double memory_usage() override;

      HeatFluxData heat_flux_data;
      HeatfluxMode heatflux_mode = HF_NONE;
      double      Qs_const = 0.0;     // when HF_CONST
      double      heatflux_scale = 3.0; // legacy multiplier, now user-configurable
      double      rocket_eta = 0.0;    // asymmetry parameter for rocket force [0,1]

protected:
    int imix;
    void end_of_step() override;
    void start_of_step() override;

    std::string heatfluxFilename;
    void droplet_evaporation_model(Particle::OnePart *ip,
                                        const double dt_half);
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
