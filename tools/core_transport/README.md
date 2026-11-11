# tools/core_transport

Inputs for `fix core/transport` built from a case's `input/plasma.h5`.

    python tools/core_transport/make_core_profiles.py --case <case dir> [--mode parabolic|linear|flat]
        -> input/core_profiles.dat   (rho_N n_e T_e T_i r_vol; `profiles` keyword)
    python tools/core_transport/core_geometry.py --case <case dir>
        -> input/core_geometry.dat   (rho_N r_vol g1 g2; `coefficients physical geometry`)
    python tools/core_transport/core_geometry.py --selftest      # circular and elliptical tori vs closed forms

`core_geometry.py` assumes D_perp constant on a flux surface and a pinch speed along the surface
normal. Factors below the rho_N written in the file header are carried from the first resolved
shell to the axis and are not shell measurements.
