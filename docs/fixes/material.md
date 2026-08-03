# material command

Define or override a grain material consumed by grain/ablate, grain/drag
and grain/charge (built-ins: Li, B).

    material NAME key value [key value ...]

Keys: `rho` [kg/m³], `cp` [J/kg/K], `mass_amu`, `hvap_J_mol`,
`antoine_a`/`antoine_b` (log10 p_sat[atm] = a + b/T, b < 0),
`emissivity`, `work_function_eV`, `richardson_A`, `tmelt_K`,
`hmelt_J_mol`, `tensile_Pa` (0 = no electrostatic breakup).

    material B emissivity 0.75
    material MyStuff rho 1850 cp 1800 mass_amu 9.0 hvap_J_mol 5e5 \
             antoine_a 9.1 antoine_b -28000
