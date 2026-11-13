# Prompt redeposition in a uniform magnetic field

O atoms leave the wall along its normal and ionize at the rate nu in a uniform
plasma with B parallel to the wall; no electric field, no collisions. An ion born
at height h returns to the wall if h < rho = v/Omega, so the redeposited fraction
is f = 1 - exp(-nu/Omega) for any launch speed (0.6326 here, nu/Omega = 1).

    python3 scripts/scan.py        # SPA=<binary> NP=<ranks>; --quick for the regression

The scan varies the timestep (nu*dt = Omega*dt = 0.2 ... 0.01) with emission every
step and in batches of one ionization time. Emission every step and batched agree
at every timestep. The remaining error is the transport-chemistry splitting (an atom
is ionized at the end of the step it reacts in) plus the Boris phase error: f is
5.5 % high at nu*dt = 0.2, 2.9 % at 0.1, 1.5 % at 0.05, 0.7 % at 0.02 and 0.2 % at
0.01, first order in the timestep. PASS requires 1 % at nu*dt <= 0.02.
