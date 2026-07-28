# fix grain/drag (droplet/drag)

Epstein (free-molecular) drag toward the background parallel flow, exact
exponential integrator, gravity folded in.

    fix ID grain/drag Nevery A_bg Z_bg background PD [keywords]

Keywords:

- `gravity gx gy gz` — physical (R, Z, phi) components; mapped to SPARTA
  slots internally (do not combine with `fix gravity`).
- `model epstein|coulomb` — Coulomb adds the ion-drag multiplier
  (collection + orbit terms).
- `coulomb/self yes|no` — compute chi, delta = Ti/Te and lnLambda per
  particle from the OML charge (`grain/charge`) and local plasma
  (DUSTT closure, Hutchinson-fit lnLambda). Without it the three
  constants below are used.
- `coulomb/chi V`, `coulomb/delta V`, `coulomb/lnlambda V` — manual values
  (defaults 0 / 1 / 10).
- `material NAME` — grain density for the Epstein frequency (default: the
  legacy hardcoded Li 534 kg/m³).
- `mass|radius|temp V` — seed values for particles missing attributes.
- `mixture ID` — restrict to a mixture (default: all particles).

nu_E = alpha_E·rho_gas·v_th,i/(rho_d·R); update
v = u_par + (v − u_par − g/nu)·exp(−nu·dt) + g/nu (unconditionally stable).
