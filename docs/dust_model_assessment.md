# OpenEdge dust/droplet model — assessment vs. DUSTT/DIS, and the path to boron

Scope: the five `fix droplet/*` models (emit, drag, viscous, charge, evaporate),
compared against DUSTT (Pigarov et al., Phys. Plasmas 12, 122508, 2005) and the
DIS usage in Afonin et al., Nucl. Fusion 63 (2023) 126057. Written 2026-07-28.

## 1. What exists today (and is genuinely good)

| Area | Model in OpenEdge | Notes |
|---|---|---|
| Grain state | per-particle `radius`, `temp`, `mass` evolve; set from 12-column species file | `particle.h:77-80` |
| Emission | `droplet/emit` = SPARTA emit/surf + `vmin/vmax`, `angle cosine`, fixed speed/angle | one size bin per species |
| Drag | Epstein free-molecular vs. background parallel flow, exact exponential integrator, gravity folded in; optional Coulomb multiplier | unconditionally stable |
| Charging | instantaneous OML floating potential (bisection) + Richardson–Dushman thermionic with small-grain Schottky correction; feeds the Boris pusher via `droplet_charge` | `richardson_A`, `work_function_eV` are user-settable |
| Evaporation | Antoine vapor pressure + Hertz–Knudsen (Langmuir) flux; dR/dt from flux; lumped-capacitance dT/dt = (heating − latent)·3/(ρCpR); rocket recoil along −∇Te; vapor spawned as real kinetic atoms | adaptive substepping (25 K / 2 % R per substep) |

The evaporation law is the same *class* as DUSTT's sublimation flux (Eq. 24
there is the identical Langmuir form with a 10^(A−B/T) vapor-pressure curve).
**So: yes, there is a thermal-evaporation/ablation model — for liquid lithium.**

## 2. The headline gaps

### 2.1 The energy balance is incomplete (the real "ablation model" question)
DUSTT's grain heat balance: q_net = q_plasma(OML e/i collection with sheath
coefficients + surface recombination + neutrals) − εσ(T_d⁴−T_w⁴) − latent −
sputter/reflection carry-off. OpenEdge has: q = 0.25·|q_bkg|·scale − latent.

- **Heating**: prescribed background heat-flux field, not computed from local
  ne, Te, Ti. If `plasma.h5` lacks `mesh/q_par` (converter run without
  `--heatflux total` — true of the current ST40 case) it silently falls back to
  constant defaults. Missing entirely: electron/ion OML kinetic heating with
  sheath transmission (ζ≈2.5 each), surface-recombination (potential) heating,
  neutral heating.
- **Cooling**: no radiative cooling (no σT⁴ anywhere in the droplet code), no
  thermionic cooling, no back-condensation.
- Consequence for Li: acceptable — liquid Li stays 800–1200 K where εσT⁴
  (ε≈0.1) is a few kW/m² against MW/m² plasma flux, and latent-heat cooling
  dominates by construction.
- **Consequence for B: fatal.** Boron barely evaporates below ~2000 K
  (p_sat ~15 orders below Li at 900 K), so grains ride the heat flux up to
  2000–2500 K where εσT⁴ (ε≈0.8) is 1–2 MW/m² — a *leading* term in the
  balance. Without radiation, B grains in the model overheat and ablate far
  too early. Radiative cooling is the single most important missing term.

### 2.2 Phase and material behavior
- No melting/solidification (Li enters pre-melted at 773 K so it never
  mattered; B melts at 2349 K — grains are solid through most of their life,
  so the sublimation branch *is* the ablation channel).
- No temperature-dependent ρ, Cp (B's Cp roughly doubles from 300 K to 2000 K).
- No non-thermal mass loss (physical sputtering of the grain), no breakup
  (electrostatic disruption / Rayleigh limit), no vapor shielding. DUSTT has
  the first two; nobody in this class has real shielding (pellet codes do).

### 2.3 Self-consistency between fixes
- The OML charge computed by `droplet/charge` is used by the pusher, but the
  Coulomb ion-drag multiplier in `droplet/drag` takes *user-constant*
  chi/delta/lnΛ — it never reads the computed charge, local Debye length, or
  flow speed. DUSTT computes the collection+orbit drag from the actual
  floating potential with a Hutchinson-fitted lnΛ.
- Charging assumes a static Maxwellian ion current; no flow-shifted OML
  F_Γ(u) factor although the grains sit in a flowing SOL.

### 2.4 Statistics and mechanics
- 1 macro-particle = 1 physical grain (evaporate expects specwt = 1). A real
  dropper at mg/s of 25 µm powder is 10⁵–10⁶ grains/s — a per-grain weight
  (pweight-style) is needed to represent experimental drop rates.
- No dust–wall bounce/sticking model (DUSTT: mass/velocity/temperature
  restitution + diffuse/mirror mix); grains just use generic surf_collide.
- `droplet/evaporate` is 2D-only (fine for axi, blocks 3D).
- Emit quirk: in 2D the insertion point is hardcoded to each surf segment's
  midpoint (`rn = 0.5`), killing spatial randomization.

## 3. Boron by species file alone? No.
Hardcoded Li constants (no input keywords):

| Constant | Value | Where |
|---|---|---|
| atom mass | 1.15225e-26 kg | fix_droplet_evaporate.cpp:233 |
| density | 534 kg/m³ | evaporate:234, drag.h:53, viscous.h:69 |
| Cp | 4200 J/kg·K | evaporate:235 |
| latent heat | 1.47e5 J/mol | evaporate:236 |
| Antoine (mmHg) | a=5.055, b=−8023 | evaporate:297 |

Only charging is material-flexible today. Also note internal inconsistency:
`liquid_metal_strip.h` carries a *different* Li set (ρ=485, Antoine
5.66797/8310.41 in atm) — two sources of truth already for one material.

Boron needs (approximate; fit Antoine/JANAF properly before use): ρ≈2340
kg/m³, m=10.81 amu, T_melt=2349 K, ΔH_sub≈560–570 kJ/mol, Cp(T) ~1000→2100
J/kg·K, ε≈0.8, W=4.45 eV, and a sublimation vapor-pressure curve (log₁₀p
linear in 1/T with slope ~−29000 K).

## 4. Recommended refactor

1. **Rename the family `droplet/*` → `grain/*` (or `dust/*`)**, keeping the
   old style strings as aliases (the charge fix already demonstrates aliasing).
   "Grain" covers liquid Li droplets, solid B powder, and future pellet-like
   objects without implying a liquid.
2. **Introduce a material table** — one definition consumed by every grain fix
   and by `liquid_metal_strip`, e.g. a `material` command or file:
   `material Li rho 534 cp 4200 mass_amu 6.94 hvap_J_mol 1.47e5 antoine_mmHg
   5.055 -8023 emissivity 0.10 work_function 2.9 richardson_A 1.2e6 tmelt 453
   hmelt_J_mol 3.0e3`, and per-species assignment (`species ... material Li`).
   This removes every hardcoded constant and the Li/Li inconsistency in one
   move, and makes B a data problem instead of a code problem.
3. **Complete the energy balance** in (renamed) `grain/ablate`:
   q_net = q_heat − εσ(T⁴−T_w⁴) − Γ_evap·ΔH − (optional thermionic), with
   `q_heat` selectable: `heatflux background` (current behavior) or
   `oml ne Te Ti` (DUSTT Eqs. 26–27 style: OML fluxes × sheath-transmitted
   energies + recombination energy). The OML option removes the dependency on
   the h5 carrying q_par and is correct for grains immersed in plasma rather
   than sitting on a target.
4. **Melting branch**: clamp T at T_melt while integrating melt fraction
   (DUSTT Eq. 30); below T_melt use the sublimation vapor-pressure curve.
5. **Couple drag to charge**: compute chi = eφ/Te from `droplet_charge`,
   delta = Ti/Te and lnΛ locally (Khrapak/Hutchinson form) instead of the
   three user constants; keep the constants as overrides.
6. **Grain statistical weight** so one macro-grain represents N grains
   (scales vapor source and diagnostics; enables experimental mg/s rates).
7. Smaller items: fix the `rn = 0.5` midpoint hack; optional size-distribution
   sampling in emit; wall restitution/sticking coefficients; retire
   `droplet/viscous`; refresh docs/fixes/*.md (all five are stale in places:
   arg orders, defaults, and one describes a different interface entirely);
   deduplicate `src/` vs `src/OPENEDGE/` copies (they have already drifted).

## 5. Priority for the boron milestone
1. Material table + B data (blocks everything else).
2. Radiative cooling (the B-critical term).
3. OML heating option (or at minimum: rebuild plasma.h5 with
   `--heatflux total` so the current prescribed-q path uses real fluxes —
   the ST40 case today runs on fallback constants).
4. Melting/solid-sublimation branch.
5. Grain weighting for real drop rates.
6. Self-consistent ion drag; breakup; wall interaction — after first B runs.


## 6. Status update (2026-07-28, evening)

Closed since this assessment was written:
- material table (`material` command; Li, B built-ins) — §4.2
- radiative cooling, OML heating option, melting band — §4.3, §4.4
- grain statistical weighting (`grain/emit nweight` -> `grain_nweight`) — §4.6
- self-consistent Coulomb ion drag (`grain/drag coulomb/self`) — DUSTT
  closure with Hutchinson-fit lnLambda
- electrostatic breakup (DUSTT critical potential; `tensile_Pa`)
- grain-wall restitution (`surf_collide grainbounce`)
- emit midpoint hack removed (positions uniform along segments)
- `grain/*` style aliases; grain sources deduplicated into `src/`
- docs/fixes/*.md rewritten against actual syntax

Resolved 2026-07-28 (late): sheath + fix adapt now work together — fix adapt
already reallocates per-grid computes via grid->notify_changed(), and the
mover now re-invokes the sheath geometry compute per step so it recomputes
after refinement instead of silently switching off. The earlier "bus error"
was the fix-balance-after-adapt bug, not sheath.

Still open (lower priority):
- T-dependent material properties (Cp(T), emissivity(T))
- flow-shifted OML ion current in charging (F_Gamma(u))
- grain sputtering as a mass-loss channel; vapor shielding
- 3D support in grain/ablate; per-size grain distributions in emit
