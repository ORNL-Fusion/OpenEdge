# `fix liquid_metal` — MHD shallow-water liquid-metal film

Liquid metal (Li) film solver for divertor surfaces. Computes surface
temperature, evaporation flux (Antoine + Hertz-Knudsen), ad-atom flux
(Arrhenius desorption), and film thickness as per-surface custom attributes.

Based on Fortran code by Sergey Smolentsev (UCLA).

## Syntax

```
fix ID liquid_metal group Nevery hf_source \
    h0 VAL U0 VAL Bs VAL alpha VAL width VAL Tin VAL \
    [dp_flux SOURCE] [Yad VAL] [Yad_Yps VAL] \
    [E_eff VAL] [A_arr VAL] \
    [Bw VAL] [sigma_w VAL] [tw VAL] [qss VAL] \
    [Nx VAL] [Ny VAL] [ncase VAL] \
    [max_iter VAL] [eps VAL] [relax VAL] [dt VAL] \
    [evap yes|no]
```

## Heat flux source

`hf_source` can be `c_compute[col]`, `f_fix[col]`, or a constant value
[W/m²].

## Solver

Strip solver (Smolentsev): coupled momentum, continuity, free surface, and
heat transfer on a 1D strip. MHD drag via Hartmann braking. Pseudo-time
iteration to steady state.

## Evaporation (Antoine + Hertz-Knudsen)

```
P_vapor = 10^(A - B/T_K) * 101325  [Pa]     (Antoine fit, Li 298–1600 K)
Γ_evap  = α · P / sqrt(2π m_Li kB T)        [atoms/m²/s]
Q_vapor = H_vap · Γ_evap / N_A              (evaporative cooling)
```

## Ad-atom (Arrhenius desorption driven by D+ flux)

```
Γ_ad = f_ad · (Yad/Yps) / (1 + A_arr · exp(E_eff / kB·T)) · Yad · Γ_D+
```

Requires D+ ion flux via `dp_flux` keyword (`c_compute[col]`, `f_fix[col]`,
or constant). If `dp_flux` is not specified, ad-atom flux is zero
(evaporation-only mode).

Ad-atom parameters:

| keyword | meaning | default |
|---|---|---|
| `Yad` | ad-atom yield for D on Li | 1e-3 |
| `Yad_Yps` | ratio Yad/Yps | 1.0 |
| `E_eff` | effective binding energy [eV] | 0.9 |
| `A_arr` | Arrhenius pre-factor | 1e-7 |

## Per-surface output columns (`f_ID[i][col]`)

1. `Tsurf_lm` — surface temperature [°C]
2. `evap_lm` — evaporation flux [atoms/m²/s]
3. `adatom_lm` — ad-atom flux [atoms/m²/s]
4. `h_lm` — film thickness [m]

## Key strip parameters

| keyword | meaning | default |
|---|---|---|
| `h0` | initial film thickness [m] | 0.005 |
| `U0` | inlet velocity [m/s] | 8.0 |
| `Bs` | streamwise magnetic field [T] | 5.0 |
| `Bw` | wall-normal magnetic field [T] | 0.0 |
| `alpha` | inclination angle [°] | 43 |
| `width` | channel half-width [m] | 1.67 |
| `Tin` | inlet temperature [°C] | 350 |
| `qss` | heat flux scale [W/m²] | 1e6 |
| `ncase` | 1 = sidewall (Hartmann via Bs), 2 = axisymmetric (via Bw) | — |

## Files

- `liquid_metal_strip.h` — standalone solver, no SPARTA dependency.
- `fix_liquid_metal.{h,cpp}` — SPARTA fix wrapper.

## Examples

```
# constant 2 MW/m² heat flux, no ad-atoms
fix flm liquid_metal wall 100 2.0e6 \
    h0 0.005 U0 8.0 Bs 5.0 alpha 43 width 1.67 Tin 350

# with D+ flux from compute for ad-atom calculation
fix flm liquid_metal wall 100 c_hflux[2] \
    h0 0.005 U0 8.0 Bs 5.0 alpha 43 width 1.67 Tin 350 \
    dp_flux c_pflux[3] Yad 1e-3 E_eff 0.9

# access outputs for dump
dump dsurf surf all 1000 surf.*.dat id f_flm[1] f_flm[2] f_flm[3] f_flm[4]
```
