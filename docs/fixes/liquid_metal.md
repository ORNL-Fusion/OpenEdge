# Liquid-metal divertor surface model

OpenEdge models a liquid-lithium (Li) coating on divertor tiles in
three pieces:

| Piece | Role |
|---|---|
| `fix surface/state/lm` | Smolentsev MHD strip solver. Drives surface temperature `Tsurf_lm`, film thickness `h_lm`, and incident D⁺ flux `Gamma_D_lm` per surf custom. |
| `compute surface/chemical/evaporation` | Reads `Tsurf_lm`; emits the Antoine + Hertz–Knudsen evaporation flux per surf. |
| `compute surface/chemical/adatom` | Reads `Tsurf_lm` + `Gamma_D_lm`; emits the Arrhenius ad-atom flux per surf. |
| `fix surface/emit/source` | Consumes the evap / adatom / sputter computes to launch Li atoms. |

The strip solver is the Fortran code by Sergey Smolentsev (UCLA),
ported into `src/OPENEDGE/liquid_metal_strip.h` (no SPARTA dependency
— callable standalone from `tools/lm/run_strip.cpp`).

## `fix surface/state/lm`

```
fix ID surface/state/lm <surf_group> <Nevery> <hf_source> [<hf_arg> ...] \
    h0 VAL U0 VAL Bs VAL alpha VAL width VAL Tin VAL \
    [Bw VAL] [sigma_w VAL] [tw VAL] [qss VAL] \
    [Nx VAL] [Ny VAL] [ncase VAL] \
    [max_iter VAL] [eps VAL] [relax VAL] [dt VAL] \
    [static yes|no]
```

### Heat-flux sources

| `hf_source`     | extra arg(s)                | meaning |
|-----------------|-----------------------------|---------|
| `c_compute[col]`| —                           | per-surf compute column |
| `f_fix[col]`    | —                           | per-surf fix column |
| `<value>`       | —                           | uniform constant W/m² |
| `plasma`        | `<plasma_fix>`              | reads parallel heat flux from `fix background` |
| `background`    | `<plasma_fix>`              | same as `plasma` (alias) |
| `target`        | `<file.h5> <leg>`           | precomputed q_target.h5 with `inner` / `outer` / `iu` / `ou` group |
| `solps_b2pl`    | `<ld_tg_*.dat>`             | SOLPS b2plot target loading file (Wtot column 5) |

`solps_b2pl` is the path used in the validated lower-divertor coupling
case — it also picks up `gamma_D` from the same file so adatom flux
computes correctly. See
`examples/test_solps_coupling/coupled_test/openedge/in.coupled` for a
two-leg deployment.

### Per-surf customs written

| name           | meaning                          |
|----------------|----------------------------------|
| `Tsurf_lm`     | surface temperature [°C]         |
| `h_lm`         | film thickness [m]               |
| `Gamma_D_lm`   | incident D⁺ flux at surf [m⁻²·s⁻¹] (zero unless `hf_source` carries it) |

These are SPARTA per-surf custom attributes, accessible from any
compute / dump as `s_<name>`.

### Multi-leg setup

The same custom names (`Tsurf_lm`, `h_lm`, `Gamma_D_lm`) can be shared
across multiple `fix surface/state/lm` instances — each writes only
the surfs in its own group, leaving the others untouched. This is how
the SOLPS coupling deck handles the inner and outer divertor legs:

```
group               div_il  surf id <> 103 116
group               div_ol  surf id <> 137 150

fix flm_ol surface/state/lm div_ol 100 solps_b2pl ../solps/.../ld_tg_o.dat \
    h0 5e-3 U0 8.0 Bs 5.0 alpha 43.0 width 1.67 Tin 350 static yes
fix flm_il surface/state/lm div_il 100 solps_b2pl ../solps/.../ld_tg_i.dat \
    h0 5e-3 U0 8.0 Bs 5.0 alpha 43.0 width 1.67 Tin 350 static yes
```

`add_custom` is idempotent on name reuse (second caller gets the
existing index). Each fix only touches its own group's slots, so the
two legs coexist. Surfs outside both groups keep the default values
(`Tin`, 0).

### `static yes` vs `static no`

- `static yes`: solve the strip once at `init()`, leave per-surf
  values fixed thereafter. Right choice when the heat-flux source is
  itself static (SOLPS plasma in `static yes` mode).
- `static no`: re-solve every `Nevery` steps. Use when a kinetic
  pusher is updating the wall heat flux dynamically.

## `compute surface/chemical/evaporation`

```
compute ID surface/chemical/evaporation <surf_group> tsurf <attr> [sticking VAL]
```

Reads the surf custom named `<attr>` (typically `Tsurf_lm`) and returns
the Antoine + Hertz–Knudsen evaporation flux. The vapor pressure is

$$
P_\mathrm{vap}(T) = 10^{A - B/T}\, P_0 ,
\qquad P_0 = \SI{101325}{\pascal},
$$

with Antoine coefficients $A = 5.66797$, $B = \SI{8310.41}{\kelvin}$
baked in for Li (valid for $T \in [298, 1600]\,\mathrm{K}$). The
Hertz–Knudsen flux follows:

$$
\Gamma_\mathrm{evap}(T) = \alpha_s \,\frac{P_\mathrm{vap}(T)}
                                {\sqrt{2 \pi\, m_\mathrm{Li}\, k_B\, T}} ,
$$

where $\alpha_s$ is the sticking coefficient (default 1.0).

## `compute surface/chemical/adatom`

```
compute ID surface/chemical/adatom <surf_group> tsurf <attr> dp_flux <source> \
    [Yad VAL] [Yad_Yps VAL] [E_eff VAL] [A_arr VAL]
```

Arrhenius desorption flux driven by the D⁺ flux $\Gamma_D$:

$$
\Gamma_\mathrm{ad}(T, \Gamma_D) =
  \frac{Y_\mathrm{ad}/Y_\mathrm{ps}}
       {1 + A_\mathrm{arr}\,\exp\!\left(E_\mathrm{eff}/k_B T\right)}\,
  \Gamma_D .
$$

`dp_flux` may be `c_<id>[col]`, `f_<id>[col]`, or `s_<surf-custom>` —
the latter is what reads `Gamma_D_lm` when `surface/state/lm` runs in
`solps_b2pl` mode.

| keyword | meaning | default |
|---------|---------|---------|
| `Yad`   | ad-atom yield for D on Li | 1e-3 |
| `Yad_Yps` | Yad / Yps ratio | 1.0 |
| `E_eff` | effective binding energy [eV] | 0.9 |
| `A_arr` | Arrhenius pre-factor | 1e-7 |

## End-to-end example (excerpt from `in.coupled`)

```
fix flm_ol surface/state/lm div_ol 100 solps_b2pl ../solps/.../ld_tg_o.dat \
    h0 5e-3 U0 8.0 Bs 5.0 alpha 43.0 width 1.67 Tin 350 static yes
fix flm_il surface/state/lm div_il 100 solps_b2pl ../solps/.../ld_tg_i.dat \
    h0 5e-3 U0 8.0 Bs 5.0 alpha 43.0 width 1.67 Tin 350 static yes

compute cevap surface/chemical/evaporation divertor tsurf Tsurf_lm sticking 1.0
compute cadat surface/chemical/adatom      divertor tsurf Tsurf_lm dp_flux s_Gamma_D_lm

fix femit_evp surface/emit/source LiSource divertor cevap perspecies no normal yes \
    nlaunch_total 200 model thermal_tsurf Tsurf_lm
fix femit_ada surface/emit/source LiSource divertor cadat perspecies no normal yes \
    nlaunch_total 200 model thermal_tsurf Tsurf_lm

dump d_surf surf all 1 output/surf_fluxes.txt id v1x v1y v2x v2y \
    c_cpmiLi c_cevap c_cadat s_Tsurf_lm s_Gamma_D_lm
```

Validated against the standalone Smolentsev strip solver
(`tools/lm/run_strip`) on both legs of the SOLPS coupling case:
$\Gamma_\mathrm{evap}$ matches to $0.00\,\%$, $\Gamma_\mathrm{ad}$ to
$\le 0.3\,\%$ on every wetted surf.

```{figure} ../_static/figures/lm_validation_solps.png
:alt: OpenEdge vs standalone strip-solver comparison on inner / outer divertor legs
:width: 100%

OpenEdge `surface/state/lm` reproduces the Smolentsev strip solver on
both lower-divertor legs. Top row: surface temperature
$T_\mathrm{surf}(l - l_\mathrm{sep})$. Middle row: evaporation flux.
Bottom row: ad-atom flux. Solid blue is the standalone strip solver
fed the same SOLPS heat-flux profile; markers are OpenEdge per-surf
custom values from `dump surf`.
```

## Files

| file | role |
|------|------|
| `src/OPENEDGE/liquid_metal_strip.h` | standalone Smolentsev solver (no SPARTA dep) |
| `src/OPENEDGE/fix_surface_state_lm.{h,cpp}` | SPARTA fix wrapper |
| `src/OPENEDGE/compute_surface_chemical_evaporation.{h,cpp}` | Antoine + HK |
| `src/OPENEDGE/compute_surface_chemical_adatom.{h,cpp}` | Arrhenius |
| `tools/lm/run_strip.cpp` | C++ standalone driver for the solver |
| `tools/lm/plot_openedge_vs_standalone.py` | OpenEdge vs strip-solver comparison |
| `tools/lm/plot_openedge_lm.py` | OpenEdge-only T_surf + flux quad |
| `tools/lm/plot_wall_tsurf.py` | (R, Z) wall view of T_surf and Γ_D |
