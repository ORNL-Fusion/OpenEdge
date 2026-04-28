# IEAD database — ion energy / angle distributions for sputter yield

This database holds per-cell 2-D ion-energy / angle distributions
(IEADs) at a wall, parameterised by the dimensionless plasma /
projectile state. It is consumed by
`compute surface/physical/sputter` to fold the **distribution** of
incident ions against the Eckstein yield curve, replacing the
mean-impact approximation `Y(<E>, <theta>)` with

```
<Y>_element = sum_{Et, theta} Y_element(Et*Z*Te, theta)
              f(Et, theta; tau, psi, Z) dEt dtheta
```

with `Et = E / (Z * Te)` the projectile-sonic-normalised energy. This
normalization follows Section 4 of `docs/iead_normalization.tex` (in
the SHEATH repo) — choosing `v_star = c_s,proj = sqrt(Z*Te/m_proj)`
collapses the dimensionless EOM coefficient of `-grad(phi)` to unity
and unifies the Et range across Z slots.

## File layout

```
database/iead/
  iead_database.h5                 5-D table f(tau, psi, Z, Ehat, theta)
  README.md                        this file
  IEAD_REGENERATE.md               step-by-step rebuild instructions
  Mellet_2017_PPCF_59_035006.pdf   reference paper (place here when vendored)
```

## HDF5 schema

```
/iead_database.h5
    /tau_grid              shape (n_tau,)        Ti / Te
    /psi_grid_deg          shape (n_psi,)        magnetic angle from
                                                 wall normal (deg)
    /Z_grid                shape (n_Z,)          projectile charge
    /Etilde_bin_edges      shape (n_E + 1,)      E / (Z*Te) bin edges
    /theta_bin_edges_deg   shape (n_theta + 1,)  angle-from-normal edges
    /m_proj_per_Z_amu      shape (n_Z,)          m_proj used per Z slot
    /f                     shape (n_tau, n_psi, n_Z, n_E, n_theta)
                                                 per-cell PDF, sum=1
    Te_eV, ne_m3, Bmag_T, mi_bg_amu              sweep conditions (attrs)
    NP_per_species                               particles per cell (attr)
    energy_axis                                  "Etilde = E / (Z * Te)"
    scheme                                       projectile-mass scheme
```

## Parameter coverage (current build)

| axis   | grid                              | rationale |
|--------|-----------------------------------|-----------|
| tau    | `geomspace(0.3, 5.0, 8)`          | Ti/Te in SOL/divertor |
| psi    | `90 - geomspace(0.5, 30, 8)[::-1]`| refined near grazing |
| Z      | `1..10`                            | covers D+, Ne+1..+10 (W table separate, see below) |
| Etilde | uniform `linspace(0, 60, 120+1)`  | 0.5 / bin; covers p99.9 over the whole grid |
| theta  | uniform `linspace(0, 90, 30+1)`   | 3-deg bins |

Fixed during the sweep: `Te = 40 eV`, `ne = 1e19 m^-3`, `|B| = 3 T`,
`m_i_bg = m_D = 2.014 amu`. The dimensionless ratio
`rho_i / lambda_D` sits near ~40 across typical SOL/divertor
conditions, so the IEAD shape in `(Ehat, theta)` is approximately
universal in `(tau, psi, Z)`.

## Projectile mass scheme — option 1 (no element axis, light table)

The IEAD shape in `(Etilde, theta)` is set by the dimensionless plasma
/ sheath state `(tau, psi)` and the projectile **charge** `Z`. The mass
ratio `mu_m = m_proj / m_bg` enters only via `Omega = sqrt(Z/mu_m)`
(Larmor rotation in the MPS) and `sqrt((1+tau)*mu_m/(2Z))` (entry drift
in projectile-sonic units). For elements up to Ne we follow Mellet 2017
and drop the element axis:

- During generation, `m_proj = 2 * Z amu` (D-scaling).
- At run time the element-specific Eckstein yield
  `Y_element(E_eV, theta)` is convolved against
  `f(Etilde, theta; tau, psi, Z)` with `E_eV = Etilde * Z * Te_local`.

Bias from this approximation: <15% for D, Li, B, O, Ne.

## Tungsten — separate table (Path 3, follow-on)

`m_proj = 184 amu` for W means `mu_m = 92` on D, breaking the D-scaling
assumption. Plan: a second HDF5 table at the same `(tau, psi)` grid but
`Z = 1..20` and `m_proj = 184 amu` for every Z slot. Consumer side
selects light-table vs W-table by element. ~160 cells, ~30 min of
sheath_tracker_v2 runtime; not yet generated.

## How `compute surface/physical/sputter` uses it

```
compute cpmiLi surface/physical/sputter divertor background pd \
    target Li projectiles D,Ne,Li \
    iead auto                                          # <-- new option
```

- `iead auto`  : resolve via `database_paths.h::resolve_iead_file()`
  (looks under `${OPENEDGE_ROOT}/database/iead/iead_database.h5`,
  or the compile-time database dir, or `database/` cwd-relative).
- `iead <path>`: explicit absolute path.
- absent       : fall back to mean-impact `Y(<E>, <theta>)`.

At init: precompute `<Y>_element(tau_local, psi_local, Z, surf_idx)`
per surf for each projectile-element table. The convolution uses
`E_eV = Etilde * Z * Te_local` to map the dimensionless axis to the
yield curve before integrating. With `static yes`, the inner per-step
loop does a single multiply per slot.

## References

- B. Mellet, K. Bystrov, *Energy and angle distributions of ions
  impacting on a wall in a magnetised plasma*, Plasma Phys. Control.
  Fusion 59 (2017) 035006.
- A.V. Chankin et al., *On the energy of ions striking the divertor
  target*, Plasma Phys. Control. Fusion 56 (2014) 025003.
- W. Eckstein, *Sputtering Yields*, in *Sputtering by Particle
  Bombardment*, Topics in Applied Physics 110 (2007) 33.

## Regeneration

See `IEAD_REGENERATE.md`.
