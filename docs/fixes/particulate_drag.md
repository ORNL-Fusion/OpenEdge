# `fix particulate/drag`

Apply particulate momentum transfer from background ions and neutrals, plus
optional electric and gravitational acceleration.

## Syntax

```text
fix ID particulate/drag Nevery A_bg Z_bg background BACKGROUND_ID \
    [mixture MIXTURE_ID] [material NAME] [model dustt2005] \
    [coulomb/self yes|no] \
    [coulomb/chi X] [coulomb/delta X] [coulomb/lnlambda X] \
    [neutrals yes|no] [efield yes|no] \
    [gravity gR gZ gphi] [radius R] [mass M] [temp T]
```

## Arguments

- `A_bg` and `Z_bg` are the background-ion mass in atomic mass units and
  charge state.
- `background` selects the local plasma and neutral provider.
- `mixture` limits the update to the named particulate mixture.
- `material` supplies the particulate density. The built-in default density
  is liquid lithium when no material is selected.
- `model dustt2005` explicitly selects the implemented collection and
  Coulomb-orbit drag formulation.
- `coulomb/self yes` derives the potential, `Ti/Te`, and Coulomb logarithm
  from the local plasma and `particulate_charge`. This is the default.
  With `no`, the three `coulomb/*` values are used.
- `neutrals yes` enables neutral friction when the background contains
  positive `nn` and `tn`. It automatically contributes zero otherwise.
- `efield yes` applies `Z_d e E/m_d` using the local electric field and
  particulate charge. Both `neutrals` and `efield` default to `yes`.
- `gravity` is specified in physical `(R, Z, phi)` components in m/s².
- `radius`, `mass`, and `temp` initialize missing particulate state.

## Integration

Ion and neutral drag are combined into one relaxation rate. The velocity is
advanced with the exact constant-coefficient exponential solution in two
half-kicks around particle transport. Gravity and electric acceleration are
included in the same update.

Do not apply gravity both here and through a separate gravity fix.

## Example

```text
fix fd particulate/drag 1 2.0 1 background pd \
       mixture particulates material Li \
       coulomb/self yes neutrals yes efield yes \
       gravity 0.0 -9.80665 0.0
```
