# `fix particulate/charge`

Calculate the charge of finite-size particulates from the local plasma
background. The resulting charge number is stored per particle and is
consumed by `fix particulate/drag` and `fix particulate/thermal`.

## Syntax

```text
fix ID particulate/charge Nevery background BACKGROUND_ID \
    [mixture MIXTURE_ID] [material NAME] \
    [ion_mass_amu M] [fixed_charge Zd] \
    [thermionic yes|no] [richardson_A A] [work_function_eV W] \
    [see yes|no] [validity warn|error|no] \
    [radius R] [mass M] [temp T]
```

## Arguments

- `Nevery` updates the charge at the start and end of matching timesteps.
- `background` selects the `fix background` providing local `ne`, `ni`,
  `Te`, `Ti`, flow, and magnetic field.
- `mixture` limits the calculation to species in the named mixture. Without
  it, every particle with a positive radius is eligible.
- `material` supplies thermionic and tensile-strength properties. Explicit
  `richardson_A` and `work_function_eV` values override the material.
- `ion_mass_amu` sets the background-ion mass; the default is `2.0`.
- `fixed_charge` writes the specified charge number and bypasses the OML
  solve. It is intended for controlled force and sensitivity tests.
- `thermionic yes` adds Richardson-Dushman thermionic emission with the
  finite-radius work-function correction. The default is `no`.
- `see yes` enables the experimental electron-impact secondary-emission
  channel. The default is `no`; the code reports its incomplete physics.
- `validity warn|error|no` controls the response when `R_d > lambda_D`.
  The default is `warn`.
- `radius`, `mass`, and `temp` initialize missing particulate state. Values
  already present on a particle are not replaced.

## Model

The floating potential is obtained from electron, flow-dependent ion, and
enabled emission currents. The charge number is

$$Z_d = \frac{4\pi\epsilon_0 R_d\phi_d}{e}$$

For materials with nonzero tensile strength, the model also applies the
configured electrostatic-disruption criterion and creates two
mass-conserving fragments when the critical potential is exceeded.

## Example

```text
fix pd background file input/plasma.h5 static yes
fix qd particulate/charge 1 background pd mixture particulates \
       material Li validity warn
```
