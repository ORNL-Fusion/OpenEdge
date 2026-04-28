# `fix droplet/charge` — OML / thermionic dust charging

Updates each droplet's charge per timestep using the orbital-motion-
limited (OML) collection of plasma electrons + ions, with optional
thermionic emission from the heated droplet surface. The droplet
charge is stored in a per-particle custom and consumed by
`fix droplet/drag` and the Boris pusher (Lorentz force).

## Syntax

```
fix ID droplet/charge <Nevery> background <plasma_fix> \
    [ion_mass_amu <amu>] \
    [thermionic yes|no \
        [richardson_A <A>] [work_function_eV <W>]] \
    [radius <r>] [mass <m>] [temp <T>]
```

| keyword | meaning | default |
|---|---|---|
| `background <fix_id>` | plasma source for $T_e$, $n_e$, $T_i$, $n_i$ | required |
| `ion_mass_amu <amu>` | background ion mass for OML ion saturation | 2.014 (D) |
| `thermionic yes` | enable Richardson–Dushman emission from `T_droplet` | `no` |
| `richardson_A` | Richardson constant [A·m⁻²·K⁻²] | 1.2e6 |
| `work_function_eV` | metal work function | 2.5 (Li) |
| `radius`, `mass`, `temp` | per-particle custom names overriding the defaults | `radius`, `mass`, `temp` |

## Physics

OML balance solved per particle:

$$
I_e + I_i + I_\mathrm{th} = 0
\quad\Longrightarrow\quad
\phi_d(t) ,
$$

with electron, ion, and thermionic currents

$$
I_e   = -e\,n_e\,\sqrt{\tfrac{T_e}{2\pi m_e}}
        \exp(e\phi_d/T_e)\,(4\pi r^2),
\qquad
I_i   = +e\,n_i\,\sqrt{\tfrac{T_i}{2\pi m_i}}
        (1 - e\phi_d/T_i)\,(4\pi r^2),
$$

$$
I_\mathrm{th} = e\,A\, T_d^2\, \exp(-W/T_d)\,(4\pi r^2).
$$

Updated charge feeds the Boris Lorentz force at the next subcycle.

## Example

```
fix pd      background file plasma.h5 static yes
fix fcharge droplet/charge 1 background pd ion_mass_amu 2.0 thermionic no
```

## Files

- `src/OPENEDGE/fix_droplet_charge.{h,cpp}`
- See `examples/test_droplet_charging` for an OML validation case.
