# `fix droplet/evaporate` — droplet mass loss + recoil thrust

Per-particle mass loss from evaporation (heat-flux driven) and
optional recoil thrust ("rocket effect") from the asymmetric vapour
flow. Updates each droplet's mass, radius, and velocity per timestep
using the local plasma heat flux from `fix background`.

## Syntax

```
fix ID droplet/evaporate <Nevery> <mix-ID> background <plasma_fix> \
    mass <m_attr> radius <r_attr> temp <T_attr> \
    heatflux/scale <s> [rocket_eta <eta>]
```

| keyword | meaning |
|---|---|
| `background <fix_id>` | plasma fix providing $T_e$, $n_e$, $V_\|$, $B$ at the droplet position |
| `mass`, `radius`, `temp` | per-particle custom names for droplet state |
| `heatflux/scale <s>` | fudge factor on the parallel heat flux to the droplet (default 1.0; `2.0` matches the plasma-electron heat-flux convention used in test cases) |
| `rocket_eta <eta>` | recoil-thrust efficiency, $0 \le \eta \le 1$. `0.5` is a common literature value. |

## Physics

Mass loss balances the absorbed plasma heat flux with the latent heat
of vaporisation $H_\mathrm{vap}$:

$$
\dot m = -\frac{\eta_q\, q_\|\, A_d}{H_\mathrm{vap}/m_\mathrm{Li}},
\qquad A_d = 4\pi r^2.
$$

If `rocket_eta > 0`, asymmetric vaporisation (more on the hot face)
gives the droplet a kick along the heat-flux direction:

$$
\Delta v_\mathrm{recoil} = \frac{\eta_\mathrm{rocket}}
                                {m_d}\, \dot m\, c_s ,
$$

with $c_s$ the local Bohm sound speed.

## Example

```
fix pd    background file plasma.h5 static yes
fix fevap droplet/evaporate 1 dropmix background pd \
    mass mass radius radius temp temp \
    heatflux/scale 2.0 rocket_eta 0.5
```

## Files

- `src/OPENEDGE/fix_droplet_evaporate.{h,cpp}`
- See `examples/test_evaporation`.
