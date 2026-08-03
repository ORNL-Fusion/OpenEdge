# fix grain/emit (droplet/emit)

Surface-flux emission of grain macro-particles from a surf group
(SPARTA emit/surf plus grain options). Grain radius/temp/mass come from the
12-column species file.

    fix ID grain/emit MIXTURE surf-group [keywords]

Grain-specific keywords (beyond emit/surf's):

- `n N` + `nevery K` — N grains per batch every K steps.
- `vmin V vmax V` — uniform launch-speed sampling (bypasses Maxwellian).
- `angle cosine|uniform` — launch-angle law about the surface normal.
- `magVelocity V`, `incidentAngle deg` — fixed speed / polar angle.
- `nweight W` — one macro-grain represents W real grains (written to the
  per-particle custom `grain_nweight`; scales the vapor source and atom
  tally in grain/ablate). Mass rate: mdot = n·W·m_grain/(nevery·dt).

Emission position is uniform along each surf segment.
