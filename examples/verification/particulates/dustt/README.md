# DUSTT grain-model unit test

One-to-one grain-chain benchmark against DUSTT (Pigarov PoP 12, 122508)
in a uniform prescribed plasma (`fix pd background constant`, B_z = 1 T)
— no SOLPS input, no geometry, so any deviation is the model
implementation itself.

## Run

```bash
./run.sh    # runs all sub-cases + check_dustt.py; exit 0 = PASS
```

Sub-cases: `drag` (OML charging + ion friction vs python integration),
`efield` (F = Z_d eE vs analytic), `free` (centrifugal kinematics),
`neut` (neutral drag), `see` (secondary electron emission), `vac`
(fixed-charge vacuum acceleration, gates the no-drag fallback), and a
2-rank MPI-invariance repeat of `drag`.

The integration-level counterpart (same fixes, real CAT geometry and
SOLPS background) is `../droplet_transport/`.
