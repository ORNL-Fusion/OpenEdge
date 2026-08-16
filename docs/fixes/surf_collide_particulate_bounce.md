# `surf_collide particulate/bounce`

Apply sticking or restitution when a finite-size particulate strikes a wall.
Particles without a positive radius are removed by this collision style.

## Syntax

```text
surf_collide ID particulate/bounce pstick pvel pdiff \
    [pmass F] [ptemp F]
```

- `pstick` is the probability that an impacting particulate is removed.
- `pvel` is the retained speed fraction for a reflected particulate.
- `pdiff` is the probability of diffuse cosine-law reflection; otherwise the
  reflection is specular.
- `pmass` is the retained radius fraction. Mass is scaled by `pmass^3`.
- `ptemp` multiplies the particulate temperature after reflection.

`pstick`, `pvel`, and `pdiff` must lie in `[0, 1]`; `pmass` must lie in
`(0, 1]`; and `ptemp` must be positive.

## Example

```text
surf_collide particulate_wall particulate/bounce 0.7 0.4 0.5 \
             pmass 0.9 ptemp 0.8
surf_modify wall collide particulate_wall
```
