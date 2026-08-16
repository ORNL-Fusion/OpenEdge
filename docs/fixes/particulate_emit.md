# `fix particulate/emit`

Inject finite-size particulates from a selected surface group. Radius, mass,
and initial temperature come from the particulate species definition.

## Syntax

```text
fix ID particulate/emit MIXTURE_ID SURFACE_GROUP \
    [nevery N] [n N|v_name] [perspecies yes|no] [region REGION_ID] \
    [normal yes|no] [magVelocity V | vmin VMIN vmax VMAX] \
    [incidentAngle DEG | cone HALF_DEG | angle cosine|uniform] \
    [dir dR dZ dphi] [planar yes|no] [nweight W]
```

The command also accepts `subsonic`, `twopass`, and per-surface `custom`
source controls described by the core surface-emission machinery.

## Source rate

- `nevery` is the emission cadence; the default is `1`.
- `n N` requests a fixed total number of macroparticles per emission event,
  distributed by emitting area. Use `perspecies no` with a fixed count.
- `n v_name` evaluates an equal-style variable for the event count.
- `n 0` uses the mixture's flow-based source rate.
- `nweight W` declares that one emitted macroparticle represents `W` physical
  particulates. Thermal vapor production and cumulative atom tallies are
  scaled by this value; single-particulate trajectory dynamics are not.

## Launch velocity

- `magVelocity V` sets one launch speed.
- `vmin` and `vmax` sample speed uniformly over the supplied interval. Set
  both bounds.
- `incidentAngle` fixes the polar angle from the launch axis.
- `cone` samples uniformly in solid angle inside a cone.
- `angle cosine` samples a Knudsen/Lambert cosine distribution;
  `angle uniform` samples polar angle uniformly. The default is `uniform`.
- Angular priority is `incidentAngle`, then `cone`, then `angle`.
- `dir dR dZ dphi` replaces the local surface normal with a fixed physical
  launch axis. This is useful for a gravity-fed dropper.
- `planar yes` removes the out-of-plane launch component in 2D. With `dir`,
  `dphi` must be zero.

Emission positions are sampled uniformly along a 2D segment or over a 3D
surface element.

## Example

```text
fix source particulate/emit particulates injector \
    nevery 1 n 10 perspecies no \
    vmin 0.1 vmax 2.0 angle cosine planar yes \
    dir 0.0 -1.0 0.0
```
