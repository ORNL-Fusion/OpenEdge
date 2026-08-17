<!-- Adapted from the SPARTA manual (stats.txt); Plimpton, Gallis et al., Sandia National Laboratories. OpenEdge extensions marked. -->

# stats command

## Syntax

```
stats N
```

N = output statistics every N timesteps :ul

## Examples

```
stats 100
```

## Description

Compute and print statistical info (e.g. particle count, temperature) on timesteps that are a multiple of N and at the beginning and end of a simulation run.  A value of 0 will only print statistics at the beginning and end.

The content and format of what is printed is controlled by the stats_style and stats_modify commands.

The timesteps on which statistical output is written can also be controlled by a variable.  See the "stats_modify every"_stats_modify command.

[Restrictions:] none

## Related commands

stats_style, stats_modify

## Default

```
stats 0
```

