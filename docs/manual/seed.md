<!-- Adapted from the SPARTA manual (seed.txt); Plimpton, Gallis et al., Sandia National Laboratories. OpenEdge extensions marked. -->

# seed command

## Syntax

```
seed Nvalue
```

Nvalue = seed for a random number generator (positive integer) :ul

## Examples

```
seed 5838959
```

## Description

This command sets the random number seed for a master random number generator.  This generator is used by OpenEdge to initialize auxiliary random number generators, which in turn are used for all operations in the code requiring random numbers.  This means you can effectively run a statistically-independent simulation by simply changing this single seed.

The various random number generators used in OpenEdge are portable, which means they produce the same random number streams on any machine.

This command is required to perform a OpenEdge simulation.

[Restrictions:] none

[Related commands:] none

[Default:] none

