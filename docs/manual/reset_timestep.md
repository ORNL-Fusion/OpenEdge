<!-- Adapted from the SPARTA manual (reset_timestep.txt); Plimpton, Gallis et al., Sandia National Laboratories. OpenEdge extensions marked. -->

# reset_timestep command

## Syntax

```
reset_timestep N
```

N = timestep number :ul

## Examples

```
reset_timestep 0
reset_timestep 4000000
```

## Description

Set the timestep counter to the specified value.  This command normally comes after the timestep has been set by reading a restart file via the read_restart command, or a previous simulation advanced the timestep.

The create_box command sets the timestep to 0; the read_restart command sets the timestep to the value it had when the restart file was written.

[Restrictions:] none

This command cannot be used when any fixes are defined that keep track of elapsed time to perform certain kinds of time-dependent operations. Examples are the fix ave/time, "fix ave/grid"_fix_ave_grid, and fix ave/surf commands.  Thus these fixes should be specified after the timestep has been reset.

Resetting the timestep clears flags for computes that may have calculated some quantity from a previous run.  This means these quantity cannot be accessed by a variable in between runs until a new run is performed.  See the variable command for more details.

[Related commands:] none

[Default:] none

