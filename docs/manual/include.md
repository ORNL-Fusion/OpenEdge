<!-- Adapted from the SPARTA manual (include.txt); Plimpton, Gallis et al., Sandia National Laboratories. OpenEdge extensions marked. -->

# include command

## Syntax

```
include file
```

file = filename of new input script to switch to :ul

## Examples

```
include newfile
include in.run2
```

## Description

This command opens a new input script file and begins reading OpenEdge commands from that file.  When the new file is finished, the original file is returned to.  Include files can be nested as deeply as desired.  If input script A includes script B, and B includes A, then OpenEdge could run for a long time.

If the filename is a variable (see the variable command), different processor partitions can run different input scripts.

[Restrictions:] none

## Related commands

variable, jump

[Default:] none

