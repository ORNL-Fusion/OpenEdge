<!-- Adapted from the SPARTA manual (clear.txt); Plimpton, Gallis et al., Sandia National Laboratories. OpenEdge extensions marked. -->

# clear command

## Syntax

```
clear
```

## Examples

```
(commands for 1st simulation)
clear
(commands for 2nd simulation)
```

## Description

This command deletes all atoms, restores all settings to their default values, and frees all memory allocated by OpenEdge.  Once a clear command has been executed, it is almost as if OpenEdge were starting over, with only the exceptions noted below.  This command enables multiple jobs to be run sequentially from one input script.

These settings are not affected by a clear command: the working directory (shell command), log file status (log command), echo status (echo command), and input script variables (variable command).

[Restrictions:] none

[Related commands:] none

[Default:] none

