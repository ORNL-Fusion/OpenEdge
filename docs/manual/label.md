<!-- Adapted from the SPARTA manual (label.txt); Plimpton, Gallis et al., Sandia National Laboratories. OpenEdge extensions marked. -->

# label command

## Syntax

```
label ID
```

ID = string used as label name :ul

## Examples

```
label xyz
label loop
```

## Description

Label this line of the input script with the chosen ID.  Unless a jump command was used previously, this does nothing.  But if a jump command was used with a label argument to begin invoking this script file, then all command lines in the script prior to this line will be ignored.  I.e. execution of the script will begin at this line.  This is useful for looping over a section of the input script as discussed in the jump command.

[Restrictions:] none

[Related commands:] none

[Default:] none

