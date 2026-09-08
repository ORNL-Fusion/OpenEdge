# Native ParaView output

This small zero-step case writes the OpenEdge grid and embedded surface
directly as ParaView-readable VTK XML files. It uses the production pattern:
native `grid/vtk` and `surf/vtk` dumps, `dump_modify ... first yes`, then
`run 0`. There is no Python conversion or Catalyst step.

## Build

The executable must include the VTK package:

```bash
cmake -C ../OpenEdge/cmake/presets/mpi.cmake ../OpenEdge/cmake \
  -DPKG_OPENEDGE=ON -DPKG_VTK=ON
cmake --build . -j 8
```

## Run

```bash
./run.sh /path/to/spa_mpi
```

Set `NP` to exercise the same input with more MPI ranks:

```bash
NP=4 ./run.sh /path/to/spa_mpi
```

The case writes:

- `output/grid_0.vtu`: the 2D grid cells
- `output/surface_0.vtu`: the circular obstacle

Open either file directly in ParaView. For a time-dependent simulation,
change the dump interval and advance more than zero steps; the `*` in each
filename is replaced by the timestep.

The same native interface can export simulation fields, for example:

```text
dump dgrid grid/vtk all 100 output/grid_*.vtu id c_fields[*]
dump dsurf surf/vtk wall 100 output/wall_*.vtu id f_tally[*]
dump dpart particle/vtk all 100 output/particles_*.vtu id type x y z
```
