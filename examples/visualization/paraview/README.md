# Native ParaView output

This small particle-flow case writes the OpenEdge grid, embedded surface, and
particles directly as ParaView-readable VTK XML files. An N/O stream enters
through the left boundary with `fix emit/face`, reflects specularly from the
circular obstacle, and exits through the right boundary. The case is
deliberately collisionless because its purpose is demonstrating native VTK
output. There is no Python conversion or Catalyst step.

## Build

The executable must include the VTK package:

```bash
cmake -C ../OpenEdge/cmake/presets/mpi.cmake ../OpenEdge/cmake \
  -DPKG_OPENEDGE=ON -DPKG_VTK=ON
cmake --build . -j 8
```

## Run

```bash
./run.sh /path/to/spa_mpi             # 100 steps, 10 output intervals
RUN_STEPS=1000 ./run.sh               # 1000 steps, 10 output intervals
```

With no path argument, `run.sh` searches common build directories under
`$HOME` and selects an executable that advertises the native VTK dump styles.
An explicitly supplied executable is checked before the simulation starts.

The default is 100 steps. `RUN_STEPS` changes the duration, and `DUMP_EVERY`
can set the output interval explicitly. The runner clears older VTU snapshots
first so ParaView does not combine files from runs with different intervals.

Set `NP` to exercise the same input with more MPI ranks:

```bash
NP=4 ./run.sh /path/to/spa_mpi
```

The case writes:

- `output/grid_*.vtu`: the 2D grid cells
- `output/surface_*.vtu`: the circular obstacle
- `output/particles_*.vtu`: particle positions, IDs, species types, and
  velocity vectors

Open the three file series directly in ParaView. The `*` in each filename is
replaced by the timestep. ParaView exposes particle species as `type` and the
grouped `(vx,vy,vz)` components as the vector field `v`.

## View in ParaView

1. Choose **File > Open** and select `grid_0.vtu`, `surface_0.vtu`, and
   `particles_0.vtu`. ParaView recognizes each numbered set as a time series.
2. Click **Apply** for each source.
3. Select the particle source, use **Point Gaussian** representation, and
   color by `type` or `v`.
4. Press **Play** to animate the inlet stream around the circle.

The same native interface can export simulation fields, for example:

```text
dump dgrid grid/vtk all 100 output/grid_*.vtu id c_fields[*]
dump dsurf surf/vtk wall 100 output/wall_*.vtu id f_tally[*]
dump dpart particle/vtk all 100 output/particles_*.vtu id type x y z
```
