# Input Layout

This example keeps `input/` organized by role instead of file extension.

- `data/`: physics inputs consumed directly by the run, such as `plasma.h5`, `bfield.h5`, `mesh.extra`, `c.species`, and `.equ` files
- `geometry/`: surface and mesh assets used by the case, including `.surf` and `.stl`
- `generated/`: reproducible generated outputs such as metadata JSON and SPARTA include files
- `scripts/`: geometry builders and conversion helpers
- `plots/`: plotting and inspection utilities

The main case input file [`in.input`](/Users/42d/OpenEdge/examples/test_d3d_walldyn/in.input) has been updated to use these subfolders.
