.. hand-edited: this page is maintained here, not generated from doc/*.txt

4. How-to discussions
=====================

This chapter explains how to set up the geometry, the plasma
background and the data files of an OpenEdge simulation, and how to
carry out common tasks inherited from SPARTA.

| 4.1 :ref:`2d simulations <howto_1>`
| 4.2 :ref:`Axisymmetric simulations <howto_2>`
| 4.3 :ref:`Plasma and magnetic-field backgrounds <howto_3>`
| 4.4 :ref:`The plasma.h5 file <howto_4>`
| 4.5 :ref:`Wall geometry and surface normals <howto_5>`
| 4.6 :ref:`The processes.h5 database <howto_6>`
| 4.7 :ref:`Running multiple simulations from one input script <howto_7>`
| 4.8 :ref:`Output from OpenEdge (stats, dumps, computes, fixes, variables) <howto_8>`
| 4.9 :ref:`Visualizing snapshots <howto_9>`
| 4.10 :ref:`Details of grid geometry <howto_10>`
| 4.11 :ref:`Details of surfaces <howto_11>`
| 4.12 :ref:`Restarting a simulation <howto_12>`
| 4.13 :ref:`Visualizing output with ParaView <howto_13>`
| 4.14 :ref:`Custom per-particle, per-grid, per-surf attributes <howto_14>`
| 4.15 :ref:`Variable timestep simulations <howto_15>`

----------

.. _howto_1:

4.1 2d simulations
------------------

The plasma background and the magnetic field are defined in the
cylindrical coordinates :math:`(R, \phi, Z)` of the device.  The
:doc:`dimension <dimension>` of the run and the
:doc:`boundary <boundary>` setting decide how the axes of the
simulation box map onto them:

.. list-table::
   :header-rows: 1
   :widths: 24 12 12 12 40

   * - Geometry
     - x
     - y
     - z
     - Cylindrical coordinates of a point
   * - 3d
     - Cartesian
     - Cartesian
     - :math:`Z`
     - :math:`R = [(x-x_0)^2 + (y-y_0)^2]^{1/2}`,
       :math:`\phi = \mathrm{atan2}(y-y_0,\, x-x_0)`, :math:`Z = z`
   * - 2d planar
     - :math:`R`
     - :math:`Z`
     - not resolved
     - :math:`R = x`, :math:`Z = y`
   * - 2d axisymmetric
     - :math:`Z`
     - :math:`R`
     - not resolved
     - :math:`R = y`, :math:`Z = x`

In 3d the symmetry axis is parallel to the z axis of the box and
passes through :math:`(x_0, y_0)`, which is the origin unless the
*column_axis* keyword of :doc:`fix background <fix_background>` moves
it.  In both 2d geometries the z component of a particle velocity is
its toroidal component :math:`v_\phi`.  The velocity components along
x and y are :math:`(v_R, v_Z)` in the planar geometry and
:math:`(v_Z, v_R)` in the axisymmetric one.  The same mapping applies
to the components of the electric and magnetic fields, of the
temperature gradients and of the forces.

A 2d simulation, as in SPARTA, moves the particles in the xy plane
only, but keeps all three velocity components.  The planar geometry
treats the poloidal plane as a slab: a grid cell has unit depth in z
and its volume does not grow with :math:`R`.  The background is still
evaluated at :math:`(R, Z) = (x, y)`, so the toroidal field varies as
:math:`1/x`.  The triple (x, y, z) = :math:`(R, Z, \phi)` is
left-handed.  The pushers allow for this, and the sign of
:math:`B_\phi` has the same meaning in all three geometries.

A planar 2d input script contains these settings:

* Use the :doc:`dimension <dimension>` command to specify a 2d
  simulation.

* Make the simulation box periodic in z via the
  :doc:`boundary <boundary>` command.  This is the default.

* Using the :doc:`create_box <create_box>` command, set the z
  boundaries of the box to values that straddle the z = 0.0 plane,
  i.e. zlo < 0.0 and zhi > 0.0.  Typical values are -0.5 and 0.5.
  Whatever the values, OpenEdge computes the volume of a 2d grid cell
  as if its length in z were 1.0.  This volume is used with the
  :doc:`global nrho <global>` setting to calculate the number of
  particles to create or insert, and to compute collision
  frequencies.

* If surfaces are defined via the :doc:`read_surf <read_surf>`
  command, use 2d objects made of line segments.

The *rz_axes* keyword of :doc:`fix background <fix_background>` set to
*axi* makes a planar 2d run use the axisymmetric assignment x =
:math:`Z`, y = :math:`R`, so that it can read the surface and plasma
files of an axisymmetric case unchanged.

----------

.. _howto_2:

4.2 Axisymmetric simulations
----------------------------

An axisymmetric simulation is a 2d simulation in which the x axis of
the box is the symmetry axis of the device and y is the major radius:
x = :math:`Z` and y = :math:`R`.  A particle keeps its three velocity
components :math:`(v_x, v_y, v_z) = (v_Z, v_R, v_\phi)`.  After each
move it is rotated back into the xy plane, so the toroidal motion
changes :math:`R` and the velocity components but not the position in
z.  The volume of a grid cell is the volume that its area sweeps out
when it is rotated around the y = 0 axis, and the area of a surface
line segment is the area of the corresponding ring.

An axisymmetric problem is set up with these commands:

* Set dimension = 2 via the :doc:`dimension <dimension>` command.
* Set the lower boundary of the y dimension to *a* via the
  :doc:`boundary <boundary>` command.  The upper y boundary can be
  anything except *a* or *p*.
* Use the :doc:`create_box <create_box>` command to define a 2d
  simulation box with ylo = 0.0.  The x extent of the box covers the
  range of :math:`Z` and the y extent covers the range of :math:`R`.

For example, a device with :math:`0 < R < 3.25` m and
:math:`-1 < Z < 0.85` m uses

.. code-block:: text

   dimension   2
   boundary    o ao p
   create_box  -1.0 0.85 0.0 3.25 -0.05 0.05

A wall file for this geometry lists each point as (:math:`Z`,
:math:`R`).  The plasma file is unaffected, because it stores its mesh
in :math:`(R, Z)`.

Grid cell weighting can be enabled via the
:doc:`global weight <global>` command with the *volume* or *radius*
setting.  It affects how many particles per grid cell are created by
the :doc:`create_particles <create_particles>` and
:doc:`fix emit <fix_emit_face>` command variants.  During a run it
also clones and deletes particles as they move from cell to cell, so
that cells of very different volume hold similar numbers of
particles.  This applies to every particle, whatever its species.
Do not use cell weighting in a simulation with particulates, because
a grain that crosses into a cell of different weight would be cloned
or deleted.  :doc:`fix particulate/emit <fix_particulate_emit>` stops
with an error if cell weighting is enabled.

----------

.. _howto_3:

4.3 Plasma and magnetic-field backgrounds
-----------------------------------------

OpenEdge does not solve for the plasma.  The electron and ion
densities and temperatures, the parallel flow, the electric and
magnetic fields and the temperature gradients are prescribed by one
:doc:`fix background <fix_background>` command.  The pusher and the
other OpenEdge commands refer to this fix by its ID and query it at
the position of each particle.

Kinds of background
^^^^^^^^^^^^^^^^^^^

The fix has two modes.

* *file* reads a plasma file.  The default format is the
  :ref:`plasma.h5 file <howto_4>`, which holds an axisymmetric
  solution on a triangulated mesh in the :math:`(R, Z)` plane.  The
  converters in tools/converters write it from SOLPS-ITER, SOLEDGE3X
  and OEDGE runs.  The format *zones3d* reads a three-dimensional
  solution on the curvilinear zones of EMC3-EIRENE and needs a 3d
  simulation.
* *constant* sets uniform values of the plasma quantities, of the
  gradients and of the magnetic field from keywords of the fix.  Most
  verification cases use it.

Sampling the plasma
^^^^^^^^^^^^^^^^^^^

For a plasma.h5 background the position of a particle is converted to
:math:`(R, Z)` as described in :ref:`Section 4.1 <howto_1>`, and the
triangle of the plasma mesh that contains the point is located.
Each triangle belongs to one cell of the code that produced the
background, and the plasma quantities are constant within a cell.  A
particle therefore sees the values of the cell it is in, without
interpolation between cells.  A point outside the triangulation gets
zero density and temperature, that is, vacuum, unless the file also
holds the plasma on a regular grid as described in
:ref:`Section 4.4 <howto_4>`.

For a *zones3d* background the point :math:`(R, Z, \phi)` is located
in the hexahedral cells of the zones.  The *outside* keyword of the
fix selects whether a point outside all zones stops the run or is
treated as vacuum.

The particle grid and the plasma mesh are independent.  The grid can
be refined near the wall or adapted during the run without any change
to the plasma data.

The parallel flow is stored as a signed speed :math:`u_\parallel`.
The flow velocity is :math:`u_\parallel \hat{\mathbf{b}}`, so its
direction follows the sign of the magnetic field in the file.

Magnetic field
^^^^^^^^^^^^^^

The fix takes the magnetic field from the first of these sources that
is available at the point:

#. the uniform Cartesian field of the *bcart* keyword;
#. the field stored at the vertices of the plasma mesh.  The value in
   a triangle is the mean of its three vertex values;
#. the equilibrium, on a regular :math:`(R, Z)` grid that normally
   extends beyond the plasma mesh;
#. the uniform cylindrical components of the *br*, *bz* and *bt*
   keywords.

The mesh field is piecewise constant and has no derivatives.  The
guiding-centre pusher needs :math:`\nabla B` and the curvature of the
field, so it asks for the equilibrium first and uses the mesh field
only where the equilibrium is not available.

The equilibrium gives the field in one of two ways.  If the file
holds maps of the three field components, they are interpolated
bilinearly, and the derivatives are those of the interpolant.
Otherwise the field is derived from the poloidal flux per radian
:math:`\psi(R, Z)`:

.. math::
   :label: howto-bpsi

   B_R = -\frac{1}{R}\frac{\partial\psi}{\partial Z}, \qquad
   B_Z = \frac{1}{R}\frac{\partial\psi}{\partial R}, \qquad
   B_\phi = \frac{B_\mathrm{tf} R_\mathrm{tf}}{R} ,

where :math:`B_\mathrm{tf}` is the toroidal field at the reference
radius :math:`R_\mathrm{tf}`.  The derivatives of :math:`\psi` are
central differences at the node of the equilibrium grid nearest to
the point.  The field derivatives follow from
Eq. :eq:`howto-bpsi`,

.. math::

   \frac{\partial B_R}{\partial R} &=
     \frac{1}{R^2}\frac{\partial\psi}{\partial Z}
     - \frac{1}{R}\frac{\partial^2\psi}{\partial R\,\partial Z}, \\
   \frac{\partial B_R}{\partial Z} &=
     -\frac{1}{R}\frac{\partial^2\psi}{\partial Z^2}, \\
   \frac{\partial B_Z}{\partial R} &=
     -\frac{1}{R^2}\frac{\partial\psi}{\partial R}
     + \frac{1}{R}\frac{\partial^2\psi}{\partial R^2}, \\
   \frac{\partial B_Z}{\partial Z} &=
     \frac{1}{R}\frac{\partial^2\psi}{\partial R\,\partial Z}, \\
   \frac{\partial B_\phi}{\partial R} &=
     -\frac{B_\mathrm{tf} R_\mathrm{tf}}{R^2}, \\
   \frac{\partial B_\phi}{\partial Z} &= 0 ,

with the second derivatives of :math:`\psi` from three-point
differences at the same node, and

.. math::

   \frac{\partial B}{\partial R} = \frac{1}{B}\left(
     B_R\frac{\partial B_R}{\partial R}
     + B_\phi\frac{\partial B_\phi}{\partial R}
     + B_Z\frac{\partial B_Z}{\partial R}\right) ,

and likewise for :math:`\partial B/\partial Z`.  In a 3d simulation
the cylindrical components are rotated to the box axes at the
toroidal angle of the particle:

.. math::

   B_x = B_R\cos\phi - B_\phi\sin\phi, \qquad
   B_y = B_R\sin\phi + B_\phi\cos\phi, \qquad
   B_z = B_Z .

The normalized poloidal flux used by
:doc:`fix reflect/psi <fix_reflect_psi>` and by the *pinch_psi*
option of
:doc:`fix cross_field_diffusion <fix_cross_field_diffusion>` is

.. math::
   :label: howto-psin

   \psi_N = \frac{\psi - \psi_\mathrm{axis}}
                 {\psi_\mathrm{sep} - \psi_\mathrm{axis}} ,

where :math:`\psi_\mathrm{axis}` is the flux on the magnetic axis and
:math:`\psi_\mathrm{sep}` the flux on the separatrix.  Here
:math:`\psi` is interpolated bilinearly on the equilibrium grid.  A
point outside the grid takes the value at the nearest point of its
edge.

The *equilibrium* keyword of the fix reads the equilibrium from a
separate file when the plasma file does not hold one.

A *zones3d* background stores the field strength and the direction of
the field line in each cell.  The *b_sign* keyword of the fix sets
the sign of the field along that direction.  This format provides no
field derivatives.

Where no file supplies the field, the components can be given per
particle with :doc:`fix bfield/particle <fix_bfield_particle>`.

Checking a background
^^^^^^^^^^^^^^^^^^^^^

:doc:`compute plasma/fields <compute_plasma_fields>` samples the
background quantities on the particle grid, and a grid
:doc:`dump <dump>` writes them out.  Plot them before a production
run to confirm that the background was read and is oriented as
intended, in particular after a change of geometry between planar and
axisymmetric.

----------

.. _howto_4:

4.4 The plasma.h5 file
----------------------

A plasma.h5 file is an HDF5 file that holds an axisymmetric plasma
solution, its magnetic equilibrium and its list of ion species.  The
converters in tools/converters write it:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Converter
     - Source
   * - convert_solps_plasma.py
     - SOLPS-ITER (b2fgmtry, b2fstate and the EIRENE triangulation)
   * - convert_s3x_plasma.py
     - SOLEDGE3X
   * - convert_oedge_plasma.py
     - OEDGE (NetCDF output and an equilibrium file)

Each converter prints its options with --help or documents them in
its header.  The plasma files of the example cases are either stored
with the case or produced by a converter command given in the README
of the case.

The tables below list the datasets that
:doc:`fix background <fix_background>` reads.  All values are double
precision unless marked as integer or string.  The dimensions are
:math:`N_R \times N_Z` nodes of the equilibrium grid, :math:`N_v`
vertices, :math:`N_t` triangles and :math:`N_c` cells of the plasma
mesh, :math:`N_i` ion species and :math:`N_w` elements of the wall
surface file.  Two-dimensional maps are stored with :math:`Z` as the
first index.  A file may hold further datasets, which the fix ignores.

Equilibrium
^^^^^^^^^^^

The equilibrium group is used when it holds at least *r*, *z* and
*psi*.  The grid must be uniform in :math:`R` and in :math:`Z`.

.. list-table::
   :header-rows: 1
   :widths: 30 16 12 42

   * - Dataset
     - Shape
     - Unit
     - Meaning
   * - /equilibrium/r
     - :math:`N_R`
     - m
     - :math:`R` of the grid nodes
   * - /equilibrium/z
     - :math:`N_Z`
     - m
     - :math:`Z` of the grid nodes
   * - /equilibrium/psi
     - :math:`N_Z \times N_R`
     - T m\ :sup:`2`
     - poloidal flux per radian :math:`\psi`
   * - /equilibrium/psib
     - scalar
     - T m\ :sup:`2`
     - flux on the separatrix :math:`\psi_\mathrm{sep}`
   * - /equilibrium/psi_axis
     - scalar
     - T m\ :sup:`2`
     - flux on the magnetic axis :math:`\psi_\mathrm{axis}`; optional,
       the minimum of *psi* is used if it is absent
   * - /equilibrium/btf
     - scalar
     - T
     - toroidal field :math:`B_\mathrm{tf}` at :math:`R_\mathrm{tf}`
   * - /equilibrium/rtf
     - scalar
     - m
     - reference radius :math:`R_\mathrm{tf}`
   * - /equilibrium/br, bt, bz
     - :math:`N_Z \times N_R`
     - T
     - maps of :math:`B_R`, :math:`B_\phi`, :math:`B_Z`; optional,
       but all three must be present together.  If they are present
       they replace Eq. :eq:`howto-bpsi`

Ion species
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 34 12 12 42

   * - Dataset
     - Shape
     - Unit
     - Meaning
   * - /ion_species/names
     - :math:`N_i`
     - string
     - species names with the charge state, for example D+ or O2+
   * - /ion_species/elements
     - :math:`N_i`
     - string
     - element symbols; optional, derived from the names by removing
       the charge suffix if absent
   * - /ion_species/mass_amu
     - :math:`N_i`
     - amu
     - ion masses
   * - /ion_species/charge_state_z
     - :math:`N_i`
     - integer
     - charge states

The element symbols are what
:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`
uses to match the background ions to its projectiles and to the
tables of the :ref:`processes.h5 database <howto_6>`.  The converters
also write the datasets *spec_index* and *main_ion_spec_index*, which
give the index of each species in the source code and identify the
main ion.  No OpenEdge command reads these two datasets.  They are
kept for scripts that edit or inspect the file.

Plasma mesh
^^^^^^^^^^^

The mesh is a triangulation of the poloidal plane.  Several triangles
can belong to one cell, for example the two halves of a quadrilateral
B2 cell.  Field values are stored per cell.

.. list-table::
   :header-rows: 1
   :widths: 34 14 12 40

   * - Dataset
     - Shape
     - Unit
     - Meaning
   * - /mesh/vtx_r, vtx_z
     - :math:`N_v`
     - m
     - vertex coordinates
   * - /mesh/triangles
     - :math:`N_t \times 3`
     - integer
     - vertex indices of each triangle, counted from 0
   * - /mesh/cell_index
     - :math:`N_t`
     - integer
     - index of the cell of each triangle, counted from 0; a
       negative value marks a triangle without plasma data
   * - /mesh/dens_e, dens_i
     - :math:`N_c`
     - m\ :sup:`-3`
     - :math:`n_e` and :math:`n_i`
   * - /mesh/temp_e, temp_i
     - :math:`N_c`
     - eV
     - :math:`T_e` and :math:`T_i`
   * - /mesh/parr_flow
     - :math:`N_c`
     - m/s
     - :math:`u_\parallel`
   * - /mesh/grad_te_r, grad_te_z, grad_ti_r, grad_ti_z
     - :math:`N_c`
     - eV/m
     - components of :math:`\nabla T_e` and :math:`\nabla T_i`;
       optional
   * - /mesh/e_r, e_z, e_t
     - :math:`N_c`
     - V/m
     - :math:`E_R`, :math:`E_Z`, :math:`E_\phi`; optional
   * - /mesh/q_par, q_perp
     - :math:`N_c`
     - W/m\ :sup:`2`
     - parallel and perpendicular plasma heat flux density;
       optional, used for particulate heating
   * - /mesh/dens_n, temp_n
     - :math:`N_c`
     - m\ :sup:`-3`, eV
     - density and temperature of the background neutrals; optional
   * - /mesh/vtx_br, vtx_bz, vtx_bt
     - :math:`N_v`
     - T
     - :math:`B_R`, :math:`B_Z`, :math:`B_\phi` at the vertices;
       optional
   * - /mesh/ions/dens, temp, parr_flow
     - :math:`N_i \times N_c`
     - m\ :sup:`-3`, eV, m/s
     - density, temperature and parallel flow of each ion species
   * - /mesh/wall_face_area
     - :math:`N_c`
     - m\ :sup:`2`
     - area of the wall face of each boundary cell, integrated over
       the toroidal angle; zero for interior cells; optional
   * - /mesh/wall_surf_cell
     - :math:`N_w`
     - integer
     - index of the plasma cell next to each wall element; optional
   * - /mesh/wall_surf_area
     - :math:`N_w`
     - m\ :sup:`2`
     - area of the boundary faces of the plasma mesh assigned to each
       wall element; optional

The two *wall_surf* datasets refer to the elements of the wall
surface file in file order, so they are valid only for the surface
file that the converter wrote together with the plasma file.
:doc:`fix surface/emit/recycle <fix_surface_emit_recycle>` and
:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`
use them to find the plasma cell in front of a wall element.

Wall fluxes
^^^^^^^^^^^

The optional group /wall_flux holds the ion fluxes that the source
code computed at its own boundary faces, as a set of :math:`N_f`
points along the wall.  The fix interpolates them to the wall
elements of the simulation.

.. list-table::
   :header-rows: 1
   :widths: 34 14 14 38

   * - Dataset
     - Shape
     - Unit
     - Meaning
   * - /wall_flux/r, z
     - :math:`N_f`
     - m
     - midpoint of each boundary face
   * - /wall_flux/s
     - :math:`N_f`
     - m
     - distance along the wall
   * - /wall_flux/gamma_i
     - :math:`N_i \times N_f`
     - m\ :sup:`-2` s\ :sup:`-1`
     - flux density :math:`\Gamma` of each ion species onto the wall
   * - /wall_flux/te, ti
     - :math:`N_f`
     - eV
     - :math:`T_e` and :math:`T_i` at the face
   * - /wall_flux/area
     - :math:`N_f`
     - m\ :sup:`2`
     - area of the face
   * - /wall_flux/normal_r, normal_z
     - :math:`N_f`
     - --
     - unit normal of the face; optional
   * - /wall_flux/b_r, b_z, b_t
     - :math:`N_f`
     - T
     - magnetic field at the face; optional

Regular grid
^^^^^^^^^^^^

The converters write the plasma on the mesh only.  The fix also reads
plasma quantities on a regular grid if a file has the top-level
datasets *r* (:math:`N_r` values) and *z* (:math:`N_z` values).  The grid must be
uniform.  The maps are top-level datasets of shape
:math:`N_z \times N_r` with the names and units of the mesh datasets:
*dens_e*, *dens_i*, *temp_e*, *temp_i*, *parr_flow*, *grad_te_r*,
*grad_te_z*, *grad_ti_r*, *grad_ti_z*, *q_par* and *q_perp*.  They
are interpolated bilinearly, and a point outside the grid takes the
value at the nearest point of its edge.  If the file also has a mesh,
the regular grid is used only where the point is in no triangle with
plasma data.  A magnetic field stored on the regular grid is not
read.

zones3d file
^^^^^^^^^^^^

A file for *format zones3d* has a different layout.  Its root has the
string attributes *format* = openedge_background3d, *length_units* =
m and *phi_units* = rad and the integer attribute *version* = 1.  The
optional attribute *ion_mass_amu* is the mass of the main ion, with
default 2.014.  Each zone is a group /zones/zone_<name> with the
integer attributes *zone* (a unique number) and *sector*, and the
string attribute *topology*, one of SOL, PFR and CORE.  A zone is a
structured block of :math:`N_1 \times N_2 \times N_3` vertices in the
radial, poloidal and toroidal directions, with at least two in each,
and of :math:`(N_1-1)(N_2-1)(N_3-1)` cells.  The datasets of a zone
group are:

.. list-table::
   :header-rows: 1
   :widths: 26 26 10 38

   * - Dataset
     - Shape
     - Unit
     - Meaning
   * - vertex_r, vertex_z
     - :math:`N_1 \times N_2 \times N_3`
     - m
     - :math:`R` and :math:`Z` of the vertices
   * - phi
     - :math:`N_3`
     - rad
     - toroidal angle of the vertex planes, strictly increasing
   * - cell_valid
     - :math:`(N_1-1) \times (N_2-1) \times (N_3-1)`
     - integer
     - 0 for a cell without plasma data, non-zero otherwise
   * - cell/te_eV, cell/ti_eV
     - as *cell_valid*
     - eV
     - :math:`T_e` and :math:`T_i`
   * - cell/ne_m3, cell/ni_m3
     - as *cell_valid*
     - m\ :sup:`-3`
     - :math:`n_e` and :math:`n_i`
   * - cell/mach
     - as *cell_valid*
     - --
     - Mach number of the parallel flow
   * - cell/b_mag_T
     - as *cell_valid*
     - T
     - field strength :math:`B`
   * - cell/trace_r, cell/trace_phi, cell/trace_z
     - as *cell_valid*
     - --
     - cylindrical components of the direction of the field line
   * - cell/epar_trace_Vm
     - as *cell_valid*
     - V/m
     - electric field along that direction; optional

All datasets except *epar_trace_Vm* are required, and a dataset with
another shape is an error.  The cell data are held in single
precision.  The zones of one sector must span the same range of
*phi*.  The toroidal angle of a query point is taken in the interval
:math:`[-\pi, \pi)`.  If a point lies in cells of more than one zone,
the SOL zone is used before a PFR zone and a PFR zone before a CORE
zone, and the lower zone number among equals.  The
:doc:`fix background <fix_background>` page describes how the Mach
number and the direction are turned into a flow velocity and a field
vector.

----------

.. _howto_5:

4.5 Wall geometry and surface normals
-------------------------------------

The wall is a surface file read with :doc:`read_surf <read_surf>`.  It
consists of line segments in 2d and of triangles in 3d.  The plasma
converters write the wall contour of the source run to a surface file
together with the plasma file, so that wall elements and plasma cells
correspond.  The points of a 2d surface file are given in box
coordinates: (:math:`R`, :math:`Z`) for a planar run and
(:math:`Z`, :math:`R`) for an axisymmetric run.

The normal of every surface element must point into the volume that
the particles occupy.  All OpenEdge wall models rely on this:
particles are emitted along the normal, and impact angles
:math:`\theta` are measured from it.  The direction of the normal
follows from the order of the points in the file, as described on the
:doc:`read_surf <read_surf>` page, and the *invert* keyword of that
command reverses it.  A closed wall contour needs the normals pointing
inwards.  A surface that encloses a region the particles must not
enter, such as a core boundary drawn along a flux surface, needs them
pointing outwards.  After reading a surface file OpenEdge prints the
flow volume.  Check that it equals the volume of the plasma region
and not the volume of the excluded solid; if it does not, add or
remove *invert*.

Wall models are attached to surface groups.
:doc:`surf_collide <surf_collide>` sets the kinematics of a particle
that hits the wall and :doc:`surf_react <surf_react>` sets the
plasma-wall interaction.  Both are assigned with
:doc:`surf_modify <surf_modify>`.  Wall sources are fixes that emit
from a surface group.

Wall fluxes, erosion rates and the surface composition are resolved
per surface element.  A wall contour taken from a plasma code can
have segments of several centimetres.  The script
tools/refine_wall_2d.py splits them into shorter ones.

----------

.. _howto_6:

4.6 The processes.h5 database
-----------------------------

The atomic rate coefficients and the surface-interaction tables are
stored in one HDF5 file, database/processes.h5, which is part of the
distribution.  The commands look for the file in this order:

#. in the database sub-directory of the path given by the environment
   variable OPENEDGE_ROOT;
#. in the database directory of the source tree the executable was
   built from, whose path is compiled in;
#. in a database sub-directory of the directory where the run was
   started.

A run therefore needs no copy of the file.

The file has two top-level groups, /volume for processes in the
plasma and /surface for processes at the wall.  Element names in the
paths are lower-case chemical symbols, written <el> below.  The
hydrogen isotopes share the rate tables of h, and the reaction
catalog of deuterium is stored under d.  Projectile-target
pairs are written <p>_on_<t>, for example d_on_w.

Volume processes
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 36 16 16 32

   * - Dataset
     - Shape
     - Unit
     - Meaning
   * - /volume/rates/<cls>/<el>/coefficient
     - :math:`N_q \times N_T \times N_n`
     - log\ :sub:`10` of cm\ :sup:`3`/s
     - rate coefficient of class <cls>: scd ionization, acd
       recombination, ccd charge exchange.  The first index is the
       charge state
   * - /volume/radiation/<cls>/<el>/coefficient
     - :math:`N_q \times N_T \times N_n`
     - log\ :sub:`10` of W cm\ :sup:`3`
     - radiated power coefficient: plt line radiation, prb
       recombination and bremsstrahlung
   * - .../temperature
     - :math:`N_T`
     - log\ :sub:`10` of eV
     - :math:`T_e` axis of the table in the same group
   * - .../density
     - :math:`N_n`
     - log\ :sub:`10` of cm\ :sup:`-3`
     - :math:`n_e` axis of the table in the same group
   * - /volume/thresholds/ionization/<el>/energy
     - nuclear charge
     - eV
     - ionization potential of each charge state
   * - /volume/reactions/<el>/catalog
     - scalar
     - string
     - reaction list of the element, in the format of a reactions
       file of :doc:`fix volume/chem/adas <fix_volume_chem_adas>`
   * - /volume/pec/<el>/<id>/<line>/coefficient
     - :math:`N_T \times N_n`
     - log\ :sub:`10` of photons cm\ :sup:`3`/s
     - photon emissivity coefficient of one spectral line, with
       *temperature* (log\ :sub:`10` of eV) and *density*
       (log\ :sub:`10` of m\ :sup:`-3`) axes in the same group

The rate and radiation tables are the ADAS ADF11 data of
`OPEN-ADAS <https://open.adas.ac.uk>`_ and keep its units, and the
photon emissivity coefficients come from ADF15 files.
:doc:`fix volume/chem/adas <fix_volume_chem_adas>` reads the rates,
the ionization potentials and, when it is given an element symbol or
*auto* in place of a reactions file, the catalog.
:doc:`compute volume/emissivity/grid <compute_volume_emissivity_grid>`
reads the photon emissivity coefficients.

Surface processes
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 36 18 14 32

   * - Dataset
     - Shape
     - Unit
     - Meaning
   * - /surface/sputter/<pair>/E
     - :math:`N_E`
     - eV
     - impact energies :math:`E_0`, ascending
   * - /surface/sputter/<pair>/theta
     - :math:`N_\theta`
     - degrees
     - impact angles :math:`\theta` from the surface normal,
       ascending
   * - /surface/sputter/<pair>/Y
     - :math:`N_E \times N_\theta`
     - atoms per ion
     - sputter yield :math:`Y(E_0, \theta)`
   * - /surface/sputter/<pair>/C
     - :math:`N_C`
     - --
     - optional composition axis for a mixed-material target: atomic
       fraction of its first element.  *Y* then has the shape
       :math:`N_C \times N_E \times N_\theta`
   * - /surface/sputter/<pair>/T
     - :math:`N_T`
     - K
     - optional surface-temperature axis, used for liquid lithium.
       *Y* then has the shape
       :math:`N_T \times N_E \times N_\theta`
   * - /surface/reflection/<pair>/E
     - 12
     - eV
     - impact energies
   * - /surface/reflection/<pair>/theta
     - 7
     - degrees
     - impact angles from the surface normal
   * - /surface/reflection/<pair>/raar
     - 5
     - --
     - probability levels of the stored quantiles
   * - /surface/reflection/<pair>/R_N
     - :math:`12 \times 7`
     - --
     - particle reflection coefficient :math:`R_N`
   * - /surface/reflection/<pair>/Eout_min, Eout_max
     - :math:`12 \times 7`
     - eV
     - range of the energy of the reflected particle
   * - /surface/reflection/<pair>/Eout_q
     - :math:`12 \times 7 \times 5`
     - eV
     - quantiles of the reflected energy
   * - /surface/reflection/<pair>/polar_min, polar_max
     - :math:`12 \times 7 \times 5`
     - --
     - range of the polar direction for each energy quantile; stored
       but not used by the sampler
   * - /surface/reflection/<pair>/cos_polar_q
     - :math:`12 \times 7 \times 5 \times 5`
     - --
     - quantiles of the polar direction cosine for each energy
       quantile
   * - /surface/reflection/<pair>/cos_azim_q
     - :math:`12 \times 7 \times 5 \times 5 \times 5`
     - --
     - quantiles of the azimuthal direction cosine for each energy
       and polar quantile

A sputter group can carry the attribute *Es_eV*, the surface binding
energy :math:`E_\mathrm{s}` of the emitted species.  A yield is
interpolated linearly in :math:`\ln E_0` and in :math:`\theta` and is
held constant beyond the ends of both axes.  Each reflection group
has the attributes *Z1*, *M1*, *Z2* and *M2*, the atomic number and
mass of the projectile and of the target.  The dimensions of a
reflection table are fixed; a table of another size is not used.

:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`
reads the sputter tables and uses an analytic yield formula for
pairs without one.
:doc:`surf_react surface/pwi <surf_react_surface_pwi>` reads the
reflection tables and the sputter tables.  The sputter tables are
computed with the RustBCA binary collision code.  The scripts
tools/converters/add_sputter_tables.py and
tools/converters/add_liquid_li_tables.py add new pairs to the file.

----------

.. _howto_7:

4.7 Running multiple simulations from one input script
------------------------------------------------------

This can be done in several ways.  See the documentation for
individual commands for more details on how these examples work.

If "multiple simulations" means continue a previous simulation for
more timesteps, then you simply use the :doc:`run <run>` command
multiple times.  For example, this script

.. parsed-literal::

   read_grid data.grid
   create_particles 1000000
   run 10000
   run 10000
   run 10000
   run 10000
   run 10000

would run 5 successive simulations of the same system for a total of
50,000 timesteps.

If you wish to run totally different simulations, one after the other,
the :doc:`clear <clear>` command can be used in between them to
re-initialize OpenEdge.  For example, this script

.. parsed-literal::

   read_grid data.grid
   create_particles 1000000
   run 10000
   clear
   read_grid data.grid2
   create_particles 500000
   run 10000

would run 2 independent simulations, one after the other.

For large numbers of independent simulations, you can use
:doc:`variables <variable>` and the :doc:`next <next>` and
:doc:`jump <jump>` commands to loop over the same input script multiple
times with different settings.  For example, this script, named
in.flow

.. parsed-literal::

   variable d index run1 run2 run3 run4 run5 run6 run7 run8
   shell cd $d
   read_grid data.grid
   create_particles 1000000
   run 10000
   shell cd ..
   clear
   next d
   jump in.flow

would run 8 simulations in different directories, using a data.grid
file in each directory.  The same concept could be used to run the
same system at 8 different gas densities, using a density variable and
storing the output in different log and dump files, for example

.. parsed-literal::

   variable a loop 8
   variable rho index 1.0e18 4.0e18 1.0e19 4.0e19 1.0e20 4.0e20 1.0e21 4.0e21
   log log.$a
   read data.grid
   global nrho ${rho}
   ...
   compute myGrid grid all all n temp
   dump 1 grid all 1000 dump.$a id c_myGrid
   run 100000
   clear
   next rho
   next a
   jump in.flow

All of the above examples work whether you are running on 1 or
multiple processors, but assumed you are running OpenEdge on a single
partition of processors.  OpenEdge can be run on multiple partitions via
the "-partition" command-line switch as described in
:ref:`Section 2.5 <start_5>` of the manual.

In the last 2 examples, if OpenEdge were run on 3 partitions, the same
scripts could be used if the "index" and "loop" variables were
replaced with *universe*-style variables, as described in the
:doc:`variable <variable>` command.  Also, the "next rho" and "next a"
commands would need to be replaced with a single "next a rho" command.
With these modifications, the 8 simulations of each script would run
on the 3 partitions one after the other until all were finished.
Initially, 3 simulations would be started simultaneously, one on each
partition.  When one finished, that partition would then start the 4th
simulation, and so forth, until all 8 were completed.

----------

.. _howto_8:

4.8 Output from OpenEdge (stats, dumps, computes, fixes, variables)
-------------------------------------------------------------------

There are four basic kinds of OpenEdge output:

* :doc:`Statistical output <stats_style>`, which is a list of quantities
  printed every few timesteps to the screen and logfile.

* :doc:`Dump files <dump>`, which contain snapshots of particle, grid
  cell, or surface element quantities and are written at a specified
  frequency.

* Certain fixes can output user-specified quantities directly to files:
  :doc:`fix ave/time <fix_ave_time>` for time averaging, and
  :doc:`fix print <fix_print>` for single-line output of
  :doc:`variables <variable>`.  Fix print can also output to the
  screen.

* :doc:`Restart files <restart>`.

A simulation prints one set of statistical output and (optionally)
restart files.  It can generate any number of dump files and fix
output files, depending on what :doc:`dump <dump>` and :doc:`fix <fix>`
commands you specify.

As discussed below, OpenEdge gives you a variety of ways to determine
what quantities are computed and printed when the statistics, dump, or
fix commands listed above perform output.  Throughout this discussion,
note that users can also add their own computes and fixes to OpenEdge
which can generate values that
can then be output with these commands.

The following sub-sections discuss different OpenEdge commands related
to output and the kind of data they operate on and produce:

* :ref:`Global/per-particle/per-grid/per-surf data <global>`
* :ref:`Scalar/vector/array data <scalar>`
* :ref:`Statistical output <stats>`
* :ref:`Dump file output <dump>`
* :ref:`Fixes that write output files <fixoutput>`
* :ref:`Computes that process output quantities <computeoutput>`
* :ref:`Computes that generate values to output <compute>`
* :ref:`Fixes that generate values to output <fix>`
* :ref:`Variables that generate values to output <variable>`
* :ref:`Summary table of output options and data flow between commands <table>`

.. _global:

Global/per-particle/per-grid/per-surf data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Various output-related commands work with four different styles of
data: global, per particle, per grid, or per surf.  A global datum is
one or more system-wide values, e.g. the temperature of the system.  A
per particle datum is one or more values per partice, e.g. the kinetic
energy of each particle.  A per grid datum is one or more values per
grid cell, e.g. the temperature of the particles in the grid cell.  A
per surf datum is one or more values per surface element, e.g. the
count of particles that collided with the surface element.

.. _scalar:

Scalar/vector/array data
^^^^^^^^^^^^^^^^^^^^^^^^

Global, per particle, per grid, and per surf datums can each come in
three kinds: a single scalar value, a vector of values, or a 2d array
of values.  The doc page for a "compute" or "fix" or "variable" that
generates data will specify both the style and kind of data it
produces, e.g. a per grid vector.

When a quantity is accessed, as in many of the output commands
discussed below, it can be referenced via the following bracket
notation, where ID in this case is the ID of a compute.  The leading
"c\_" would be replaced by "f\_" for a fix, or "v\_" for a variable:

.. list-table::

   * - c_ID
     - entire scalar, vector, or array
   * - c_ID[I]
     - one element of vector, one column of array
   * - c_ID[I][J]
     - one element of array

In other words, using one bracket reduces the dimension of the data
once (vector -> scalar, array -> vector).  Using two brackets reduces
the dimension twice (array -> scalar).  Thus a command that uses
scalar values as input can typically also process elements of a vector
or array.

.. _stats:

Statistical output
^^^^^^^^^^^^^^^^^^

The frequency and format of statistical output is set by the
:doc:`stats <stats>`, :doc:`stats_style <stats_style>`, and
:doc:`stats_modify <stats_modify>` commands.  The
:doc:`stats_style <stats_style>` command also specifies what values are
calculated and written out.  Pre-defined keywords can be specified
(e.g. np, ncoll, etc).  Three additional kinds of keywords can also be
specified (c_ID, f_ID, v_name), where a :doc:`compute <compute>` or
:doc:`fix <fix>` or :doc:`variable <variable>` provides the value to be
output.  In each case, the compute, fix, or variable must generate
global values to be used as an argument of the
:doc:`stats_style <stats_style>` command.

.. _dump:

Dump file output
^^^^^^^^^^^^^^^^

Dump file output is specified by the :doc:`dump <dump>` and
:doc:`dump_modify <dump_modify>` commands.  There are several
pre-defined formats: dump particle, dump grid, dump surf, etc.

Each of these allows specification of what values are output with each
particle, grid cell, or surface element.  Pre-defined attributes can
be specified (e.g. id, x, y, z for particles or id, vol for grid
cells, etc).  Three additional kinds of keywords can also be specified
(c_ID, f_ID, v_name), where a :doc:`compute <compute>` or :doc:`fix <fix>`
or :doc:`variable <variable>` provides the values to be output.  In each
case, the compute, fix, or variable must generate per particle, per
grid, or per surf values for input to the corresponding
:doc:`dump <dump>` command.

.. _fixoutput:

Fixes that write output files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two fixes take various quantities as input and can write output files:
:doc:`fix ave/time <fix_ave_time>` and :doc:`fix print <fix_print>`.

The :doc:`fix ave/time <fix_ave_time>` command enables direct output to
a file and/or time-averaging of global scalars or vectors.  The user
specifies one or more quantities as input.  These can be global
:doc:`compute <compute>` values, global :doc:`fix <fix>` values, or
:doc:`variables <variable>` of any style except the particle style which
does not produce single values.  Since a variable can refer to
keywords used by the :doc:`stats_style <stats_style>` command (like
particle count), a wide variety of quantities can be time averaged
and/or output in this way.  If the inputs are one or more scalar
values, then the fix generates a global scalar or vector of output.
If the inputs are one or more vector values, then the fix generates a
global vector or array of output.  The time-averaged output of this
fix can also be used as input to other output commands.

The :doc:`fix print <fix_print>` command can generate a line of output
written to the screen and log file or to a separate file, periodically
during a running simulation.  The line can contain one or more
:doc:`variable <variable>` values for any style variable except the
particle style.  As explained above, variables themselves can contain
references to global values generated by
:doc:`stats keywords <stats_style>`, :doc:`computes <compute>`,
:doc:`fixes <fix>`, or other :doc:`variables <variable>`.  Thus the
:doc:`fix print <fix_print>` command is a means to output a wide
variety of quantities separate from normal statistical or dump file
output.

.. _computeoutput:

Computes that process output quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :doc:`compute reduce <compute_reduce>` command takes one or more per
particle or per grid or per surf vector quantities as inputs and
"reduces" them (sum, min, max, ave) to scalar quantities.  These are
produced as output values which can be used as input to other output
commands.

.. _compute:

Computes that generate values to output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every :doc:`compute <compute>` in OpenEdge produces either global or per
particle or per grid or per surf values.  The values can be scalars or
vectors or arrays of data.  These values can be output using the other
commands described in this section.  The doc page for each compute
command describes what it produces.  Computes that produce per
particle or per grid or per surf values have the word "particle" or
"grid" or "surf" in their style name.  Computes without those words
produce global values.

.. _fix:

Fixes that generate values to output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some :doc:`fixes <fix>` in OpenEdge produces either global or per particle
or per grid or per surf values which can be accessed by other
commands.  The values can be scalars or vectors or arrays of data.
These values can be output using the other commands described in this
section.  The doc page for each fix command tells whether it produces
any output quantities and describes them.

Two fixes of particular interest for output are the
:doc:`fix ave/grid <fix_ave_grid>` and
:doc:`fix ave/surf <fix_ave_surf>` commands.

The :doc:`fix ave/grid <fix_ave_grid>` command enables time-averaging of
per grid vectors.  The user specifies one or more quantities as input.
These can be per grid vectors or ararys from :doc:`compute <compute>` or
:doc:`fix <fix>` commands.  If the input is a single vector, then the
fix generates a per grid vector.  If the input is multiple vectors or
array, the fix generates a per grid array.  The time-averaged output
of this fix can also be used as input to other output commands.

The :doc:`fix ave/surf <fix_ave_surf>` command enables time-averaging of
per surf vectors.  The user specifies one or more quantities as input.
These can be per surf vectors or ararys from :doc:`compute <compute>` or
:doc:`fix <fix>` commands.  If the input is a single vector, then the
fix generates a per surf vector.  If the input is multiple vectors or
array, the fix generates a per surf array.  The time-averaged output
of this fix can also be used as input to other output commands.

.. _variable:

Variables that generate values to output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:doc:`Variables <variable>` defined in an input script generate either a
global scalar value or a per particle vector (only particle-style
variables) when it is accessed.  The formulas used to define equal-
and particle-style variables can contain references to the
:doc:`stats_style <stats_style>` keywords and to global and per particle
data generated by computes, fixes, and other variables.  The values
generated by variables can be output using the other commands
described in this section.

.. _table:

Summary table of output options and data flow between commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This table summarizes the various commands that can be used for
generating output from OpenEdge.  Each command produces output data of
some kind and/or writes data to a file.  Most of the commands can take
data from other commands as input.  Thus you can link many of these
commands together in pipeline form, where data produced by one command
is used as input to another command and eventually written to the
screen or to a file.  Note that to hook two commands together the
output and input data types must match, e.g. global/per atom/local
data and scalar/vector/array data.

Also note that, as described above, when a command takes a scalar as
input, that could be an element of a vector or array.  Likewise a
vector input could be a column of an array.

.. list-table::
   :header-rows: 1

   * - Command
     - Input
     - Output
   * - :doc:`stats_style <stats_style>`
     - global scalars
     - screen, log file
   * - :doc:`dump particle <dump>`
     - per particle vectors
     - dump file
   * - :doc:`dump grid <dump>`
     - per grid vectors
     - dump file
   * - :doc:`dump surf <dump>`
     - per surf vectors
     - dump file
   * - :doc:`fix print <fix_print>`
     - global scalar from variable
     - screen, file
   * - :doc:`print <print>`
     - global scalar from variable
     - screen
   * - :doc:`computes <compute>`
     - N/A
     - global or per particle/grid/surf scalar/vector/array
   * - :doc:`fixes <fix>`
     - N/A
     - global or per particle/grid/surf scalar/vector/array
   * - :doc:`variables <variable>`
     - global scalars, per particle vectors
     - global scalar, per particle vector
   * - :doc:`compute reduce <compute_reduce>`
     - per particle/grid/surf vectors
     - global scalar/vector
   * - :doc:`fix ave/time <fix_ave_time>`
     - global scalars/vectors
     - global scalar/vector/array, file
   * - :doc:`fix ave/grid <fix_ave_grid>`
     - per grid vectors/arrays
     - per grid vector/array
   * - :doc:`fix ave/surf <fix_ave_surf>`
     - per surf vectors/arrays
     - per surf vector/array

----------

.. _howto_9:

4.9 Visualizing snapshots
-------------------------

The :doc:`dump image <dump_image>` command can be used to do on-the-fly
visualization as a simulation proceeds.  It works by creating a series
of JPG or PNG or PPM files on specified timesteps, as well as movies.
The images can include particles, grid cell quantities, and/or surface
element quantities.  This is not a substitute for using an interactive
visualization package in post-processing mode, but on-the-fly
visualization can be useful for debugging or making a high-quality
image of a particular snapshot of the simulation.

The :doc:`dump <dump>` command can be used to create snapshots of
particle, grid cell, or surface element data as a simulation runs.
These can be post-processed and read in to other visualization
packages.

A Python-based toolkit distributed by our group can read OpenEdge
particle dump files with columns of user-specified particle
information, and convert them to various formats or pipe them into
visualization software directly.  See the
`Pizza.py WWW site <http://lammps.github.io/pizza>`_ for details.
Specifically, Pizza.py can convert OpenEdge particle dump files into
PDB, XYZ, `Ensight <http://www.ensight.com>`_, and VTK formats.
Pizza.py can
pipe OpenEdge dump files directly into the Raster3d and RasMol
visualization programs.  Pizza.py has tools that do interactive 3d
OpenGL visualization and one that creates SVG images of dump file
snapshots.

Additional Pizza.py tools may be added that allow visualization of
surface and grid cell information as output by OpenEdge.

----------

.. _howto_10:

4.10 Details of grid geometry
-----------------------------

OpenEdge overlays a grid over the simulation domain which is used to
track particles, to look up the background plasma along their paths
and to tally volume processes such as ionization and radiation, and
DSMC collisions when they are enabled.  Surface elements are
also assigned to grid cells they intersect with, so that
particle/surface collisions can be efficiently computed.

OpenEdge uses a Cartesian hierarchical grid.  Cartesian means that the
faces of a grid cell, at any level of the hierarchy, are aligned with
the Cartesian xyz axes.  I.e. each grid cell is an axis-aligned
parallelepiped or rectangular box.

The hierarchy of grid cells is defined for N levels, from 1 to N.  The
entire simulation box is a single parent grid cell, conceptually at
level 0.  It is subdivided into a regular grid of Nx by Ny by Nz cells
at level 1.  "Regular" means all the Nx\*Ny\*Nz sub-divided cells within
any parent cell are the same size.  Each of those cells can be a child
cell (no further sub-division) or it can be a parent cell which is
further subdivided into Nx by Ny by Nz cells at level 2.  This can
recurse to as many levels as desired.  Different cells can stop
recursing at different levels.  The Nx,Ny,Nz values for each level of
the grid can be different, but they are the same for every grid cell
at the same level.  The per-level Nx,Ny,Nz values are defined by the
:doc:`create_grid <create_grid>`, :doc:`read_grid <read_grid>`,
:doc:`adapt_grid <adapt_grid>`, or :doc:`fix_adapt <fix_adapt>` commands.

As described below, each child cell is assigned an ID which encodes
the cell's logical position within in the hierarchical grid, as a
32-bit or 64-bit unsigned integer ID.  The precision is set by the
-DSPARTA_BIG or -DSPARTA_SMALL or -DSPARTA_BIGBIG compiler switch; see
the presets in :ref:`Section 2.2 <start_2>`.  The number of
grid levels that can be used depends on this precision and the
resolution of the grid at each level.  For example, in a 3d
simulation, a level that is refined with a 2x2x2 sub-grid requires 4
bits of the ID.  Thus a maximum of 8 levels can be used for 32-bit IDs
and 16 levels for 64-bit IDs.

This manner of defining a hierarchical grid allows for flexible grid
cell refinement in any region of the simulation domain, e.g. near
plasma-facing surfaces such as the divertor targets.  Also note that
a 3d oct-tree (quad-tree in 2d) is a special case of the OpenEdge
hierarchical grid, where Nx = Ny = Nz = 2 is used at every level.

The diagram shows a 2d axisymmetric grid for the WEST example in
examples/workflows/impurity_transport/west_tungsten_transport, with
the wall and the core boundary in red and the separatrix in orange.
The level 1 grid is a uniform 50x88 grid of 3.7 cm cells (left).
:doc:`adapt_grid <adapt_grid>` then splits a cell 2x2 wherever the
ion density changes by more than 50% across it, using the *ni_jump*
value of :doc:`compute plasma/fields <compute_plasma_fields>`:

.. code-block:: text

   fix         pd background file input/plasma.h5 static yes
   compute     nij plasma/fields all background pd ni_jump
   adapt_grid  all refine value c_nij 0.5 0 maxlevel 4 cells 2 2 1 iterate 3

The fine cells follow the steep density region along the separatrix
and into the divertor, down to 4.6 mm at level 4, while the core and
the far scrape-off layer stay coarse.  The grid has 21794 child
cells.  The input script and plotting script for this figure are in
doc/utils/west_grid_figure.

.. image:: JPG/west_refine_grid.png
   :align: center

In the rest of the OpenEdge manual, the following terminology is used to
refer to the cells of the hierarchical grid.  The flow region is the
portion of the simulation domain that is "outside" any surface objects
and is typically filled with particles.

* root cell = the overall simulation box
* parent cell = a grid cell that is sub-divided (the root cell is a parent cell)
* child cell = a grid cell that is not sub-divided further
* unsplit cell = a child cell not intersected by any surface elements
* cut cell = a child cell intersected by one or more surface elements, resulting in a single flow region
* split cell = a child cell intersected by two or more surface elements, resulting in two or more disjoint flow regions
* sub cell = one disjoint flow region portion of a split cell

Note that in OpenEdge, parent cells are only conceptual.  They do not
exist as individual entities or require memory.  Child cells store
various attributes and are distributed across processors, so that each
child cell is owned by exactly one processor, as discussed below.

When surface objects are defined via the :doc:`read_surf <read_surf>`
command, they intersect child cells.  In this context "intersection" by
a surface element means a geometric overlap between the area of the
surface element and the volume of the grid cell (or length of element
and area of grid cell in 2d).  Thus an intersection includes a surface
triangle that only touches a grid cell on its face, edge, or at its
corner point.  When intersected by one or more surface elements, a
child cell becomes one of 3 flavors: unsplit, cut, or split.  A child
cell not intersected by any surface elements is an unsplit cell.  It
can be entirely in the flow region or entirely inside a surface
object.  If a child cell is intersected so that it is partitioned into
two contiguous volumes, one in the flow region, the other inside a
surface object, then it is a cut cell.  This is the usual case.  Note
that either the flow volume or inside volume can be of size zero, if
the surface only "touches" the grid cell, i.e. the intersection is
only on a face, edge, or corner point of the grid cell.  The left side
of the diagram below is an example, where red represents the flow
region.  Sometimes a child cell can be partitioned by surface elements
so that more than one contiguous flow region is created.  Then it is a
split cell.  Additionally, each of the two or more contiguous flow
regions is a sub cell of the split cell.  The right side of the
diagram shows a split cell with 3 sub cells.

.. image:: JPG/split.jpg
   :align: center

The union of (1) unsplit cells that are in the flow region (not
entirely interior to a surface object) and (2) flow region portions of
cut cells and (3) sub cells is the entire flow region of the
simulation domain.  These are the only kinds of child cells that store
particles.  Split cells and unsplit cells interior to surface objects
have no particles.

Child cell IDs can be output in integer or string form by the
:doc:`dump grid <dump>` command, using its *id* and *idstr*
attributes.  The integer form can also be output by the
:doc:`compute property/grid <compute_property_grid>`.

Here is how a grid cell ID is computed by OpenEdge, either for parent or
child cells.  Say the level 1 grid is a 10x10x20 sub-division (2000
cells) of the root cell (simulation box).  The level 1 cells are
numbered from 1 to 2000 with the x-dimension varying fastest, then y,
and finally the z-dimension slowest.  Consider the 376th level 1 cell.
It would be the 6th cell in the x direction of the grid, 8th cell in
y, and 4th cell in z.  I.e. 376 = (z-1)\*100 + (y-1)\*10 + (x-1) + 1.
Now consider the case where level 2 cells use a 2x2x2 sub-division (8
cells) of level 1 cells and consider the 4th level 2 cell within the
376th level 1 cell.  This would be the 2nd cell in x, 2nd cell in y,
and 1st cell in z.  I.e. 4 = (z-1)\*4 + (y-1)\*2 + (x-1) + 1.

This level 2 cell could itself be a parent cell if it were further
sub-divided, or a child cell if not.  In either case its ID is the
same and is calculated as follows.  The rightmost 11 bits of the
integer ID are encoded with 376.  This is because it requires 11 bits
to represent 2000 cells (1 to 2000) at level 1.  The next 4 bits are
encoded with 4, because it requires 4 bits to represent 8 cells (1 to
8) at level 2.  Thus the level 2 cell ID in integer format is 4\*2048 +
376 = 8568.  In string format it would be 376-4, with dashes
separating each of the levels.  Either of these formats (integer or
string) can be specified as id or idstr for output of grid cell info
with the :doc:`dump grid <dump>` command; see its doc page for more
details.

Note that a child cell has the same ID whether it is unsplit, cut, or
split.  Currently, sub cells of a split cell also have the same ID,
though that may change in the future.

The :doc:`create_grid <create_grid>` and
:doc:`balance_grid <balance_grid>` and
:doc:`fix balance <fix_balance>` commands determine the assignment of child
cells to processors.  If a child cell is assigned to a processor, that
processor owns the cell whether it is an unsplit, cut, or split cell.
It also owns any sub cells that are part of a split cell.

Depending on which assignment options in these commands are used, the
child cells assigned to each processor will either be "clumped" or
"dispersed".

Clumped means each processor's cells will be geometrically compact.
Dispersed means the processor's cells will be geometrically dispersed
across the simulation domain and so they cannot be enclosed in a small
bounding box.

The diagram shows a clumped assignment: the WEST grid above after
``balance_grid rcb cell`` on 8 processors, cropped to the vessel.
Each processor owns one compact block of cells, and the cell edges
show the refined regions inside each block.  By contrast a dispersed
assignment could scatter a processor's cells throughout the entire
simulation domain.  The *cell* weighting gives each processor the same
number of cells, about 2725.  In an axisymmetric run the box starts
at the axis, R = 0, so processor 0 owns only empty space inboard of
the wall, and processor 1 owns little more than the thin inner
scrape-off layer.  Weighting by particle count with ``balance_grid rcb part`` or
:doc:`fix balance <fix_balance>` during the run avoids this.

.. image:: JPG/west_partition.png
   :align: center

It is important to understand the difference between the two kinds of
assignments and the effects they can have on performance of a
simulation.  For example the create_grid and read_grid commands may
produce dispersed assignments, depending on the options used, which
can be converted to a clumped assignment by the balance_grid command.

Simulations typically run faster with clumped grid cell assignments.
This is because the cost of communicating particles is reduced if
particles that move to a neighboring grid cell often stay
on-processor.  Similarly, some stages of simulation setup may run
faster with a clumped assignment.  Examples are the finding of nearby
ghost grid cells and the computation of surface element intersections
with grid cells.  The latter operation is invoked when the
:doc:`read_surf <read_surf>` command is used.

If the spatial distribution of particles is highly irregular and/or
dynamically changing, or if the computational work per grid cell is
otherwise highly imbalanced, a clumped assignment of grid cells to
processors may not lead to optimal balancing.  In these scenarios a
dispersed assignment of grid cells to processors may run faster even
with the overhead of increased particle communication.  This is
because randomly assigning grid cells to processors can balance the
computational load in a statistical sense.

----------

.. _howto_11:

4.11 Details of surfaces
------------------------

A OpenEdge simulation can define one or more surface objects, each of
which are read in via the :doc:`read_surf <read_surf>`.  For 2d
simulations a surface object is a collection of connected line
segments.  For 3d simulations it is a collection of connected
triangles.  The outward normal of lines or triangles, as defined in
the surface file, points into the flow region of the simulation box
which is typically filled with particles.  Depending on the
orientation, surface objects can thus be obstacles that particles flow
around, or they can represent the outer boundary of an irregular
shaped region which particles are inside of.

See the :doc:`read_surf <read_surf>` doc page for a discussion of these
topics:

* Requirement that a surface object be "watertight", so that particles
  do not enter inside the surface or escape it if used as an outer
  boundary.

* Surface objects (one per file) that contain more than one physical
  object, e.g. two or more spheres in a single file.

* Use of geometric transformations (translation, rotation, scaling,
  inversion) to convert the surface object in a file into different
  forms for use in different simulations.

* Clipping a surface object to the simulation box to effectively use a
  portion of the object in a simulation, e.g. a half sphere instead of a
  full sphere.

* The kinds of surface objects that are illegal, including infinitely
  thin objects, ones with duplicate points, or multiple surface or
  physical objects that touch or overlap.

The :doc:`read_surf <read_surf>` command assigns an ID to the surface
object in a file.  This can be used to reference the surface elements
in the object in other commands.  For example, every surface object
must have a collision model assigned to it so that particle bounces
off the surface can be computed.  This is done via the
:doc:`surf_modify <surf_modify>` and :doc:`surf_collide <surf_collide>`
commands.

As described in the previous :ref:`Section 4.10 <howto_10>`, OpenEdge
overlays a grid over the simulation domain to track particles.  Surface elements
are also assigned to grid cells they intersect with, so that
particle/surface collisions can be efficiently computed.  Typically a
grid cell size larger than the surface elements that intersect it may
not desirable since it means flow around the surface object will not
be well resolved.  The size of the smallest surface element in the
system is printed when the surface file is read.  Note that if the
surface object is clipped to the simulation box, small lines or
triangles can result near the box boundary due to the clipping
operation.

The maximum number of surface elements that can intersect a single
child grid cell is set by the :doc:`global surfmax <global>` command.
The default limit is 100.  The actual maximum number in any grid cell
is also printed when the surface file is read.  Values this large or
larger may cause particle moves to become expensive, since each time a
particle moves within that grid cell, possible collisions with all its
overlapping surface elements must be computed.

----------

.. _howto_12:

4.12 Restarting a simulation
----------------------------

There are two ways to continue a long OpenEdge simulation.  Multiple
:doc:`run <run>` commands can be used in the same input script.  Each
run will continue from where the previous run left off.  Or binary
restart files can be saved to disk using the :doc:`restart <restart>`
command.  At a later time, these binary files can be read via a
:doc:`read_restart <read_restart>` command in a new script.

Here is an example of a script that reads a binary restart file and
then issues a new run command to continue where the previous run left
off.  It illustrates what settings must be made in the new script.
Details are discussed in the documentation for the
:doc:`read_restart <read_restart>` and
:doc:`write_restart <write_restart>` commands.

Look at the *in.collide* input script provided in the *bench*
directory of the OpenEdge distribution to see the original script that
this script is based on.  If that script had the line

.. parsed-literal::

   restart             50 tmp.restart

added to it, it would produce 2 binary restart files (tmp.restart.50
and tmp.restart.100) as it ran for 130 steps, one at step 50, and one
at step 100.

This script could be used to read the first restart file and re-run
the last 80 timesteps:

.. parsed-literal::

   read_restart        tmp.restart.50

   seed                12345
   collide             vss air ar.vss

   stats               10
   compute             temp temp
   stats_style         step cpu np nattempt ncoll c_temp

   timestep            7.00E-9
   run                 80

Note that the following commands do not need to be repeated because
their settings are included in the restart file: *dimension, global,
boundary, create_box, create_grid, species, mixture*.  However these
commands do need to be used, since their settings are not in the
restart file: *seed, collide, compute, fix, stats_style, timestep*.
The :doc:`read_restart <read_restart>` doc page gives details.

If you actually use this script to perform a restarted run, you will
notice that the statistics output does not match exactly.  On step 50,
the collision counts are 0 in the restarted run, because the line is
printed before the restarted simulation begins.  The collision counts
in subsequent steps are similar but not identical.  This is because
new random numbers are used for collisions in the restarted run.  This
affects all the randomized operations in a simulation, so in general
you should only expect a restarted run to be statistically similar to
the original run.

----------

.. _howto_13:

4.13 Visualizing output with ParaView
-------------------------------------

OpenEdge can write VTK XML data sets directly for visualization in
ParaView.  No Python conversion step and no ParaView Python interpreter
are required.  Build OpenEdge with *-DPKG_VTK=ON* to enable these dump
styles.

The native VTK styles correspond to the three kinds of simulation data:

.. parsed-literal::

   dump dg grid/vtk all 100 output/grid\_\*.vtu id c_fields[\*]
   dump ds surf/vtk wall 100 output/wall\_\*.vtu id f_tally[\*]
   dump dp particle/vtk all 100 output/particles\_\*.vtu id type x y z vx vy vz

The *grid/vtk* style writes grid cells as VTK UnstructuredGrid data and
requires a *.vtu* filename.  The *surf/vtk* and *particle/vtk* styles
accept *.vtu* for UnstructuredGrid data or *.vtp* for PolyData.  Their
field arguments use the same syntax as the corresponding *grid*, *surf*,
and *particle* dump styles.

Every native VTK filename must contain *\** because one data set is
written per output timestep.  The asterisk is replaced by the timestep.
For example, *output/grid\_\*.vtu* at timestep 500 becomes
*output/grid_500.vtu*.  The *pad* option of the
:doc:`dump_modify <dump_modify>` command can add leading zeroes.

Without a *%* character in the filename, MPI ranks gather their data to
one writer and produce one file per timestep.  Add *%* when each rank or
processor cluster should write a piece.  OpenEdge then also writes the
parallel *.pvtu* or *.pvtp* index file that should be opened in ParaView.

To export the initial grid or surface without advancing the simulation,
request the first dump and run zero steps:

.. parsed-literal::

   dump dgrid grid/vtk all 1 output/grid\_\*.vtu id
   dump_modify dgrid first yes
   dump dsurf surf/vtk all 1 output/surface\_\*.vtu id
   dump_modify dsurf first yes
   run 0

The runnable *examples/visualization/paraview* case demonstrates
time-dependent native output for the grid, an embedded surface, and
particles injected with *fix emit/face*.  Open its generated *.vtu* file
series directly in ParaView.

----------

.. _howto_14:

4.14 Custom per-particle, per-grid, per-surf attributes
-------------------------------------------------------

Particles, grid cells, and surface elements can have custom attributes
which store either single or multiple values per particle, per grid
cell, or per surface element.  If a single value is stored, the
attribute is referred to as a custom per-particle, per-grid, or
per-surf vector.  If multiple values are stored, the attribute is
referred to as a custom per-particle, per-grid, or per-surf array (an
array can have a single column and thus a single value per entity).
Each custom attribute has a name, which allows them to be specified in
input scripts as arguments to various commands.  The values each
attricute stores can be either integer or floating point numbers.

Here are lists of current commands (as of August 2023) which use custom
attributes in various ways:

Per-particle custom attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`compute reduce <compute_reduce>` - reduce a per-particle attribute to a scalar value
* :doc:`custom <custom>` - set the values of a per-particle attribute and optionally create it
* :doc:`dump particle <dump>` - output per-particle attributes to a dump file
* :doc:`fix ambipolar <fix_ambipolar>` - use a per-particle vector and array for ambipolar quantities
* :doc:`variable <variable>` - use a per-particle attribute in a particle-style variable formula

Per-grid custom attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`compute reduce <compute_reduce>` - reduce a per-grid attribute to a scalar value
* :doc:`custom <custom>` - set the values of a per-grid attribute and optionally create it
* :doc:`dump grid <dump>` - output per-grid attributes to a dump file
* :doc:`fix ave/grid <fix_ave_grid>` - time-average a per-grid attribute
* :doc:`read_grid <read_grid>` - define and initialize per-grid attributes
* surf_react implicit - use per-grid vectors and an array to store chemical state (not yet released in public OpenEdge)
* :doc:`variable <variable>` - use a per-grid attribute in a grid-style variable formula
* :doc:`write_grid <write_grid>` - write per-grid attributes to a grid data file

Per-surf custom attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`compute reduce <compute_reduce>` - reduce a per-surf attribute to a scalar value
* :doc:`custom <custom>` - set the values of a per-surf attribute and optionally create it
* :doc:`dump surf <dump>` - output per-surf attributes to a dump file
* :doc:`fix ave/surf <fix_ave_surf>` - time-average a per-surf attribute
* :doc:`fix surf/temp <fix_surf_temp>` - use a per-surf vector for temperature
* :doc:`read_surf <read_surf>` - define and initialize per-surf attributes
* :doc:`surf_collide <surf_collide>` - use a per-surf attribute as temperature for particle/surf collisions
* :doc:`surf_react adsorb <surf_react_adsorb>` - use per-surf vectors and an array to store chemical state
* :doc:`variable <variable>` - use a per-surf attribute in a surf-style variable formula
* :doc:`write_surf <write_surf>` - write per-surf attributes to a surf data file

Per-surf custom attributes can be defined for explicit or
explicit/distributed surface elements, as set by the
:doc:`global surfs <global>` command.  But they cannot be used for implicit
surface elements.  Conceptually, implicit surfaces are defined on a
per-grid cell basis, so per-grid custom attributes can be used instead
for attributes of those implicit surfaces.

Note that in some cases the name for a custom attribute is specified
by the user, e.g. for the :doc:`read_grid <read_grid>` or
:doc:`read_surf <read_surf>` commands.  In other cases, a command
defines the name for the attributes and documents the name(s) it uses,
e.g. for the :doc:`fix ambipolar <fix_ambipolar>` or
:doc:`surf_react adsorb <surf_react_adsorb>` commands.

Also note that custom attributes can be static or dynamic quantities.
For example, the :doc:`read_surf <read_surf>` command can be used to
define a *static* temperature for each surface element it reads,
stored as a custom per-surf vector.  By contrast, the
:doc:`fix surf/temp <fix_surf_temp>` command can be used to define a
*dynamic* temperature for each surface element which is calculated
once every N steps from the energy flux which colliding particles
impart to each surface element, also stored in a custom per-surf
vector.

In both cases, the custom per-surf vector can be passed to the
:doc:`surf_collide diffuse <surf_collide>` command to each timestep when
particle/surface element collisions take place.

Another use of dynamic custom attributes is by the
:doc:`fix ambipolar <fix_ambipolar>` and
:doc:`surf_react adsorb <surf_react_adsorb>` commands.  The former stores the ambipolar state of each particle in
per-particle attributes.  The latter stores the chemical state of each
surface element in per-surf attributes.  These will vary over the
course of a simulation, and their status can be monitored with the
various output commands listed above.

----------

.. _howto_15:

4.15 Variable timestep simulations
----------------------------------

As an alternative to utilization of a user-provided constant timestep,
the variable timestep option enables OpenEdge to compute global
timesteps based on the current state of the physical processes being
modeled. The timestep is global in the sense that all cells advance
their particles in time using the same timestep value.  The timestep
is adaptive in the sense that the global timestep can be recalculated
periodically throughout the simulation to account for flow state
changes.  Examples of situations where a variable timestep would be
desired are problems with highly varying density or velocity
throughout the domain and transient problems where the optimal
timestep changes throughout the simulation.

The global, variable timestep is computed at a user-specified
frequency using cell-based timesteps that are calculated using cell
mean collision and particle transit times.  These cell-based timesteps
are only used to compute the global timestep and are not used to
advance the solution locally. The benefit of the global timestep
calculation is that it will automatically reduce the timestep if the
intial value is too large, leading to higher accuracy, and it will
automatically increase the timestep if the initial value is too small,
speeding up the simulation. The overhead of using the variable timestep
option is the computational time involved in computing the cell-based
time quantities and performing parallel reductions over the grid to
construct the global minimum and average cell timesteps needed for the
global timestep calculation. For scenarios where ensembles of similar
problems are being run, one strategy to mitigate this cost is to
determine an optimal timestep using the variable timestep option for
the first run and then to utilize this timestep as a user-specified
value for the subsequent runs.

The :doc:`compute dt/grid <compute_dt_grid>` command is used to
calculate the cell-based timesteps, and the
:doc:`fix dt/reset <fix_dt_reset>` command uses this data to calculate
the global timestep.  An internal time variable has been added to OpenEdge
to track elapsed simulation time, and this time variable as well as
the current timestep can be output using the *time* and *dt* keywords
in the :doc:`stats_style <stats_style>` command. These *time* and *dt*
values are also included in the :doc:`read_restart <read_restart>` and
:doc:`write restart <write_restart>` commands.
