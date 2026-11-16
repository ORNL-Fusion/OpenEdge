.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix background

fix background command
======================

Syntax
------

.. parsed-literal::

   fix ID background source keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* background = style name of this fix command
* source = *file* or *constant*

  * *file* path = HDF5 plasma file written by an OpenEdge converter
  * *constant* = spatially uniform plasma defined by the keywords below

* zero or more keyword/value pairs may be appended
* keywords for either source = *equilibrium* or *static* or *rz_axes*
  or *column_axis* or *mesh_triangle_cache* or
  *mesh_lookup_diagnostics* or *wall_flux_rescale*

  * *equilibrium* value = path of a separate equilibrium file with the
    poloidal-flux map
  * *static* value = *yes* or *no*

    * *yes* = read the plasma data at the first run and keep it
    * *no* = read the plasma data again at the start of every run

  * *rz_axes* value = *axi* or *planar*

    * *axi* = take simulation x as :math:`Z` and y as :math:`R`
    * *planar* = take simulation x as :math:`R` and y as :math:`Z`

  * *column_axis* values = x0 y0 = position of the symmetry axis in the
    x-y plane of a 3d simulation (m)
  * *mesh_triangle_cache* value = *yes* or *no* = store with each
    particle the mesh triangle in which it was last found
  * *mesh_lookup_diagnostics* value = *yes* or *no* = print lookup
    counters at the end of each run
  * *wall_flux_rescale* value = *auto* or *yes* or *no* = rescale the
    interpolated wall fluxes to the totals of the wall-flux samples

* keywords for *file* = *format* or *flow_sign* or *b_sign* or
  *cs_model* or *outside* or *precision*

  * *format* value = *legacy* or *zones3d*

    * *legacy* = axisymmetric plasma on an :math:`(R,Z)` mesh
    * *zones3d* = three-dimensional zoned background

  * *flow_sign* value = +1 or -1 = sign of the parallel flow along the
    stored field-line direction (*zones3d* only, required)
  * *b_sign* value = -1 or 0 or +1 = sign of :math:`\mathbf{B}` along
    the stored field-line direction, 0 if unknown (*zones3d* only)
  * *cs_model* value = *te_plus_ti* = sound speed that converts the
    stored Mach number to a flow speed (*zones3d* only, required)
  * *outside* value = *fail* or *vacuum* (*zones3d* only)

    * *fail* = a query outside the zones is an error
    * *vacuum* = a query outside the zones returns zero plasma

  * *precision* value = *single* = storage precision of the *zones3d*
    file

* keywords for *constant* = *r_bounds* or *z_bounds* or *dens_e* or
  *temp_e* or *dens_i* or *temp_i* or *ions* or *dens_n* or *temp_n*
  or *upar* or *upar_r* or *upar_t* or *upar_z* or *epar* or
  *grad_te_r* or *grad_te_t* or *grad_te_z* or *grad_ti_r* or
  *grad_ti_t* or *grad_ti_z*

  * *r_bounds* values = Rlo Rhi = radial extent of the constant
    plasma (m)
  * *z_bounds* values = Zlo Zhi = vertical extent of the constant
    plasma (m)
  * *dens_e* (or *ne*) value = electron density (m\ :sup:`-3`)
  * *temp_e* (or *te*) value = electron temperature (eV)
  * *dens_i* (or *ni*) value = ion density (m\ :sup:`-3`)
  * *temp_i* (or *ti*) value = ion temperature (eV)
  * *ions* values = N name1 mass1 frac1 ... nameN massN fracN

    * N = number of background ion species
    * name = species name with its charge, such as D+ or O3+
    * mass = species mass (amu)
    * frac = species density as a fraction of the ion density

  * *dens_n* (or *nn*) value = neutral density (m\ :sup:`-3`)
  * *temp_n* (or *tn*) value = neutral temperature (eV)
  * *upar* (or *parr_flow*) value = parallel flow speed (m/s)
  * *upar_r*, *upar_t*, *upar_z* (or *parr_flow_r*, *parr_flow_t*,
    *parr_flow_z*) value = :math:`R`, :math:`\phi` or :math:`Z`
    component of the parallel flow velocity (m/s)
  * *epar* value = parallel electric field (V/m)
  * *grad_te_r*, *grad_te_t*, *grad_te_z* value = :math:`R`,
    :math:`\phi` or :math:`Z` component of :math:`\nabla T_e` (eV/m)
  * *grad_ti_r*, *grad_ti_t*, *grad_ti_z* value = :math:`R`,
    :math:`\phi` or :math:`Z` component of :math:`\nabla T_i` (eV/m)

* field keywords, accepted with either source = *br* or *bz* or *bt*
  or *bcart* or *er* or *et* or *ez*

  * *br*, *bz*, *bt* value = :math:`B_R`, :math:`B_Z` or
    :math:`B_\phi` of a uniform cylindrical magnetic field (T)
  * *bcart* values = Bx By Bz = uniform Cartesian magnetic field (T)
  * *er*, *ez*, *et* value = :math:`E_R`, :math:`E_Z` or
    :math:`E_\phi` of a uniform cylindrical electric field (V/m)

Examples
--------

.. code-block:: text

   fix pd background file plasma.h5 static yes
   fix pd background file plasma.h5 equilibrium shot.equ
   fix pd background file w7x_background3d.h5 format zones3d &
       flow_sign +1 cs_model te_plus_ti b_sign -1 outside vacuum
   fix pd background constant temp_e 20 dens_e 1e19 temp_i 20 &
       dens_i 1e19 br 0.0 bz 5.0
   fix pd background constant te 10 ne 1e18 ti 10 ni 1e18 &
       bcart 0.0 0.0 2.0 upar 1.0e4
   fix pd background constant te 25 ne 2e19 bt 2.0 &
       ions 2 D+ 2.014 0.97 O3+ 15.999 0.01

Description
-----------

Define the plasma background for the OpenEdge fixes and computes that
need one.  The fix holds the electron and ion density and temperature,
the parallel flow, the electric and magnetic fields and, when the
source provides them, the temperature gradients, heat fluxes, neutral
density and temperature, the poloidal-flux map and the list of
background ion species.  The fix does not act on particles.  Other
commands name it by its ID and query it at particle positions or at
grid-cell centres: the *pusher* keyword of the :doc:`global <global>`
command, :doc:`compute plasma/fields <compute_plasma_fields>`,
:doc:`fix reflect/psi <fix_reflect_psi>`, and the collision,
sputtering, sheath and particulate models.  An input script normally
defines one fix background and points every consumer at it.

Exactly one of *file* or *constant* must be given.  Symbols are
defined on the :ref:`notation <notation>` page.

.. _background_coords:

Coordinates
^^^^^^^^^^^

The plasma is stored in cylindrical coordinates :math:`(R, \phi, Z)`.
The mapping to the simulation box depends on the
:doc:`dimension <dimension>` of the run and on whether the box is
axisymmetric.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Simulation
     - Position
     - Vector components :math:`(V_x, V_y, V_z)`
   * - 2d planar
     - :math:`R = x,\; Z = y`
     - :math:`(V_R,\; V_Z,\; V_\phi)`
   * - 2d axisymmetric
     - :math:`Z = x,\; R = y`
     - :math:`(V_Z,\; V_R,\; V_\phi)`
   * - 3d
     - :math:`R = \sqrt{(x-x_0)^2 + (y-y_0)^2},\;
       \phi = \operatorname{atan2}(y-y_0,\, x-x_0),\; Z = z`
     - :math:`(V_R\cos\phi - V_\phi\sin\phi,\;
       V_R\sin\phi + V_\phi\cos\phi,\; V_Z)`

In both 2d layouts the toroidal component of a vector is carried in
the z slot of the particle velocity and of the fields, and
:math:`\phi = 0`.  With these assignments :math:`(R, \phi, Z)` is a
right-handed triad in the axisymmetric and 3d layouts.  In the planar
layout :math:`\hat{\mathbf{x}}\times\hat{\mathbf{y}}` points along
:math:`-\hat{\boldsymbol{\phi}}` of a right-handed
:math:`(R, \phi, Z)`.

The *rz_axes* keyword changes the 2d mapping.  *axi* selects the
axisymmetric assignment (x = :math:`Z`, y = :math:`R`) in a planar
2d run, so that a surface file and plasma file prepared for an
axisymmetric run can be reused.  *planar* removes that override.  An
axisymmetric box always uses the axisymmetric assignment.  The setting
is global: it applies to every command that converts between the box
and :math:`(R, Z)`.  *rz_axes axi* in a 3d box is an error.

The *column_axis* keyword sets :math:`(x_0, y_0)`, the position of the
symmetry axis in a 3d box.  It is meant for linear devices whose box
is not centred on the plasma column.  It is ignored in 2d.

File source
^^^^^^^^^^^

The *file* source reads an HDF5 plasma file produced by one of the
OpenEdge converters (SOLPS, SOLEDGE3X, OEDGE, EMC3).  With
*format legacy*, the default, the file describes an axisymmetric
plasma on a triangulated :math:`(R,Z)` mesh.  Each triangle belongs to
a mesh cell that carries one value of :math:`n_e`, :math:`T_e`,
:math:`n_i`, :math:`T_i`, :math:`u_\parallel` and, if present in the
file, the neutral density and temperature, the
:math:`R` and :math:`Z` components of :math:`\nabla T_e` and
:math:`\nabla T_i`, the heat fluxes :math:`q_\parallel` and
:math:`q_\perp`, and :math:`\mathbf{E}`.  A query returns the value of
the cell whose triangle contains the point, without interpolation.
The file may also hold an equilibrium group with the poloidal-flux
map, a table of background ion species with per-species density,
temperature and parallel flow, and wall-flux samples.

A file that carries the plasma on a regular :math:`(R,Z)` grid is also
accepted.  Its fields are interpolated bilinearly, and a point outside
the grid takes the value at the nearest grid edge.  If a file has both
a mesh and a regular grid, the regular grid is used only for points
outside the mesh.  A point outside a mesh-only file returns zero
density and temperature.  If the file has no ion density, :math:`n_i`
is set equal to :math:`n_e`.

The flow velocity returned to consumers is
:math:`u_\parallel\hat{\mathbf{b}}`.

The *static* keyword controls re-reading.  With *static no* the source
is read again at the start of every :doc:`run <run>` command, as
needed by an outer loop that rewrites the plasma file between runs.
With *static yes* it is read once.

Magnetic field
^^^^^^^^^^^^^^

For a *legacy* file or a *constant* source the magnetic field at a
point is taken from the first of these that is available:

1. the uniform Cartesian field of *bcart*;
2. the field stored on the mesh vertices of the plasma file.  The
   three vertex values of a triangle are averaged, so the field is
   constant within a triangle;
3. :math:`B_R`, :math:`B_\phi` and :math:`B_Z` maps stored with the
   equilibrium on its :math:`(R,Z)` grid, interpolated bilinearly;
4. the poloidal-flux map :math:`\psi(R,Z)` of the equilibrium, Eq.
   :eq:`background-bpsi`;
5. the uniform cylindrical field of *br*, *bz* and *bt*.

A point outside the mesh skips source 2.  If none applies the field
is zero.  The field from the flux map is

.. math::
   :label: background-bpsi

   B_R = -\frac{1}{R}\frac{\partial\psi}{\partial Z} ,
   \qquad
   B_Z = \frac{1}{R}\frac{\partial\psi}{\partial R} ,
   \qquad
   B_\phi = \frac{B_\mathrm{tf}\,R_\mathrm{tf}}{R} ,

where :math:`\psi` is the poloidal flux per radian (Wb/rad) and
:math:`B_\mathrm{tf}` is the toroidal field at the reference radius
:math:`R_\mathrm{tf}`, both read from the equilibrium.  The
derivatives are central differences at the node of the equilibrium
grid nearest to the point, so this field is piecewise constant on the
scale of that grid.

The guiding-centre pusher needs the spatial derivatives of
:math:`\mathbf{B}`, which the mesh field does not have.  It therefore
asks for sources 3 and 4 before source 2, and falls back to the order
above if there is no equilibrium.

The normalized flux used by :doc:`fix reflect/psi <fix_reflect_psi>`
and by the *sepdist* value of
:doc:`compute plasma/fields <compute_plasma_fields>` is

.. math::
   :label: background-psin

   \psi_N = \frac{\psi - \psi_\mathrm{axis}}
                 {\psi_\mathrm{sep} - \psi_\mathrm{axis}} ,

with :math:`\psi` interpolated bilinearly on the equilibrium grid and
clamped to the edge of that grid outside it.
:math:`\psi_\mathrm{sep}` is the boundary flux stored with the
equilibrium.  :math:`\psi_\mathrm{axis}` is the stored axis value if
the file has one, and otherwise the minimum of the map.

The *equilibrium* keyword names a separate text file in the layout of
the SOLPS equilibrium files: header entries *jm*, *km*, *psib*, *btf*
and *rtf*, followed by the arrays *r(1:jm)*, *z(1:km)* and
*psi(j,k)*.  Its flux map replaces the equilibrium group of the plasma
file.  The keyword may be combined with either source.

*bcart* sets the same Cartesian vector everywhere in a 3d box.  The
cylindrical keywords cannot express this, because :math:`B_R` and
:math:`B_\phi` rotate with :math:`\phi`.  In 2d the three *bcart*
values are taken as :math:`(B_R, B_Z, B_\phi)`.

Electric field
^^^^^^^^^^^^^^

The electric field is the per-cell :math:`(E_R, E_Z, E_\phi)` stored
on the mesh of a *legacy* file.  A point outside the mesh, or a file
without these datasets, has no electric field.  If any of *er*, *ez*
or *et* is given, the uniform field they define is returned instead,
also with a *file* source.

zones3d format
^^^^^^^^^^^^^^

*format zones3d* selects the three-dimensional background written by
the EMC3 converter.  The plasma is stored on hexahedral cells in one
or more toroidal zones, each labelled as scrape-off layer, private
flux region or core.  A query locates the zone and cell that contain
the point :math:`(R, \phi, Z)` and returns the cell values of
:math:`T_e`, :math:`T_i`, :math:`n_e`, :math:`n_i`, :math:`B`, the
Mach number :math:`M` and the unit vector
:math:`\hat{\mathbf{t}}` along the stored field-line direction.
Nothing is projected onto an :math:`(R,Z)` plane.  The format requires
a 3d simulation and the *file* source.

The file stores a Mach number and a field-line direction without the
sense of :math:`\mathbf{B}`, so two keywords are mandatory.
*flow_sign* gives the sign :math:`s_u = \pm 1` of the flow relative to
:math:`\hat{\mathbf{t}}`, and *cs_model te_plus_ti* declares the sound
speed that converts :math:`M` to a velocity:

.. math::

   \mathbf{u} = s_u\, M\, c_s\, \hat{\mathbf{t}} ,
   \qquad
   c_s = \sqrt{\frac{T_e + T_i}{m_i}} ,

with :math:`m_i` the ion mass recorded in the file (2.014 amu if the
file does not record one).

*b_sign* gives the sign :math:`s_B` of :math:`\mathbf{B}` along
:math:`\hat{\mathbf{t}}`.  With :math:`s_B = \pm 1` the fix returns
:math:`\mathbf{B} = s_B B\,\hat{\mathbf{t}}` and
:math:`u_\parallel = s_B s_u M c_s`.  The pusher and the *br*, *bt*
and *bz* values of
:doc:`compute plasma/fields <compute_plasma_fields>` require this.
With the default 0 only :math:`B` and :math:`\hat{\mathbf{t}}` are
available, and a consumer that asks for the vector :math:`\mathbf{B}`
stops with an error.

The *outside* keyword decides what happens when a point lies outside
every zone or in a cell marked invalid.  With *fail* the run stops
with an error that reports the point.  With *vacuum* the query returns
zero density, temperature and field, which is appropriate when the box
extends beyond the plasma grid.

*precision single* is the only accepted value.  It records the storage
precision of the file and does not change how the file is read.

The temperature gradients returned for this format are parallel
gradients, :math:`\hat{\mathbf{t}}\,\partial T/\partial s`, computed
by differencing neighbouring cells along the field line.  If the file
has a parallel electric field, :math:`\mathbf{E} = E_\parallel
\hat{\mathbf{t}}` is returned.  The format has no heat-flux data, so
consumers that need :math:`q_\parallel` or :math:`q_\perp` cannot be
used with it.

Constant source
^^^^^^^^^^^^^^^

The *constant* source defines a spatially uniform plasma for tests and
benchmarks.  A scalar keyword that is not given is zero, except that
:math:`n_i` defaults to :math:`n_e` and :math:`T_i` to :math:`T_e`.
*dens_n* and *temp_n* add a uniform neutral background.

*r_bounds* and *z_bounds* set the :math:`(R,Z)` rectangle on which the
constant plasma is tabulated.  The upper bound must exceed the lower
bound.  A bound that is not given is taken from the simulation box: in
2d the box limits along the :math:`R` and :math:`Z` axes, in 3d
:math:`R` from 0 to the largest distance of a box corner from the
origin and :math:`Z` between the z limits of the box.

*upar* sets :math:`u_\parallel`.  *upar_r*, *upar_t* and *upar_z* set
the cylindrical components of the flow velocity, *epar* a parallel
electric field, and the gradient keywords the components of
:math:`\nabla T_e` and :math:`\nabla T_i` used by
:doc:`fix force/thermal <fix_force_thermal>`.  The fix stores these
values.  Which of them a consumer reads is described on the page of
that consumer.

The *ions* keyword defines a table of background ion species, as a
*legacy* file can.  The element and the charge state are taken from
the name: O3+ is oxygen with charge state 3, and a name without a
number has charge state 1.  Each species has density *frac* times
:math:`n_i`, temperature :math:`T_i` and parallel flow
:math:`u_\parallel`.

Lookup options
^^^^^^^^^^^^^^

With *mesh_triangle_cache yes* each particle remembers the mesh
triangle in which it was last found, and the next query for that
particle starts its search there and walks to neighbouring triangles.
The hint is a per-particle custom integer named *oe_mesh_tri_ID*, with
ID the ID of this fix.  The keyword applies to *format legacy* only
and does not change results.

With *mesh_lookup_diagnostics yes* the fix prints counters to the
screen and log file after each run.  For a *legacy* source these are
the number of triangle searches resolved by the particle hint, by the
grid-cell hint and by the full search, the number that needed a walk
to a neighbour, the number of points outside the mesh, the number of
magnetic-field queries answered by each source, the particle count per
processor, and the fastest particle.  For *zones3d* they are the
numbers of queries that were answered, outside, in an invalid cell,
and claimed by more than one cell.

A *legacy* file may contain wall-flux samples, ion fluxes at points
along the wall, which :doc:`fix surface/emit/recycle
<fix_surface_emit_recycle>` and
:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`
interpolate onto the surface elements by inverse-distance weighting.  With *wall_flux_rescale yes* the
interpolated flux is rescaled so that, for the elements nearest to
each sample, flux times element area sums to the flux times area of
that sample.  *auto* does this only in a 2d axisymmetric run and *no*
never.  The rescaling needs sample areas in the file.  It is not
applied in a planar 2d run or with distributed surfaces.

Restrictions
------------

This command is part of the OPENEDGE package.  It is only enabled if
OpenEdge was built with that package.  See
:ref:`Section 2.3 <start_3>` for more info.

Reading a plasma file requires a build with HDF5.

*format zones3d* requires a 3d simulation, the *file* source, and
explicit *flow_sign* and *cs_model te_plus_ti* settings.  It cannot be
combined with a nonzero *column_axis*.

In a 3d simulation a *bcart* field is available only to consumers
that query the fix with the full particle position.

Related commands
----------------

:doc:`compute plasma/fields <compute_plasma_fields>`,
:doc:`global pusher <global>`,
:doc:`fix reflect/psi <fix_reflect_psi>`,
:doc:`compute nearest_surf/grid <compute_nearest_surf_grid>`

Default
-------

The keyword defaults are static = no, format = legacy, b_sign = 0,
outside = fail, mesh_triangle_cache = yes, mesh_lookup_diagnostics =
no, wall_flux_rescale = auto and column_axis = 0 0.  The 2d axis
assignment follows the box, axisymmetric or planar.  *r_bounds* and
*z_bounds* default to the extent of the simulation box.  *flow_sign*
and *cs_model* have no default.  The constant-plasma values default to
0, except *dens_i* and *temp_i*, which default to *dens_e* and
*temp_e*.  No ion species table is defined unless *ions* is given.
