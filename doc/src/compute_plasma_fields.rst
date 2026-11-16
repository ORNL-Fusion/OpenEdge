.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: compute plasma/fields

compute plasma/fields command
=============================

Syntax
------

.. parsed-literal::

   compute ID plasma/fields group-ID source sargs ... value1 value2 ...

* ID is documented in the :doc:`compute <compute>` command
* plasma/fields = style name of this compute command
* group-ID = ID of the grid group whose cells are evaluated
* source = *background* or *constant* or *analytic*

  * *background* fix-ID = ID of a :doc:`fix background <fix_background>`
  * *constant* = zero or more constant keyword/value pairs, optionally
    ended by the word *values*
  * *analytic* = zero or more constant or analytic keyword/value pairs,
    optionally ended by the word *values*

* constant keywords = *magnetic_field* or *electric_field* or *temp_e*
  or *temp_i* or *dens_e* or *dens_i* or *parrflow*

  * *magnetic_field* values = Br Bt Bz = uniform
    :math:`(B_R, B_\phi, B_Z)` (T)
  * *electric_field* values = Er Et Ez = uniform
    :math:`(E_R, E_\phi, E_Z)` (V/m)
  * *temp_e* value = electron temperature (eV)
  * *temp_i* value = ion temperature (eV)
  * *dens_e* value = electron density (m\ :sup:`-3`)
  * *dens_i* value = ion density (m\ :sup:`-3`)
  * *parrflow* value = parallel flow speed (m/s)

* analytic keywords = *r0* or *ne0* or *te0* or *te1* or *ti0* or
  *eps* or *x0* or *y0*

  * *r0* value = radius of the column (m)
  * *ne0* value = peak electron density (m\ :sup:`-3`)
  * *te0* value = electron temperature outside the column (eV)
  * *te1* value = peak electron temperature minus *te0* (eV)
  * *ti0* value = uniform ion temperature (eV)
  * *eps* value = softening added to the squared radius (m\ :sup:`2`)
  * *x0*, *y0* value = position of the column axis (m)

* one or more values must be appended
* value = *br* or *bt* or *bz* or *bx* or *by* or *er* or *et* or *ez*
  or *ex* or *ey* or *vr* or *vt* or *vz* or *vx* or *vy* or *dens_e*
  or *temp_e* or *dens_i* or *temp_i* or *parrflow* or *epar* or
  *q_par* or *q_perp* or *q_mag* or *grad_te_r* or *grad_te_t* or
  *grad_te_z* or *grad_ti_r* or *grad_ti_t* or *grad_ti_z* or
  *grad_ne_r* or *grad_ne_z* or *grad_te_mag* or *sepdist* or
  *ni_jump*

  * *br*, *bt*, *bz* = :math:`B_R`, :math:`B_\phi`, :math:`B_Z` (T)
  * *bx*, *by* = magnetic field along the simulation x and y axes (T)
  * *er*, *et*, *ez* = :math:`E_R`, :math:`E_\phi`, :math:`E_Z` (V/m)
  * *ex*, *ey* = electric field along the simulation x and y axes (V/m)
  * *vr*, *vt*, *vz* = :math:`R`, :math:`\phi`, :math:`Z` components of
    the parallel flow velocity (m/s)
  * *vx*, *vy* = flow velocity along the simulation x and y axes (m/s)
  * *dens_e*, *dens_i* = :math:`n_e`, :math:`n_i` (m\ :sup:`-3`)
  * *temp_e*, *temp_i* = :math:`T_e`, :math:`T_i` (eV)
  * *parrflow* = :math:`u_\parallel` (m/s)
  * *epar* = :math:`\mathbf{E}\cdot\hat{\mathbf{b}}` (V/m)
  * *q_par*, *q_perp* = parallel and perpendicular heat flux
    (W/m\ :sup:`2`)
  * *q_mag* = :math:`(q_\parallel^2 + q_\perp^2)^{1/2}`
    (W/m\ :sup:`2`)
  * *grad_te_r*, *grad_te_t*, *grad_te_z* = :math:`R`, :math:`\phi`,
    :math:`Z` components of :math:`\nabla T_e` (eV/m)
  * *grad_ti_r*, *grad_ti_t*, *grad_ti_z* = :math:`R`, :math:`\phi`,
    :math:`Z` components of :math:`\nabla T_i` (eV/m)
  * *grad_ne_r*, *grad_ne_z* = :math:`R` and :math:`Z` components of
    :math:`\nabla n_e` (m\ :sup:`-4`)
  * *grad_te_mag* = magnitude of :math:`\nabla T_e` in the poloidal
    plane (eV/m)
  * *sepdist* = :math:`|\psi_N - 1|`
  * *ni_jump* = relative change of :math:`n_i` across the cell
    (dimensionless)

Examples
--------

.. code-block:: text

   fix pd background file plasma.h5
   compute cplasma plasma/fields all background pd dens_e temp_e temp_i &
       er ez et br bz bt
   compute csep plasma/fields all background pd sepdist
   compute nij plasma/fields all background pd ni_jump
   adapt_grid all refine value c_nij 0.5 0 maxlevel 4 cells 2 2 1 iterate 3
   compute cconst plasma/fields all constant temp_e 20 dens_e 1e19 &
       magnetic_field 0.0 2.0 0.0 values br bt bz temp_e
   compute cana plasma/fields all analytic r0 0.05 ne0 5e18 te0 2 te1 8 &
       magnetic_field 0 0 1 values dens_e temp_e grad_te_r

Description
-----------

Evaluate a plasma background at the centre of each grid cell in the
group and store the requested quantities as per-grid values.  The
output can be dumped or averaged next to particle data, and it can
serve as the refinement criterion of :doc:`adapt_grid <adapt_grid>`,
from plasma gradients or from the distance to the separatrix.  The
compute can also be named as the plasma provider of the *pusher*
keyword of the :doc:`global <global>` command in place of a
:doc:`fix background <fix_background>`.

Symbols are defined on the :ref:`notation <notation>` page.  The
relation between :math:`(R, \phi, Z)` and the simulation box is given
on the :ref:`fix background <background_coords>` page.

The cell values are sampled when a run starts and again whenever the
grid changes through adaptation or load balancing.

Background source
^^^^^^^^^^^^^^^^^

With *background* all data come from the named
:doc:`fix background <fix_background>`.

For an axisymmetric (*format legacy*) background:

* :math:`n_e`, :math:`T_e`, :math:`n_i` and :math:`T_i` are the values
  the fix returns at the cell centre.
* *parrflow* is the value of the mesh cell that contains the cell
  centre.  If the centre is outside the mesh, the nearest mesh
  triangle within 0.05 m is used.
* If the fix has an equilibrium, the magnetic field is the
  per-triangle mesh field where the file has one and the cell centre
  lies in the mesh.  Elsewhere it is taken from the equilibrium of the
  fix, Eq. :eq:`background-bpsi`, and is zero outside the rectangle
  covered by the equilibrium grid.
* If the fix has no equilibrium, the magnetic field is the one the
  fix returns at the cell centre: the per-triangle mesh field, or
  the uniform field set by the *br*, *bz*, *bt* or *bcart* keywords
  of the fix.  The order of these sources is given on the
  :doc:`fix background <fix_background>` page.  A cell for which no
  source applies stores zero.
* The electric field is the per-cell field stored on the mesh, taken
  from the mesh triangle whose centroid is nearest to the cell centre
  and within 0.1 m.  It is zero if the file has no electric field.
* *q_par* and *q_perp* are the heat fluxes the fix returns at the cell
  centre.
* The flow components *vr*, *vt*, *vz* and all gradient values are
  available only from a background on a regular :math:`(R,Z)` grid,
  which includes the *constant* source of the fix.  They are
  interpolated bilinearly, and :math:`\nabla n_e` is the gradient of
  the bilinear interpolant of :math:`n_e`.  For a background stored on
  a mesh these values are zero.

For a *format zones3d* background each cell centre is sampled through
the three-dimensional lookup of the fix, which supplies the densities,
temperatures, flow, :math:`\mathbf{B}` and the parallel temperature
gradients.  A cell centre outside the zones stores zeros.  The
*sepdist* and heat-flux values are not available, and the *br*, *bt*,
*bz*, *bx* and *by* values require a nonzero *b_sign* on the fix.

Constant source
^^^^^^^^^^^^^^^

With *constant* every cell receives the same plasma, defined by the
constant keywords.  *magnetic_field* and *electric_field* take
cylindrical components in the order :math:`(R, \phi, Z)`.  A keyword
that is not given is zero.  The flow components and all gradients are
zero.

Analytic source
^^^^^^^^^^^^^^^

With *analytic* the plasma is a column with a flat top and a steep
edge,

.. math::

   n_e = n_i = n_{e0}\, f(r) , \qquad
   T_e = T_{e0} + T_{e1}\, f(r) , \qquad
   T_i = T_{i0} , \qquad
   f(r) = \exp\!\left[-\left(\frac{r}{r_0}\right)^{12}\right] ,

where

.. math::

   r = \sqrt{(x-x_0)^2 + (y-y_0)^2 + \epsilon}

in 3d and :math:`r = \sqrt{(x-x_0)^2 + \epsilon}` in 2d.  The symbols
:math:`n_{e0}`, :math:`T_{e0}`, :math:`T_{e1}`, :math:`T_{i0}`,
:math:`r_0` and :math:`\epsilon` are the values of *ne0*, *te0*,
*te1*, *ti0*, *r0* and *eps*.  If *x0* or *y0* is omitted, the centre
of the simulation box is used.  The *grad_ne_r* and *grad_te_r* values
return the analytic derivatives
:math:`n_{e0}\, \mathrm{d}f/\mathrm{d}r` and
:math:`T_{e1}\, \mathrm{d}f/\mathrm{d}r`.  The other gradients are
zero.  The fields and the parallel flow come from the constant
keywords *magnetic_field*, *electric_field* and *parrflow*.

For *constant* and *analytic* the keyword list ends at the word
*values* or at the first word that is not one of these keywords.  Use
*values* whenever the first output value has the same name as a
keyword, as *temp_e* and *dens_e* do.

Values
^^^^^^

The values *br*, *bt*, *bz*, *er*, *et*, *ez*, *vr*, *vt* and *vz* are
cylindrical components.  The values *bx*, *by*, *ex*, *ey*, *vx* and
*vy* are the same vectors along the simulation x and y axes.  In a
planar 2d simulation these are the :math:`(R, Z)` components and in an
axisymmetric simulation the :math:`(Z, R)` components.  In 3d the
:math:`R` and :math:`\phi` components are rotated by the toroidal
angle of the cell centre about the column axis of the background,
:math:`\phi = \operatorname{atan2}(y - y_0,\, x - x_0)`, where
:math:`(x_0, y_0)` is the *column_axis* of the
:doc:`fix background <fix_background>`.  For the *constant* and
*analytic* sources the axis passes through the origin.

.. note::

   In 2d the *bz*, *ez* and *vz* values return the component in the
   simulation z slot, which is the toroidal component.  For the
   cylindrical :math:`Z` component in 2d use *by*, *ey*, *vy* (planar)
   or *bx*, *ex*, *vx* (axisymmetric).

The *epar* value is the projection of the electric field on
:math:`\hat{\mathbf{b}}`, and is zero where :math:`B = 0`.

The *q_par*, *q_perp* and *q_mag* values are the heat fluxes stored in
the background.  They require a *background* source whose plasma file
has heat-flux data.

The *grad_te_mag* value is
:math:`[(\partial_R T_e)^2 + (\partial_Z T_e)^2]^{1/2}`.  The toroidal
component is excluded.  It is meant as a refinement criterion for
:doc:`adapt_grid <adapt_grid>`.

The *sepdist* value is :math:`|\psi_N - 1|` at the cell centre, with
:math:`\psi_N` from Eq. :eq:`background-psin`.  Refining with the
*thresh less* option of :doc:`adapt_grid <adapt_grid>` on this value
gives a band of fine cells that follows the separatrix.

The *ni_jump* value is

.. math::
   :label: plasma-fields-nijump

   \frac{\max n_i - \min n_i}{\langle n_i \rangle} ,

taken over a 3x3 stencil of points spanning the cell (its centre,
corners and edge midpoints; 3x3x3 in 3d), with :math:`n_i` sampled
from the background at each point.  Points outside the plasma are
skipped, and a cell with fewer than two points inside the plasma
gets 0.  It is meant as a refinement criterion for
:doc:`adapt_grid <adapt_grid>`: a threshold of 0.5 refines every cell
across which the ion density changes by more than half its mean.  An
example is the WEST grid in :ref:`Section 4.10 <howto_10>`.

Output info
-----------

This compute calculates a per-grid vector if one value is given, or a
per-grid array with one column per value, in the order the values are
listed.  The output can be written by :doc:`dump grid <dump>`,
time-averaged by :doc:`fix ave/grid <fix_ave_grid>`, or used as the
refinement criterion of :doc:`adapt_grid <adapt_grid>` and
:doc:`fix adapt <fix_adapt>`.

Cells that are not in the group and split cells store zeros.  The
units of each value are given in the syntax list above.

Restrictions
------------

This command is part of the OPENEDGE package.  It is only enabled if
OpenEdge was built with that package.  See
:ref:`Section 2.3 <start_3>` for more info.

At least one value must be given.  The *background* source requires a
:doc:`fix background <fix_background>` with the given ID to exist when
a run starts.

The *sepdist* value requires a *background* source whose fix has an
equilibrium flux map.  The *ni_jump* value requires a *background*
source.  The *grad_ne_r* and *grad_ne_z* values are not available
from a background with a plasma mesh or a native 3d plasma, which
carry no density gradient; use *ni_jump* to refine on density.  The heat-flux values require a *background*
source with heat-flux data.  The compute stops with an error
otherwise.

Related commands
----------------

:doc:`fix background <fix_background>`,
:doc:`compute grid <compute_grid>`,
:doc:`adapt_grid <adapt_grid>`,
:doc:`global pusher <global>`

Default
-------

The constant keywords default to zero.  The analytic keywords default
to r0 = 0.02, ne0 = 1.0e19, te0 = 1.0, te1 = 4.0, ti0 = 1.0, eps =
1.0e-10, and x0, y0 = centre of the simulation box.
