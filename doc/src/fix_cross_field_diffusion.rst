.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix cross_field_diffusion

fix cross_field_diffusion command
=================================

Syntax
------

.. parsed-literal::

   fix ID cross_field_diffusion Nevery bsource keyword values ...

* ID is documented in the :doc:`fix <fix>` command
* cross_field_diffusion = style name of this fix command
* Nevery = apply the displacement every this many timesteps
* bsource = *background* or *bfield_const* or *bfield*

  * *background* fix-ID = ID of a :doc:`fix background <fix_background>`
    that supplies :math:`\mathbf{B}` and, where needed, :math:`T_e`,
    :math:`n_e` and :math:`\psi_N` at the particle position
  * *bfield_const* Bx By Bz = uniform magnetic field (T) in the
    coordinate order of the simulation box
  * *bfield* BxSRC BySRC BzSRC = sources for the magnetic field
    components (T) in the coordinate order of the simulation box.  Each
    source is *c_ID[col]*, column col of a per-grid compute, or
    *p_name*, a per-particle custom vector.

* one or more keyword/value pairs must be appended
* keyword = *D_perp* or *bohm* or *binormal* or *pinch* or *pinch_psi*
  or *gradient_pinch*

  * *D_perp* value = constant diffusivity :math:`D_\perp`
    (m\ :sup:`2`/s)
  * *bohm* values = [TeSRC] [*scale* S]

    * TeSRC = source for the electron temperature (eV).  Required with
      *bfield_const* and *bfield*, not allowed with *background*.
    * *scale* S = multiplier on the Bohm diffusivity

  * *binormal* value = *yes* or *no* = in 2d, take a second random
    step along the binormal direction or not
  * *pinch* values = Vr Vz = :math:`R` and :math:`Z` components of a
    constant pinch velocity (m/s)
  * *pinch_psi* value = V = pinch speed along :math:`\nabla\psi_N`
    (m/s).  A negative value points towards the magnetic axis.
  * *gradient_pinch* values = Cp [neSRC gradNeR_SRC gradNeZ_SRC]

    * Cp = dimensionless pinch coefficient :math:`C_p`
    * neSRC, gradNeR_SRC, gradNeZ_SRC = sources for the electron
      density (m\ :sup:`-3`) and for the :math:`R` and :math:`Z`
      components of its gradient (m\ :sup:`-4`).  Required with
      *bfield_const* and *bfield*, not allowed with *background*.

Examples
--------

.. code-block:: text

   fix pd background file plasma.h5 static yes
   fix fcd cross_field_diffusion 1 background pd D_perp 1.0
   fix fcd cross_field_diffusion 1 background pd bohm scale 0.1 pinch -50.0 0.0
   fix fcd cross_field_diffusion 1 background pd D_perp 0.5 gradient_pinch 2.0
   fix fcd cross_field_diffusion 1 bfield_const 0.0 0.0 2.5 D_perp 1.0

Description
-----------

Displace charged particles across the magnetic field to model
anomalous cross-field transport.  The displacement is the sum of a
random-walk step with diffusivity :math:`D_\perp` and optional
deterministic pinch steps:

.. math::
   :label: xdiff-step

   \Delta\mathbf{x} = \Delta\mathbf{x}_D
      + \left(\mathbf{V}_\mathrm{p} + \mathbf{V}_\psi
      + \mathbf{V}_\nabla\right)\Delta t_N ,
   \qquad \Delta t_N = N\,\Delta t .

Symbols are defined on the :ref:`notation <notation>` page.  Only the
position changes.  The velocity of the particle is not modified.

Random walk
^^^^^^^^^^^

In 3d the random step is taken in the plane perpendicular to
:math:`\mathbf{B}`,

.. math::
   :label: xdiff-walk

   \Delta\mathbf{x}_D = \sqrt{2 D_\perp \Delta t_N}\,
      \left(\xi_1\,\hat{\mathbf{e}}_1 + \xi_2\,\hat{\mathbf{e}}_2\right) ,

where :math:`\xi_1` and :math:`\xi_2` are independent unit normal
numbers.  The unit vector :math:`\hat{\mathbf{e}}_1` is the part of
:math:`\hat{\mathbf{x}}` perpendicular to :math:`\hat{\mathbf{b}}`,
normalized, or that of :math:`\hat{\mathbf{y}}` if
:math:`|\hat{b}_x| > 0.9`, and
:math:`\hat{\mathbf{e}}_2 = \hat{\mathbf{b}}\times\hat{\mathbf{e}}_1`.
The mean square displacement in each perpendicular direction is
:math:`2 D_\perp \Delta t_N`.

In 2d the same expression is used with

.. math::

   \hat{\mathbf{e}}_1 = \frac{(-B_y,\; B_x)}{B_\mathrm{p}} ,
   \qquad
   \hat{\mathbf{e}}_2 = \frac{B_\phi}{B}\,\hat{\mathbf{b}}_\mathrm{p}
      - \frac{B_\mathrm{p}}{B}\,\hat{\boldsymbol{\phi}} ,
   \qquad
   B_\mathrm{p} = \sqrt{B_x^2 + B_y^2} .

Here :math:`B_x` and :math:`B_y` are the field components in the
simulation plane, :math:`B_\phi` is the toroidal component, and
:math:`\hat{\mathbf{b}}_\mathrm{p} = (B_x, B_y)/B_\mathrm{p}`.  The
vector :math:`\hat{\mathbf{e}}_1` is normal to the flux surface and
:math:`\hat{\mathbf{e}}_2` is the binormal, which lies in the
surface.  The in-plane part of the binormal step moves the particle
along the poloidal field.  Its toroidal part has length
:math:`s = \sqrt{2 D_\perp \Delta t_N}\,\xi_2\,B_\mathrm{p}/B`.  In an
axisymmetric simulation this toroidal displacement is folded back
into the plane as an outward shift in major radius,
:math:`R \to \sqrt{R^2 + s^2}`.  In a planar 2d simulation it is
dropped.  With *binormal no* the :math:`\xi_2` step is omitted and the
walk is along :math:`\hat{\mathbf{e}}_1` only.  A straight step of
length :math:`l` tangent to a flux surface with radius of curvature
:math:`r_c` ends at the distance :math:`\sqrt{r_c^2 + l^2}` from the
centre of curvature, so the in-plane binormal step displaces particles
towards the outside of the curve on average.  The walk along
:math:`\hat{\mathbf{e}}_1` alone lacks this displacement and is
biased towards the inside of the curve in comparison.  Use *no* only
to reproduce results obtained without the binormal step.  The
*binormal* keyword has no effect in 3d.  A particle with :math:`B_\mathrm{p} = 0` takes
no random step in 2d.

Diffusivity
^^^^^^^^^^^

*D_perp* sets a constant :math:`D_\perp`.  *bohm* uses the Bohm
diffusivity

.. math::
   :label: xdiff-bohm

   D_\perp = S\,\frac{T_e}{16\,e\,B} ,

evaluated at each particle, where :math:`S` is the *scale* value.
With :math:`T_e` in eV and :math:`B` in T this is
:math:`S\,T_e/(16 B)` in m\ :sup:`2`/s.  With a *background* source
:math:`T_e` is sampled from the fix at the particle position.
Otherwise a TeSRC must follow the *bohm* keyword.  If both *D_perp*
and *bohm* are given, the one given last is used.

Pinch velocities
^^^^^^^^^^^^^^^^

*pinch* adds a constant velocity in cylindrical components,

.. math::

   \mathbf{V}_\mathrm{p} = V_R\,\hat{\mathbf{R}} + V_Z\,\hat{\mathbf{Z}} .

The mapping of :math:`\hat{\mathbf{R}}` and :math:`\hat{\mathbf{Z}}`
to the simulation box is given on the
:ref:`fix background <background_coords>` page.  In 3d the radial
direction is taken about the *column_axis* of the background, or about
the z axis through the origin if the field source is not a
*background*.

*pinch_psi* adds a velocity normal to the flux surfaces,

.. math::

   \mathbf{V}_\psi = V\,\frac{\nabla\psi_N}{|\nabla\psi_N|} ,

so that :math:`V < 0` points towards the magnetic axis.  The
gradient is that of the bilinear interpolant of the flux map held by the
:doc:`fix background <fix_background>`.  The term needs a *background*
source with a flux map.  With any other source, and where the particle
lies outside the equilibrium grid, it is zero.

*gradient_pinch* adds a velocity proportional to the local diffusivity
and to the perpendicular density gradient,

.. math::
   :label: xdiff-gradpinch

   \mathbf{V}_\nabla = C_p\, D_\perp\, \frac{\nabla_\perp n_e}{n_e} .

In 3d, :math:`\nabla_\perp n_e = \nabla n_e -
\hat{\mathbf{b}}\,(\hat{\mathbf{b}}\cdot\nabla n_e)`.  In 2d it is the
component of :math:`\nabla n_e` along :math:`\hat{\mathbf{e}}_1`.
Only the :math:`R` and :math:`Z` components of :math:`\nabla n_e` are
used.  The density in the denominator is not allowed to fall below
10\ :sup:`10` m\ :sup:`-3`.  The term is zero unless *D_perp* or
*bohm* is also given.

With a *background* source :math:`\nabla n_e` is a central difference
of the electron density on the regular :math:`(R,Z)` grid of the
background.  A background that holds its plasma only on a mesh, and a
*zones3d* background, have no such grid, and the gradient pinch is
then zero.  With the other sources the density and its gradient are
read from the three listed sources.

Sources and timing
^^^^^^^^^^^^^^^^^^

A per-grid compute source is read in the particle's cell, and the fix
invokes the compute itself on the steps it acts.  A per-particle
source must be a floating-point custom vector.

The displacement is computed at the start of every :math:`N`-th
timestep for each charged particle and is added to the particle move
of that step.  The mover follows the displaced path, so surface
collisions and cell crossings along it are detected.  On the other
steps no displacement is applied.  For a particle advanced by the
guiding-centre pusher the stored guiding centre is shifted by the same
displacement.

Neutral particles are not displaced.  A particle at a position where
:math:`B = 0` is skipped, including its pinch terms.

Restrictions
------------

This command is part of the OPENEDGE package.

At least one of *D_perp*, *bohm*, *pinch* or *gradient_pinch* must be
given.  *pinch_psi* alone does not satisfy this requirement.

:math:`D_\perp` must not be negative.

Related commands
----------------

:doc:`fix force/thermal <fix_force_thermal>`,
:doc:`fix coulomb/background <fix_coulomb_background>`,
:doc:`fix background <fix_background>`

Default
-------

The keyword defaults are *scale* = 1.0 for *bohm* and binormal = yes.
No diffusion or pinch is applied unless its keyword is given.
