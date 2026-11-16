.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix particulate/emit

fix particulate/emit command
============================

Syntax
------

.. parsed-literal::

   fix ID particulate/emit mix-ID group-ID keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* particulate/emit = style name of this fix command
* mix-ID = ID of the mixture whose particulate species are emitted
* group-ID = ID of the surface group that emits
* zero or more keyword/value pairs may be appended
* keyword = *nevery* or *n* or *perspecies* or *region* or *normal* or
  *magVelocity* or *vmin* or *vmax* or *incidentAngle* or *cone* or
  *angle* or *dir* or *planar* or *nweight* or *subsonic* or *twopass*
  or *custom*

  * *nevery* value = emit every this many timesteps
  * *n* value = Np or v_name

    * Np = number of particles emitted per event (integer).  With 0
      the flow rate of the mixture is used.
    * v_name = :doc:`equal-style variable <variable>` that gives Np

  * *perspecies* value = *yes* or *no*
  * *region* value = region-ID = emit only inside this
    :doc:`region <region>`
  * *normal* value = *yes* or *no* = apply the stream velocity of the
    mixture along the surface normal, as in
    :doc:`fix emit/surf <fix_emit_surf>`
  * *magVelocity* value = launch speed (m/s)
  * *vmin* value = lower bound of a uniform range of launch speeds
    (m/s)
  * *vmax* value = upper bound of that range (m/s)
  * *incidentAngle* value = fixed polar angle from the launch axis
    (degrees, 0 to 180)
  * *cone* value = half-angle of a cone about the launch axis, sampled
    uniformly in solid angle (degrees, 0 to 90)
  * *angle* value = *cosine* or *uniform* = distribution of the polar
    angle
  * *dir* values = dR dZ dphi = fixed launch axis that replaces the
    local surface normal
  * *planar* value = *yes* or *no* = set the out-of-plane velocity
    component to zero (2d only)
  * *nweight* value = number of physical particulates represented by
    one emitted particle (> 0)
  * *subsonic* values = Psub Tsub = as in
    :doc:`fix emit/surf <fix_emit_surf>`
  * *twopass* = no value, as in :doc:`fix emit/surf <fix_emit_surf>`
  * *custom* values = attribute s_name

    * attribute = *density* or *temperature* or *vstream* or *speed* or
      *fractions*
    * s_name = per-surf custom attribute that supplies the quantity

Examples
--------

.. code-block:: text

   fix source particulate/emit particulates injector nevery 1 n 10 &
       perspecies no vmin 0.1 vmax 2.0 angle cosine planar yes &
       dir 0.0 -1.0 0.0
   fix drop particulate/emit powder dropper n 5 perspecies no &
       magVelocity 1.0 cone 10 nweight 1e4

Description
-----------

Emit particulates from the surface elements of a surface group.  Each
emitted particle takes its radius :math:`r_d`, mass :math:`m_d` and
initial temperature :math:`T_d` from the :doc:`species <species>`
definition of its species in the mixture.  The position is sampled
on the part of a surface element that lies in each grid cell,
uniformly in length along a line segment (2d) and uniformly in area on
a triangle (3d).  In an axisymmetric run the position along a segment
is weighted by the radius, as in :doc:`fix emit/surf <fix_emit_surf>`,
so the particles are uniform per unit area of the surface of
revolution.  The particulate model is described in
[Diaw2026]_.

Number of particles
^^^^^^^^^^^^^^^^^^^

The *nevery* keyword sets the interval between emission events.  The
*n* keyword sets the number of particles per event.

* *n 0* (the default) uses the flow rate of the mixture through each
  element, as :doc:`fix emit/surf <fix_emit_surf>` does.
* A positive integer Np is shared among the emitting elements in
  proportion to their area.  An element with the fraction :math:`f`
  of the total area emits the integer part of :math:`f N_p + U`.
* *n v_name* evaluates an equal-style variable at each event and uses
  its value as Np.  The value must be positive.

A constant or variable count requires *perspecies no*.

The *nweight* keyword declares that one emitted particle represents
:math:`w_d` physical particulates.  The value is stored in the custom
per-particle attribute *grain_nweight*.  It multiplies the vapour
source and the atom tallies of
:doc:`fix particulate/thermal <fix_particulate_thermal>`.  The
trajectory, charge and temperature of the particle are those of a
single particulate.

Launch velocity
^^^^^^^^^^^^^^^

If a positive *magVelocity* or a *vmin*/*vmax* range is given, the
launch speed :math:`v_0` is set by the user.  *magVelocity* gives one
speed.  With *vmin* and *vmax*,

.. math::

   v_0 = v_\mathrm{min} + (v_\mathrm{max} - v_\mathrm{min})\,U

for each particle.  Set both bounds, since a bound that is not given
is 0.  If both forms are given the range is used.

The direction is sampled about a launch axis
:math:`\hat{\mathbf{n}}`, which is the local surface normal unless
*dir* is given.  With polar angle :math:`\vartheta` from the axis and
azimuth :math:`\varphi = 2\pi U`,

.. math::
   :label: pemit-v

   \mathbf{v} = v_0\left[\cos\vartheta\,\hat{\mathbf{n}}
      + \sin\vartheta\left(\sin\varphi\,\hat{\mathbf{t}}_1
      + \cos\varphi\,\hat{\mathbf{t}}_2\right)\right] ,

where :math:`\hat{\mathbf{t}}_1` and :math:`\hat{\mathbf{t}}_2` are
two unit vectors normal to the axis.  The polar angle is set by the
first of these keywords that is present:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Keyword
     - Polar angle
   * - *incidentAngle*
     - :math:`\vartheta` equal to the given angle
   * - *cone*
     - :math:`\cos\vartheta = 1 - U\,(1 - \cos\vartheta_\mathrm{c})`
       with :math:`\vartheta_\mathrm{c}` the half-angle, which is
       uniform in solid angle inside the cone
   * - *angle cosine*
     - :math:`\vartheta = \arcsin\sqrt{U}`, the cosine law of Knudsen
       and Lambert
   * - *angle uniform*
     - :math:`\vartheta = \tfrac{\pi}{2}\,U`, uniform in the polar
       angle between 0 and 90 degrees

Unless *normal yes* is set, the stream velocity of the mixture is
added to :math:`\mathbf{v}`.

If neither *magVelocity* nor a speed range is given, the velocity is
sampled from the thermal distribution and stream velocity of the
mixture as in :doc:`fix emit/surf <fix_emit_surf>`, and the
*incidentAngle*, *cone* and *angle* keywords have no effect.

The *dir* keyword replaces the surface normal by a fixed launch axis
for all elements, for example the vertical for a gravity-fed dropper.
Its three values are the :math:`(R, Z, \phi)` components.  They are
mapped onto the coordinate order of a two-dimensional planar or
axisymmetric run.  In a three-dimensional run they are taken as the
:math:`(x, y, z)` components.  The vector need not be normalized.

With *planar yes* the out-of-plane velocity component is zero.  The
tangential component :math:`v_0\sin\vartheta` then lies in the
simulation plane, with either sign at equal probability.  With *dir*
the dphi component must be zero.

Other keywords
^^^^^^^^^^^^^^

The *region*, *normal*, *subsonic* and *twopass* keywords act as for
:doc:`fix emit/surf <fix_emit_surf>`.  The *custom* keyword takes a
per-surf custom attribute, with the prefix s\_, for the number
density, temperature, stream velocity, speed or species fractions of
the emitting element, also as in that command, with one difference.  A
custom *temperature* sets the rotational and vibrational energy of the
emitted particle and the temperature passed to fixes that initialize
custom particle attributes.  It does not change the thermal speed, so
the flow rate and the thermal velocity still use the temperature of
the mixture.

Restrictions
------------

This command is part of the OPENEDGE package.

It requires explicit surface elements and cannot be used with implicit
surfaces.

*planar yes* requires :doc:`dimension <dimension>` 2.

A constant or variable *n* cannot be combined with *perspecies yes*.
The *custom* keyword requires *n 0* and cannot be combined with
*subsonic*.

Cell weighting, set by *weight cell* of the :doc:`global <global>`
command, cannot be used with this fix.  It would clone or delete
particulates as they cross cells.

Related commands
----------------

:doc:`fix emit/surf <fix_emit_surf>`,
:doc:`fix particulate/thermal <fix_particulate_thermal>`,
:doc:`fix particulate/drag <fix_particulate_drag>`,
:doc:`fix particulate/charge <fix_particulate_charge>`,
:doc:`material <material>`

Default
-------

The option defaults are nevery = 1, n = 0, perspecies = yes, normal =
no, angle = uniform, planar = no, nweight = 1, no region, no dir, and
no launch speed, so that the thermal and stream velocity of the
mixture are used.
