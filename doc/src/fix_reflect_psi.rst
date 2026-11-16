.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix reflect/psi

fix reflect/psi command
=======================

Syntax
------

.. parsed-literal::

   fix ID reflect/psi source keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* reflect/psi = style name of this fix command
* source = *equ* or *background*, exactly one is required

  * *equ* file = equilibrium text file in the SOLPS layout with the
    poloidal-flux map (*geqdsk* is accepted in place of *equ*)
  * *background* fix-ID = ID of a :doc:`fix background <fix_background>`
    that holds the poloidal-flux map

* zero or more keyword/value pairs may be appended
* keyword = *psi_norm* or *action* or *mixture*

  * *psi_norm* value = threshold :math:`\psi_\mathrm{c}` in normalized
    flux.  Particles with :math:`\psi_N < \psi_\mathrm{c}` are
    intercepted.
  * *action* value = *reflect* or *absorb*
  * *mixture* value = mixture-ID = apply only to the species of this
    mixture

Examples
--------

.. code-block:: text

   fix fcore reflect/psi equ input/g174310.03500.equ psi_norm 0.926
   fix pd background file plasma.h5 static yes
   fix fcore reflect/psi background pd action absorb
   fix fcore reflect/psi background pd psi_norm 0.97 action absorb mixture allLi
   fix coreout ave/time 10000 1 10000 f_fcore mode vector file core_flux

Description
-----------

Place a boundary on a flux surface inside the simulation domain.  A
particle whose normalized poloidal flux

.. math::
   :label: reflect-psin

   \psi_N = \frac{\psi - \psi_\mathrm{axis}}
                 {\psi_\mathrm{sep} - \psi_\mathrm{axis}}

is below the threshold :math:`\psi_\mathrm{c}` at the end of a move is
reflected back across the surface or deleted.  This gives an inner
boundary on the core side without surface elements in the closed-flux
region, and the boundary can be restricted to selected species.

The test is made in the particle move on every timestep, for every
particle in scope.  There is no *Nevery*.  The flux :math:`\psi` at
the particle position :math:`(R, Z)` is interpolated bilinearly on the
equilibrium grid.  A point outside that grid takes the value at the
nearest grid edge.

Flux map
^^^^^^^^

With *equ* the map is read from a SOLPS equilibrium text file.  The
reader looks for the entries *jm=*, *km=* and *psib=* and for the
arrays that follow the markers *r(1:jm);*, *z(1:km);* and
*((psi(j,k)-psib,j=1,jm),k=1,km)*.  :math:`\psi_\mathrm{sep}` is
*psib*.  :math:`\psi_\mathrm{axis}` is the minimum of :math:`\psi`
over the central half of the grid in each direction, so
:math:`\psi_N = 0` on the magnetic axis and 1 on the separatrix.

With *background* the map, :math:`\psi_\mathrm{axis}` and
:math:`\psi_\mathrm{sep}` are those held by the named
:doc:`fix background <fix_background>`.  What
:math:`\psi_\mathrm{axis}` stands for depends on the plasma file.  A
file with an *equilibrium* group stores it in the dataset
*equilibrium/psi_axis*, and the minimum of the flux map is used if
that dataset is absent.  The SOLEDGE3X converter
tools/converters/convert_s3x_plasma.py writes the extremum of the
flux map on the core side of the separatrix, which is the flux at the
magnetic axis if the map contains the axis.  :math:`\psi_N` is then
nowhere negative and the default threshold of 0 intercepts nothing, so
*psi_norm* must be set.  A file without that group that holds the
datasets *psi*, *psicore* and *psisep* on its regular grid uses
*psicore*, the flux of the core boundary of the plasma grid, as the
axis value.  Then :math:`\psi_N = 0` on that boundary and
:math:`\psi_N < 0` inside it, which is the region the default
threshold selects.

Reflection
^^^^^^^^^^

With *action reflect* the point where the straight path of the move
crosses the surface :math:`\psi_N = \psi_\mathrm{c}` from outside to
inside is found by bisection.  The unit normal there is

.. math::

   \hat{\mathbf{n}} = \frac{\nabla\psi_N}{|\nabla\psi_N|} ,

from the gradient of the bilinear interpolant, and the velocity is
reflected specularly,

.. math::

   \mathbf{v}' = \mathbf{v} - 2\,(\mathbf{v}\cdot\hat{\mathbf{n}})\,
                 \hat{\mathbf{n}} .

The part of the path beyond the crossing is mirrored in the tangent
plane in the same way.  The speed and the tangential velocity are
conserved in 2d, axisymmetric and 3d geometry.  If the mirrored end
point is still inside the surface, the particle is placed just outside
the crossing point.  A displacement added to the move by
:doc:`fix cross_field_diffusion <fix_cross_field_diffusion>` is part
of the path that is tested but not of the velocity that is reflected.

A particle can start a move inside the surface, for instance when it
is created there by an emission or a volume reaction.  No crossing
exists for such a move.  The move is rejected, the radial velocity
:math:`v_R` is reversed, and the particle is moved along
:math:`\nabla\psi_N` to a point just outside the surface.  These
events are counted in the global scalar of the fix.  If the
environment variable OE_PSI_STRICT is set to 1, the run stops with an
error at the first such event.

Absorption
^^^^^^^^^^

With *action absorb* a particle whose end point is inside the surface
is deleted.  The removal is recorded at the end point, so its position
and time overshoot the crossing by up to one timestep.  Check the
convergence of an absorbed flux with the timestep when it is a
reported quantity.  Sputtered impurities that enter the closed-flux
region do not normally return within the simulated time, so *absorb*
is the usual choice for a core boundary.

The *mixture* keyword restricts the boundary to the species of a
mixture.  All other species cross the surface freely.  In a
particulate run, for example, atoms and ions can be absorbed at the
core boundary while grains continue into the core plasma.

Output info
-----------

This fix computes a global vector with three entries per species, or
per mixture group if *mixture* is used.  For species or group
:math:`i`, counted from 1:

.. list-table::
   :header-rows: 1
   :widths: 14 66 20

   * - Entry
     - Quantity
     - Unit
   * - :math:`3i-2`
     - cumulative number of absorbed simulation particles
     -
   * - :math:`3i-1`
     - cumulative number of absorbed real particles.  Each simulation
       particle counts with its *pweight* attribute from
       :doc:`fix particle/weight <fix_particle_weight>` if that
       attribute exists and is positive, and with
       :math:`F_\mathrm{num}` otherwise.
     -
   * - :math:`3i`
     - entry :math:`3i-1` divided by the simulated time since the
       first run with this fix
     - s\ :sup:`-1`

For a mixture with the groups Li, Li+ and Li2+, entries 1 to 3 belong
to Li, 4 to 6 to Li+ and 7 to 9 to Li2+.  The vector can be written
with :doc:`fix ave/time <fix_ave_time>` as in the example above.  The
difference of the cumulative real-particle count between two outputs
gives the removal rate over that interval.

The fix also computes a global scalar, the cumulative number of
*reflect* moves rejected because the particle started inside the
surface, summed over species.

Restrictions
------------

This command is part of the OPENEDGE package.

Exactly one of *equ* or *background* must be given.  With *background*
the named fix must hold a flux map with its axis and separatrix
values, otherwise this fix stops with an error.  The *equ* reader
expects the SOLPS text layout described above, with increasing
:math:`R` and :math:`Z` grids.

Only one fix reflect/psi can be active in a run.

Related commands
----------------

:doc:`fix background <fix_background>`,
:doc:`fix particle/weight <fix_particle_weight>`,
:doc:`fix ave/time <fix_ave_time>`

Default
-------

The keyword defaults are action = reflect, psi_norm = 0.926 with *equ*
and psi_norm = 0.0 with *background*.  Without *mixture* the boundary
acts on all species.
