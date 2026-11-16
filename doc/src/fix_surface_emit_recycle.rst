.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix surface/emit/recycle

fix surface/emit/recycle command
================================

Syntax
------

.. parsed-literal::

   fix ID surface/emit/recycle mix-ID group-ID fix-ID keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* surface/emit/recycle = style name of this fix command
* mix-ID = ID of the mixture to use when creating particles
* group-ID = ID of the surface group that emits particles
* fix-ID = ID of the :doc:`fix background <fix_background>` that
  supplies the plasma
* one or more keyword/value pairs must be appended
* keyword = *mass* or *R* or *twall* or *twall_species* or *pwi* or
  *impact* or *wall_flux_cutoff* or *wall_flux_species* or *nevery* or
  *region*

  * *mass* value = mass of the main ions (amu), required
  * *R* value = recycling coefficient (0 to 1)
  * *twall* value = emission temperature (K)
  * *twall_species* values = sp1 T1 [sp2 T2 ...]

    * sp = ID of a species in the mixture
    * T = emission temperature of that species (K)

  * *pwi* value = ID of a
    :doc:`surf_react surface/pwi <surf_react_surface_pwi>` model that
    decides the fate of each incident ion
  * *impact* value = *maxwell* or *sheath* = velocity model of the
    incident ions in *pwi* mode
  * *wall_flux_cutoff* value = search radius for wall-flux samples (m)
  * *wall_flux_species* value = ion slot of the wall-flux samples to
    use, counted from 1
  * *nevery* value = Nevery = emit every this many timesteps
  * *region* value = region-ID = emit only at points inside the region

Examples
--------

.. code-block:: text

   fix rec surface/emit/recycle dgas wall pd mass 2.014 R 0.99 twall 400.0
   fix rec surface/emit/recycle dgas wall pd mass 2.014 R 0.98 twall 500.0 &
       twall_species D 23210.0 D2 500.0
   fix rec surface/emit/recycle dion wall pd mass 2.014 pwi wallPWI impact maxwell

Description
-----------

Emit the neutrals that recycle from the wall where the background
plasma strikes it.  The ion flux onto each surface element of the
group is taken from the :doc:`fix background <fix_background>`.  In
the default mode the fix emits thermal neutrals of the mixture at a
rate set by that flux and the recycling coefficient.  In *pwi* mode
it creates the incident ions themselves at the wall and passes them
to a :doc:`surf_react surface/pwi <surf_react_surface_pwi>` model,
which reflects, re-emits or retains each one.

Symbols are defined on the :ref:`notation <notation>` page.

Ion flux to the wall
^^^^^^^^^^^^^^^^^^^^

The fix works on tasks.  A task is the piece of one surface element
that lies inside one grid cell.  The plasma is evaluated at the
midpoint of the task.

If the background carries wall-flux samples from the plasma code, the
flux :math:`\Gamma`, :math:`T_e` and :math:`T_i` at a task are the
inverse-distance weighted means of the three nearest samples within
*wall_flux_cutoff*.  *wall_flux_species* selects the ion slot whose
flux is used.  The flux multiplies the area :math:`A` of the task:
the polygon area in three dimensions, the area of the ring swept by
the line segment in an axisymmetric simulation, and the segment
length in a planar two-dimensional simulation.

Where there is no sample in range, or the background has none, the
flux is reconstructed from the plasma cell next to the wall
[Stangeby2000]_.  The fix prints a warning when the background has no
samples, as is the case for files converted from OEDGE.  The
reconstructed flux is

.. math::
   :label: recycle-flux

   \Gamma = n_e\, c_s \sin\alpha_B ,
   \qquad
   c_s = \sqrt{\frac{T_e + T_i}{m_i}} ,
   \qquad
   \sin\alpha_B = \frac{|\mathbf{B}\cdot\hat{\mathbf{n}}|}{B} ,

where :math:`m_i` is the *mass* value and :math:`\alpha_B` is the
angle between the magnetic field and the surface plane.
:math:`T_i = T_e` is used if the background has no ion temperature,
and :math:`\sin\alpha_B = 1` if it has no magnetic field.  The cell
is the one that the plasma file assigns to the surface element, or
else the wall cell whose centre is nearest to the task.  If the
plasma file gives the wall area that the plasma code attributes to an
element or to a wall cell, that area is used for :math:`A`, divided
among the tasks of the element in proportion to their geometric
areas.  The emitted total then matches the particle balance of the
plasma code however the grid cuts the element.  A task with no cell
emits nothing.

Number of particles
^^^^^^^^^^^^^^^^^^^

On a step where the fix acts, a task creates
:math:`\lfloor n + U \rfloor` particles with

.. math::
   :label: recycle-number

   n = \frac{\dot{N}\,\Delta t_N}{F_\mathrm{num}\, w_\mathrm{cell}} ,
   \qquad
   \dot{N} =
   \begin{cases}
      \tfrac{1}{2}\, R\,\Gamma A , & \text{default mode} , \\
      \Gamma A , & \textit{pwi}\ \text{mode} ,
   \end{cases}

where :math:`\Delta t_N = N\,\Delta t` with :math:`N` the *nevery*
value, and :math:`w_\mathrm{cell}` is the weight of the grid cell set
by the :doc:`global weight <global>` command, or 1.  Every particle
has the default weight :math:`F_\mathrm{num}`.

In the default mode :math:`R` is the *R* value and the fraction
:math:`1 - R` of the ion flux is pumped.  The factor 1/2 is the
number of molecules formed per recycled ion.  The species of each
particle is drawn from the fractions of the mixture, which thereby
set the split between atoms and molecules.

Particles are created at the start of the timestep at points
distributed uniformly over the task.  A point outside the *region*,
if one is given, is rejected.  Each new particle is moved for a
random fraction of one timestep.

Thermal emission
^^^^^^^^^^^^^^^^

In the default mode a particle of mass :math:`m` leaves with

.. math::

   v_n = v_\mathrm{th}\sqrt{-\ln U_1} , \qquad
   v_t = v_\mathrm{th}\sqrt{-\ln U_2} , \qquad
   v_\mathrm{th} = \sqrt{\frac{2k_\mathrm{B}T}{m}} ,

where :math:`v_n` is the component along the normal, into the plasma,
and :math:`v_t` is the magnitude of the tangential component, whose
direction is uniform.  The velocity has the distribution of a
Maxwellian flux at the temperature :math:`T`.  The emission angle
from the normal follows the cosine law, and the mean kinetic energy
is :math:`2k_\mathrm{B}T`.  The rotational and vibrational energies
are sampled at the same temperature.

:math:`T` is the *twall* value.  *twall_species* replaces it for
individual species, for example Franck-Condon atoms at about 23000 K
(2 eV) next to molecules at the wall temperature.  Every species
named must be in the mixture.  The pairs are read until the next
keyword of this fix or the end of the command.  The keywords *impact*
and *perspecies* are not recognized as the end of the list, so they
must not follow *twall_species* directly.

Incident ions in pwi mode
^^^^^^^^^^^^^^^^^^^^^^^^^

With *pwi* the mixture must hold exactly one species, the incident
ion, with charge state :math:`Z > 0` and mass :math:`m`.  Each ion is
created on the element with a velocity directed into the wall, and
the reaction model is applied to it at once.  A reflected or
re-emitted particle stays in the simulation, with the species and
velocity that the model assigns.  A retained ion is deleted.
Sputtering channels of the model and its areal-density bookkeeping
act as for any other impact.  *R* and *twall* are not used: the
recycling coefficient and the wall temperature are those of the
reaction model.

*impact maxwell* samples the ion velocity from a Maxwellian flux at
:math:`T_i` that drifts along the field at the sound speed
:math:`c_s = [(T_e + T_i)/m]^{1/2}` of the ion species.  With
:math:`\sigma = (T_i/m)^{1/2}`, the component towards the wall has the
distribution

.. math::

   f(v_n) \propto v_n\,
      \exp\!\left[-\frac{(v_n - c_s\sin\alpha_B)^2}{2\sigma^2}\right] ,
   \qquad v_n > 0 ,

and the two tangential components are normal with standard deviation
:math:`\sigma`.  One of them has the mean
:math:`c_s\cos\alpha_B`, in a tangential direction chosen at random.
The sheath then accelerates the ion along the normal,

.. math::

   v_n \rightarrow \sqrt{v_n^2 + \frac{2 Z\, e\Delta\phi}{m}} ,
   \qquad
   e\Delta\phi = \frac{T_e}{2}\,
      \ln\!\left[\frac{m}{\pi m_e\,(1 + T_i/T_e)}\right] .

*impact sheath* gives every ion the same speed and angle.  The
thermal energy :math:`2T_i` is directed along the field line, and the
sheath energy is added to the normal motion,

.. math::

   v_t = v_0\cos\alpha_B , \qquad
   v_n = \sqrt{v_0^2\sin^2\alpha_B + \frac{2 Z\, e\Delta\phi}{m}} ,
   \qquad
   v_0 = \sqrt{\frac{4T_i}{m}} ,
   \qquad
   e\Delta\phi = \frac{T_e}{2}\,
      \ln\!\left[\frac{m_i}{2\pi m_e\,(1 + T_i/T_e)}\right] ,

so that the impact energy is :math:`2T_i + Z e\Delta\phi`, the mean
value used by
:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`.
Here :math:`m_i` is the *mass* value.

The fast reflection channel is part of the reaction model and
therefore of *pwi* mode only.  In the default mode all recycled
particles are thermal.

Tasks are rebuilt when the grid changes and when the background
plasma is reloaded.

Restart, output info
^^^^^^^^^^^^^^^^^^^^

No information about this fix is written to
:doc:`binary restart files <restart>`.

This fix computes a global vector of length 2, which can be accessed
by various output commands.  The first element is the number of
particles emitted on the most recent emission step.  The second is
the cumulative number of particles emitted.  In *pwi* mode, ions that
the wall retains are not counted.

Restrictions
------------

This command is part of the OPENEDGE package.

It requires explicit surface elements and a
:doc:`fix background <fix_background>` with an :math:`(R,Z)` plasma.
A native three-dimensional background is not supported.

*mass* must be given.  *R* and *twall* must be given unless *pwi* is
used.

The *perspecies* keyword of :doc:`fix emit/surf <fix_emit_surf>` is
accepted but has no effect.  Species are always drawn from the
mixture fractions.

Related commands
----------------

:doc:`fix emit/surf <fix_emit_surf>`,
:doc:`fix surface/emit/source <fix_surface_emit_source>`,
:doc:`fix surface/emit/puff <fix_surface_emit_puff>`,
:doc:`surf_react surface/pwi <surf_react_surface_pwi>`,
:doc:`fix background <fix_background>`

Default
-------

The option defaults are impact = maxwell, wall_flux_species = 1 and
nevery = 1.  *wall_flux_cutoff* defaults to twice the largest
distance between neighbouring wall-flux samples.  There is no default
for mass, R, twall, twall_species, pwi or region.
