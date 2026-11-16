.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: compute surface/physical/sputter

compute surface/physical/sputter command
========================================

Syntax
------

.. parsed-literal::

   compute ID surface/physical/sputter group-ID source [yield] keyword values ...

* ID is documented in the :doc:`compute <compute>` command
* surface/physical/sputter = style name of this compute
* group-ID = group ID of the surface elements to operate on
* source = *background* fix-ID

  * *background* fix-ID = ID of the
    :doc:`fix background <fix_background>` that supplies the plasma

* yield = optional, *eckstein* name or path

  * *eckstein* name = use the analytic fit of the single pair *name*
    (for example O_on_W) for all projectiles, instead of *target* and
    *projectiles*
  * path = HDF5 file with one yield table (datasets E, A and spyld)
    used for all projectiles

* one or more keyword/value pairs must be appended, at least one of
  them an output keyword
* keyword = *target* or *target_like* or *yield_scale* or
  *projectiles* or *impurity* or *mass_amu* or *us* or *tsurf* or
  *compound* or *conc* or *roughness_dm* or *projectile_slots* or
  *static* or *incidence* or *visibility* or *iead* or *iead_markers*
  or *iead_bins* or *iead_drift* or *iead_dump* or *sheath_waveform* or
  *wall_flux_cutoff* or *boundary* or *debug_interp* or an output
  keyword

  * *target* value = element symbol of the wall material (W, Li, B,
    ...)
  * *target_like* value = element whose yield data stand in for
    *target*
  * *yield_scale* value = factor applied to every yield (> 0)
  * *projectiles* value = comma-separated list of projectile
    elements, or *all*
  * *impurity* values = mass frac Zmax f1 ... fZmax

    * mass = mass of a virtual impurity projectile (amu)
    * frac = its density as a fraction of the electron density,
      0 < frac < 1
    * Zmax = number of charge states
    * f1 ... fZmax = fractions of the charge states 1 to Zmax, which
      must add up to 1

  * *mass_amu* value = mass of the main ions (amu), used for the sound
    speed and the sheath potential
  * *us* value = surface binding energy that replaces the one of the
    analytic fit (eV)
  * *tsurf* value = name of a custom per-surf vector that holds the
    surface temperature (degrees C)
  * *compound* value = mixture name of composition-resolved yield
    tables
  * *conc* values = attr species

    * attr = custom per-surf array that holds the surface composition
    * species = species whose column is the composition coordinate

  * *roughness_dm* value = mean surface inclination angle
    :math:`\delta_m` (degrees), :math:`0 \le \delta_m < 90`
  * *projectile_slots* values = lo hi = first and last ion slot of the
    plasma, counted from 1, that are treated as projectiles
  * *static* value = *yes* or *no*
  * *incidence* value = *poloidal* or *full3d* or *directed3d*
  * *visibility* value = name of a custom per-surf vector that holds
    the visible fraction of each element, from 0 (shadowed) to 1
  * *iead* value = *auto* or *inline* or *none* or the path of an
    IEAD table file
  * *iead_markers* value = number of test ions per inline
    distribution (at least 10)
  * *iead_bins* values = nE ntheta = number of energy and angle bins
    of an inline distribution (at least 2 each)
  * *iead_drift* value = *flow* or *cs* or *cs2* = drift speed of the
    test ions at the sheath entrance
  * *iead_dump* value = file to which the inline distributions are
    written
  * *sheath_waveform* values = attr frequency nphase

    * attr = custom per-surf array with 3 columns: DC voltage (V),
      RF amplitude (V) and phase (rad)
    * frequency = frequency of the RF sheath voltage (Hz, >= 0)
    * nphase = number of phases over which the yield is averaged

  * *wall_flux_cutoff* value = search radius for wall-flux samples (m)
  * *boundary* value = file with the :math:`(R,Z)` polygon that
    bounds the plasma
  * *debug_interp* values = E theta = print the interpolated yield of
    the yield file at energy E (eV) and angle theta (degrees) when the
    compute is initialized

* output keyword = *nflux_species* or *incident_angle_species* or
  *incident_energy_species* or *sputter_yield_species* or
  *sputter_flux_species* or *sputter_flux_total* or *erosion_flux* or
  *sputter_rate_total* or *erosion_rate*

  * *nflux_species* value = slot or *all* = incident flux of an ion
    slot (m\ :sup:`-2` s\ :sup:`-1`)
  * *incident_angle_species* value = slot or *all* = angle between the
    magnetic field and the surface (degrees)
  * *incident_energy_species* value = slot or *all* = mean impact
    energy (eV)
  * *sputter_yield_species* value = slot or *all* = sputter yield
  * *sputter_flux_species* value = slot or *all* = sputtered flux
    (m\ :sup:`-2` s\ :sup:`-1`)
  * *sputter_flux_total* or *erosion_flux* = gross erosion flux of all
    projectiles (m\ :sup:`-2` s\ :sup:`-1`)
  * *sputter_rate_total* or *erosion_rate* = erosion flux times
    element area (s\ :sup:`-1`)

Examples
--------

.. code-block:: text

   compute cpmi surface/physical/sputter plate background pd &
           target W projectiles all static yes erosion_flux
   compute cpmiLi surface/physical/sputter div_outer background pd &
           target Li projectiles D,Ne,Li tsurf Tsurf_lm iead none static yes &
           projectile_slots 1 14 erosion_flux sputter_flux_species all
   compute cpmiWd surface/physical/sputter wall background pd &
           target Wd target_like W yield_scale 2.0 projectiles D,O erosion_flux

Description
-----------

Compute the gross erosion of surface elements by physical sputtering
due to the ions of the background plasma.  The background ions are
not simulation particles, so nothing is tallied from particle
impacts.  The flux, impact energy and impact angle of each ion
species are derived from the plasma state at the element, and the
erosion flux is the sum of flux times yield over the projectiles.
The erosion flux is the usual source for
:doc:`fix surface/emit/source <fix_surface_emit_source>` and the usual
erosion term of
:doc:`surf_react surface/pwi <surf_react_surface_pwi>`.

Symbols are defined on the :ref:`notation <notation>` page.  An ion
slot :math:`s` is one ion species of the plasma file, for example one
charge state of an impurity, with density :math:`n_s` and charge
state :math:`Z_s`.  Slots are counted from 1 in the input script.

The plasma comes from the :doc:`fix background <fix_background>`
named by *background*, which also supplies the magnetic field.

Incident flux
^^^^^^^^^^^^^

The magnetic field meets an element with unit normal
:math:`\hat{\mathbf{n}}` at the angle :math:`\alpha_B` to the surface
plane,

.. math::
   :label: sputter-alpha

   \sin\alpha_B = \frac{|\mathbf{B}\cdot\hat{\mathbf{n}}|}{B} .

The flux of slot :math:`s` onto the element is the flow at the sound
speed along the field, projected on the normal [Stangeby2000]_,

.. math::
   :label: sputter-flux

   \Gamma_s = V\, n_s\, c_s \sin\alpha_B ,
   \qquad
   c_s = \sqrt{\frac{T_e + T_i}{m_i}} ,

where :math:`m_i` is the main-ion mass given by *mass_amu* and
:math:`V` is the *visibility* value of the element, or 1.  The plasma
state is taken at the midpoint of the element: from the plasma cell
next to the wall if the background has an unstructured mesh, and
otherwise by bilinear interpolation on its :math:`(R,Z)` grid.

If the background carries wall-flux samples from the plasma code,
these replace the reconstruction of Eq. :eq:`sputter-flux`.  The
flux, :math:`T_e`, :math:`T_i` and :math:`\mathbf{B}` at an element
are then the inverse-distance weighted means of the three nearest
samples within *wall_flux_cutoff*.  Eq. :eq:`sputter-flux` is used
for an element with no sample in range.  The compute prints a warning
when the background has no samples at all, as is the case for files
converted from OEDGE.

The interpolation does not conserve the particle rate of the samples
when the elements are finer or coarser than the sample spacing.  If
the *wall_flux_rescale* option of
:doc:`fix background <fix_background>` is active, the interpolated
flux of slot :math:`s` on an element is multiplied by the factor
:math:`C_{s,k}` of the sample :math:`k` nearest to it,

.. math::
   :label: sputter-rescale

   C_{s,k} = \frac{\Gamma_{s,k}\,A_k}
                  {\sum_{j \in k} \Gamma_{s,j}^\mathrm{int}\,A_j} ,

where :math:`\Gamma_{s,k}` and :math:`A_k` are the flux and area of
the sample, and the sum runs over the elements :math:`j` whose
nearest sample is :math:`k`, with interpolated flux
:math:`\Gamma_{s,j}^\mathrm{int}` and area :math:`A_j`.  The flux
times area of these elements then adds up to that of the sample.  The
factors are computed once from all surface elements with the
*wall_flux_cutoff* of this compute.  A sample with no element in
range contributes nothing, and the number of such samples is printed.
:math:`T_e`, :math:`T_i` and :math:`\mathbf{B}` are not rescaled.

The *incidence* keyword selects how :math:`\sin\alpha_B` is formed.

* *poloidal* uses only the :math:`R` and :math:`Z` components of
  :math:`\mathbf{B}` in the numerator of Eq. :eq:`sputter-alpha`.
  The spelling *legacy* is accepted for it.
* *full3d* uses the full vectors, including the toroidal component of
  the field and of the normal.
* *directed3d* also uses the sign of the parallel flow
  :math:`u_\parallel` of the slot,
  :math:`\sin\alpha_B = \max(0,\,
  -\mathrm{sign}(u_\parallel)\,\hat{\mathbf{b}}\cdot\hat{\mathbf{n}})`.
  An element that faces away from the arriving ions receives no flux.

With *full3d* and *directed3d* a wall-flux sample is multiplied by
the ratio of the selected :math:`\sin\alpha_B` to the *poloidal* one,
since the samples are fluxes onto the axisymmetric wall.

*visibility* multiplies every incident flux by a fixed fraction for
each element, for example a shadowing mask from a field-line trace.
Values are limited to the interval [0,1].

*boundary* names a text file with one :math:`R\ Z` pair per line,
with # for comments, that forms a polygon.  For a background without
a mesh, elements whose midpoint is outside the polygon are skipped.

Impact energy and angle
^^^^^^^^^^^^^^^^^^^^^^^

The mean impact energy of slot :math:`s` is the thermal energy flux
per ion plus the energy gained in the sheath of a floating wall
[Chankin2014]_,

.. math::
   :label: sputter-energy

   E_0 = 2\,T_i + Z_s\, e\Delta\phi ,
   \qquad
   e\Delta\phi = \frac{T_e}{2}\,
      \ln\!\left[\frac{m_i}{2\pi m_e\,(1 + T_i/T_e)}\right] .

The sheath drop depends on the main-ion mass :math:`m_i`, not on the
mass of the projectile.  For deuterium and :math:`T_i = T_e` it is
:math:`2.84\,T_e`.

The mean impact angle from the surface normal is

.. math::
   :label: sputter-theta

   \theta = 90^\circ - \alpha_B .

.. note::

   The *incident_angle_species* output is :math:`\alpha_B`, the angle
   between the magnetic field and the surface plane in degrees.  It is
   small for a grazing field line.  The yields are evaluated with
   :math:`\theta` of Eq. :eq:`sputter-theta`, which is measured from
   the surface normal.

*sheath_waveform* adds a biased or RF-driven wall.  The custom
per-surf array holds a DC voltage :math:`V_\mathrm{dc}`, an RF
amplitude :math:`V_\mathrm{rf}` and a phase :math:`\varphi_0` for
each element.  At phase :math:`\varphi` the impact energy is

.. math::

   E_0(\varphi) = E_0 + Z_s\,\max(0,\,-V_\mathrm{wall}) ,
   \qquad
   V_\mathrm{wall} = V_\mathrm{dc} + V_\mathrm{rf}\sin(\varphi + \varphi_0) ,

so only a negative wall voltage adds to the floating-sheath energy.
The energy and the yield are averaged over the *nphase* phases
:math:`\varphi_k = 2\pi k/n_\mathrm{phase}`.  With *nphase* = 1 the
phase is :math:`2\pi f t` at the simulation time :math:`t`, the
result is time resolved, and *static yes* is switched off.

Sputter yield
^^^^^^^^^^^^^

*target* names the wall element and *projectiles* the projectile
elements, or *all* for every ion element of the background.  Every
slot of an element, whatever its charge state, uses the data of that
element.  For each pair the yield :math:`Y(E_0,\theta)` comes from
the table /surface/sputter/<projectile>_on_<target> of the file
processes.h5 if it exists.  A pair without a table uses the analytic
fit of Eqs. :eq:`pwi-bohdansky` and :eq:`pwi-yamamura` on the
:doc:`surf_react surface/pwi <surf_react_surface_pwi>` page, with the
built-in parameters of the same name, and a warning is printed
because the fit is unreliable at grazing incidence.  A pair with
neither is an error.  Tables are interpolated as described on that
page.

*target_like* names an element whose data stand in for a target that
has none of its own, for example a redeposited material that
surf_react surface/pwi tracks with *deposit_as* (*target* Wd
*target_like* W).  A table named after *target* is still preferred.
*yield_scale* multiplies every yield, for example for a weakly bound
deposit.  *us* replaces the surface binding energy of the analytic
fit and rescales the threshold energy of each pair in the same
proportion.  It has no effect on tables.  *roughness_dm* evaluates
the yields at :math:`\theta_\mathrm{eff} = \max(0,\,\theta - \delta_m)`.

A table can have a temperature axis, as the liquid-lithium tables
do.  With *tsurf* it is interpolated linearly at the temperature of
each element and held constant outside the table range.  Without
*tsurf* it is evaluated at its lowest temperature.  The *tsurf*
vector is read in degrees C, the unit of the Tsurf_lm vector written
by :doc:`fix surface/state/lm <fix_surface_state_lm>`, and 273.15 is
added to match the temperature axis of the tables, which is in K.

*compound* selects composition-resolved tables named
<projectile>_on_<mix>_<target>.  The composition coordinate is read
for each element from the column of the named species in the *conc*
array, usually the concentration array of surf_react surface/pwi.
There is no analytic fit in this mode, so every table must exist.

Two older forms give one yield function to all projectile slots: the
*eckstein* name form uses the analytic fit of one pair, and a file
path reads a table with energy axis E (eV), angle axis A (degrees)
and yields spyld, interpolated bilinearly.  *debug_interp* prints the
interpolated value of such a file at the given energy and angle, and
may be repeated.

*projectile_slots* limits the slots that sputter.  The default is
slot 2 to the last slot, which leaves out the main ion in slot 1.
With *target* and *projectiles* the range is reset at initialization
to run from the first to the last slot whose element is in the
projectile list, and only slots of listed elements have a non-zero
yield.

*impurity* adds a projectile that is not in the background.  Its
density is *frac* times :math:`n_e`, divided over Zmax charge states
with the given fractions.  Its flux follows Eq. :eq:`sputter-flux`
with the impurity mass in the sound speed, and its energy follows Eq.
:eq:`sputter-energy`.  Its yield is that of the *eckstein* entry or
of the yield file, evaluated at the mean impact energy and angle.
The impurity contributes to the total erosion flux only.

The tables selected by *target* and *projectiles* are not used for
the impurity, and *impurity* together with *target* is an error.

Average over the impact distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ions do not all arrive with the mean energy and at the angle of the
field line.  Gyration and the electric field of the sheath turn them
towards the normal and spread their energies.  Because the yield
varies steeply with both, the *iead* keyword replaces
:math:`Y(E_0,\theta)` by an average over the distribution
:math:`f(E,\theta)` of impact energy and angle,

.. math::
   :label: sputter-ybar

   \bar{Y} = \sum_{j,k} f_{jk}\; Y(E_j,\theta_k) ,
   \qquad \sum_{j,k} f_{jk} = 1 .

*iead auto* reads :math:`f` from the IEAD database in the directory
iead of the OpenEdge database.  A table is an HDF5 file with these
datasets:

.. list-table::
   :header-rows: 1
   :widths: 30 26 44

   * - Dataset
     - Shape
     - Content
   * - tau_grid
     - (ntau)
     - :math:`T_i/T_e`
   * - psi_grid_deg
     - (npsi)
     - angle of the field from the surface normal (degrees)
   * - Z_grid
     - (nZ)
     - charge states
   * - Etilde_bin_edges
     - (nE + 1)
     - bin edges of :math:`E/(Z_s T_e)`
   * - theta_bin_edges_deg
     - (ntheta + 1)
     - bin edges of the impact angle from the normal (degrees)
   * - f
     - (ntau, npsi, nZ, nE, ntheta)
     - distribution, which sums to 1 over the last two axes

The distribution is interpolated bilinearly in :math:`T_i/T_e` and in
the field angle :math:`\theta` of Eq. :eq:`sputter-theta`, both
limited to the range of their axes.  The charge state must match an
entry of Z_grid exactly.  The yield is evaluated at the bin centres,
with the energy :math:`E_j = \tilde{E}_j Z_s T_e` at the local
:math:`T_e`.

With *target* and *projectiles*, the table of a slot is the file
iead_database_<element>.h5 of its projectile element if that file
exists.  Otherwise it is the shared file iead_database.h5, and a
warning is printed if a W projectile has to use it.  With the older
yield forms all slots use the shared file.  A slot falls back to the
mean-impact yield if its table is missing, if its charge state is not
in Z_grid, or if :math:`T_e` is not positive at the element.  A
missing shared file gives a warning, not an error.

*iead* with a file path uses that file as the shared table for all
projectiles, and no element tables are read.  *iead
none* evaluates the yield at the mean impact energy and angle.  At a
divertor target, where :math:`\theta` is close to 90 degrees, the
averaged yield is usually much larger than the mean-impact yield.

*iead inline* computes :math:`f` for the conditions of each element.
Test ions of the mass and charge of the slot are started at the
distance :math:`10\,L_\mathrm{MPS} + 20\,\lambda_\mathrm{D} +
2\rho\cos\theta` from the wall, or at :math:`d_\mathrm{max}` if that
is smaller, where :math:`L_\mathrm{MPS}` is the width of the magnetic
presheath, :math:`\rho` a gyroradius of the test ions and
:math:`d_\mathrm{max}` the extent of the sheath.  Their velocities
are Maxwellian at :math:`T_i` with a drift along the field towards
the wall.  They gyrate in the uniform local field and gain energy
from the sheath potential of the :doc:`global <global>` command,
which is evaluated for the :math:`T_e`, :math:`T_i`, :math:`n_e`,
:math:`B` and field angle of the element with the *mD_amu* and *dmax*
values of that command.  The potential depends on the distance from
the wall only, and there is no other electric field.  Ions that reach
the wall are binned in energy and in angle from the normal, and
:math:`f` is normalized to the number of impacts.  Ions that move
away from the wall or do not arrive within a time limit are left out.
A warning is printed if more than half of the ions of a distribution
do not arrive.

*iead_markers* and *iead_bins* set the number of test ions and bins.
The energy axis runs from 0 to an upper limit set from the sheath
drop, the drift and :math:`T_i`, and the angle axis from 0 to 90
degrees.  *iead_drift* selects the drift speed: *flow* is the
magnitude of the parallel flow of the slot, *cs* is
:math:`[(T_e + T_i)/m_\mathrm{D}]^{1/2}` and *cs2* is
:math:`[(T_e + T_i)/(2 m_\mathrm{D})]^{1/2}`, with
:math:`m_\mathrm{D}` the ion mass of the sheath model.  *flow* falls
back to *cs* where the flow is zero.  Before a distribution is
computed, :math:`T_e`, :math:`T_i`, :math:`n_e`, :math:`B` and the
drift are rounded to steps of 1 % and the field angle to 0.25
degrees.  Elements and slots with the same rounded conditions, mass
and charge share one distribution.  *iead_dump* writes every
distribution to a file: a header line starting with # followed by nE
rows of ntheta weights.  *iead inline* requires *target* and
*projectiles*.

Erosion flux and rate
^^^^^^^^^^^^^^^^^^^^^

With :math:`Y_s` the yield of slot :math:`s`, including
*yield_scale* and the average of Eq. :eq:`sputter-ybar` where it
applies, the outputs for an element of area :math:`A` are

.. math::
   :label: sputter-erosion

   \Gamma_{\mathrm{sput},s} = \Gamma_s\, Y_s ,
   \qquad
   \Gamma_\mathrm{ero} = \sum_s \Gamma_s\, Y_s ,
   \qquad
   \dot{N}_\mathrm{ero} = \Gamma_\mathrm{ero}\, A .

The sum runs over the projectile slots and the virtual impurity.  The
area is :math:`2\pi R L` for a line segment of length :math:`L` whose
midpoint is at major radius :math:`R`, and the triangle area in three
dimensions.

Static plasma
^^^^^^^^^^^^^

With *static yes* the outputs are computed once and reused until the
background is reloaded.  With *static no* they are recomputed every
time the compute is invoked.  In *compound* mode the yields at the
composition grid points are cached and interpolated at the current
composition on every invocation.

Output info
^^^^^^^^^^^

This compute calculates a per-surf vector if one output keyword is
given, and otherwise a per-surf array with one column per output
keyword in the order given.  A keyword ending in *_species* with the
value *all* expands to one column per ion slot.  For
*sputter_yield_species* and *sputter_flux_species* the expansion
covers the projectile range in effect where the keyword appears in
the command, so *projectile_slots* should precede them.

.. list-table::
   :header-rows: 1
   :widths: 34 46 20

   * - Keyword
     - Quantity
     - Unit
   * - *nflux_species*
     - :math:`\Gamma_s`, Eq. :eq:`sputter-flux`
     - m\ :sup:`-2` s\ :sup:`-1`
   * - *incident_angle_species*
     - :math:`\alpha_B`, Eq. :eq:`sputter-alpha`, from the surface
       plane
     - degrees
   * - *incident_energy_species*
     - :math:`E_0`, Eq. :eq:`sputter-energy`, averaged over the phases
       of *sheath_waveform*
     - eV
   * - *sputter_yield_species*
     - :math:`Y_s`
     -
   * - *sputter_flux_species*
     - :math:`\Gamma_{\mathrm{sput},s}`
     - m\ :sup:`-2` s\ :sup:`-1`
   * - *sputter_flux_total*, *erosion_flux*
     - :math:`\Gamma_\mathrm{ero}`
     - m\ :sup:`-2` s\ :sup:`-1`
   * - *sputter_rate_total*, *erosion_rate*
     - :math:`\dot{N}_\mathrm{ero}`
     - s\ :sup:`-1`

Values are 0 for elements that are not in the group or lie outside
the plasma domain.  The values can be written by
:doc:`dump surf <dump>`, time-averaged by
:doc:`fix ave/surf <fix_ave_surf>`, reduced by
:doc:`compute reduce <compute_reduce>` or used by
:doc:`fix surface/emit/source <fix_surface_emit_source>`.

Restrictions
------------

This command is part of the OPENEDGE package.

It requires explicit surface elements and a background with a
magnetic field.

*compound* and *conc* must be given together.  They require *target*
and *projectiles* and cannot be combined with *impurity*.

*sheath_waveform* requires *iead none*, which it selects by default,
and cannot be combined with *compound* or *impurity*.

The *tsurf* and *conc* custom attributes must exist when the compute
is first evaluated.  They may be created by a fix or surf_react that
is defined after this compute.  The *visibility* and
*sheath_waveform* attributes must exist when the run starts.

Related commands
----------------

:doc:`fix background <fix_background>`,
:doc:`fix surface/emit/source <fix_surface_emit_source>`,
:doc:`surf_react surface/pwi <surf_react_surface_pwi>`,
:doc:`compute surface/chemical/evaporation <compute_surface_chemical_evaporation>`,
:doc:`global <global>`

Default
-------

The option defaults are mass_amu = 2.0, incidence = poloidal, static =
no, iead = auto (none with sheath_waveform), iead_markers = 2000,
iead_bins = 120 30, iead_drift = flow, yield_scale = 1.0,
roughness_dm = 0.0, wall_flux_cutoff = 0.05 and projectile_slots = 2
to the last slot.  There is no default target_like, us, tsurf,
visibility, compound, impurity, boundary, iead_dump or
sheath_waveform.
