.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix surface/state/lm

fix surface/state/lm command
============================

Syntax
------

.. parsed-literal::

   fix ID surface/state/lm group-ID Nevery hf_source keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* surface/state/lm = style name of this fix command
* group-ID = ID of the surface group covered by the liquid-metal strip
* Nevery = solve the strip again every this many timesteps
* hf_source = *c_ID* or *c_ID[N]* or *f_ID* or *f_ID[N]* or Q or
  *plasma* compute-ID or *background* fix-ID or *target* file leg or
  *solps_b2pl* file

  * *c_ID* or *c_ID[N]* = per-surf vector, or column N of the per-surf
    array, of a compute that gives the wall heat flux (W/m\ :sup:`2`)
  * *f_ID* or *f_ID[N]* = the same for a fix
  * Q = uniform heat flux (W/m\ :sup:`2`)
  * *plasma* compute-ID = heat flux queried at the surface elements
    from a :doc:`compute plasma/fields <compute_plasma_fields>`
  * *background* fix-ID = heat flux
    :math:`(q_\parallel^2 + q_\perp^2)^{1/2}` from a
    :doc:`fix background <fix_background>`
  * *target* file leg = target profile from an HDF5 file.  leg is the
    name of the group in the file, for example *inner* or *outer*,
    with datasets *s* (m), *q_total* (W/m\ :sup:`2`) and *gamma_D*
    (m\ :sup:`-2` s\ :sup:`-1`).
  * *solps_b2pl* file = SOLPS b2plot target-loading file
    *ld_tg_\*.dat*

* zero or more keyword/value pairs may be appended.  The command needs
  at least 14 arguments in total.
* keyword = *h0* or *U0* or *Bs* or *Bw* or *alpha* or *width* or *Tin*
  or *sigma_w* or *tw* or *qss* or *Nx* or *Ny* or *ncase* or
  *max_iter* or *eps* or *relax* or *dt* or *evap* or *hf_scale* or
  *static* or *custom* or *csv*

  * *h0* value = film thickness at the inlet (m)
  * *U0* value = film velocity at the inlet (m/s)
  * *Bs* value = magnetic field component that enters the side-wall
    Hartmann number (T)
  * *Bw* value = magnetic field component that enters the wall
    Hartmann number (T)
  * *alpha* value = inclination angle of the strip (degrees)
  * *width* value = strip width (m)
  * *Tin* value = inlet temperature (degrees C)
  * *sigma_w* value = electrical conductivity of the wall (S/m); 0 =
    insulating wall
  * *tw* value = wall thickness (m)
  * *qss* value = heat-flux scale of the nondimensionalization
    (W/m\ :sup:`2`)
  * *Nx* value = number of stations along the strip
  * *Ny* value = number of points across the film
  * *ncase* value = 1 or 2 = form of the magnetic drag coefficient,
    Eq. :eq:`lm-drag`

    * 1 = drag set by the side walls, from *Bs*, *width*, *sigma_w*
      and *tw*
    * 2 = drag set by the field component *Bw* alone

  * *max_iter* value = maximum number of pseudo-time iterations
  * *eps* value = convergence criterion of the steady-state solve
  * *relax* value = relaxation parameter
  * *dt* value = dimensionless pseudo-time step
  * *evap* value = *yes* or *no* = include evaporation in the film
    mass and energy balance
  * *hf_scale* value = multiplier on the incoming heat flux
  * *static* value = *yes* or *no* = solve once at setup and keep the
    result
  * *custom* value = name of the per-surf custom attribute that
    receives the surface temperature
  * *csv* value = file to which the strip profile is written

Examples
--------

.. code-block:: text

   fix flm surface/state/lm divertor 100 background pd &
       h0 5e-3 U0 8.0 Bs 5.0 alpha 43.0 width 1.67 Tin 350 static yes
   fix flm_ol surface/state/lm div_ol 100 solps_b2pl ../solps/ld_tg_o.dat &
       h0 5e-3 U0 8.0 Bs 5.0 alpha 43.0 width 1.67 Tin 350 static yes &
       csv output/strip_ol.csv
   fix flm_il surface/state/lm div_il 100 solps_b2pl ../solps/ld_tg_i.dat &
       h0 5e-3 U0 8.0 Bs 5.0 alpha 43.0 width 1.67 Tin 350 static yes
   fix flm surface/state/lm divertor 50 c_cheat[1] h0 5e-3 U0 8.0 &
       Bs 5.0 alpha 43.0 width 1.67 Tin 350

Description
-----------

Model a flowing liquid-lithium film on the surface elements of a
group.  A one-dimensional strip runs along the wall.  The fix maps the
wall heat flux onto the strip, solves the shallow-water
magnetohydrodynamic flow and heat-transfer equations of Smolentsev for
the steady state, and stores the surface temperature and film
thickness on the surface elements.  The model and its use for lithium
sources are described in [Diaw2026]_.

Film model
^^^^^^^^^^

Let :math:`x` be the distance along the strip from the inlet and
:math:`y` the distance from the wall.  Lengths are scaled by the inlet
thickness :math:`h_0`, velocities by the inlet velocity :math:`U_0`,
time by :math:`h_0/U_0`, and the temperature rise above the inlet
temperature :math:`T_\mathrm{in}` by
:math:`q_\mathrm{ss} h_0/\kappa`:

.. math::

   \Theta = \frac{(T - T_\mathrm{in})\,\kappa}{q_\mathrm{ss}\, h_0} .

In these units the film velocity :math:`(U, V)`, thickness :math:`h`
and temperature obey

.. math::
   :label: lm-film

   \frac{\partial U}{\partial t} + U\frac{\partial U}{\partial x}
      + V\frac{\partial U}{\partial y}
   &= \frac{1}{\mathrm{Re}}\frac{\partial^2 U}{\partial y^2}
      - R_\mathrm{m}\,U
      + \frac{\sin\alpha}{\mathrm{Fr}}
      - \frac{\cos\alpha}{\mathrm{Fr}}\frac{\partial h}{\partial x} , \\
   \frac{\partial U}{\partial x} + \frac{\partial V}{\partial y} &= 0 , \\
   \frac{\partial h}{\partial t} + U_\mathrm{s}\frac{\partial h}{\partial x}
   &= V_\mathrm{s} - V_\mathrm{evap} , \\
   \frac{\partial \Theta}{\partial t} + U\frac{\partial \Theta}{\partial x}
      + V\frac{\partial \Theta}{\partial y}
   &= \frac{1}{\mathrm{Re}\,\mathrm{Pr}}
      \frac{\partial^2 \Theta}{\partial y^2} .

The subscript s denotes values at the free surface :math:`y = h`, and
:math:`\alpha` is the *alpha* angle.  The dimensionless groups are

.. math::

   \mathrm{Re} = \frac{U_0 h_0}{\nu}, \qquad
   \mathrm{Fr} = \frac{U_0^2}{g_0 h_0}, \qquad
   \mathrm{Pr} = \frac{\nu \rho c_p}{\kappa}, \qquad
   \mathrm{Ha_s} = B_\mathrm{s}\, b \sqrt{\frac{\sigma_\mathrm{e}}{\nu\rho}}, \qquad
   \mathrm{Ha_w} = B_\mathrm{w}\, h_0 \sqrt{\frac{\sigma_\mathrm{e}}{\nu\rho}} ,

with :math:`b` half the *width* and :math:`g_0 = 9.81` m/s\ :sup:`2`.
The magnetic drag coefficient depends on *ncase*:

.. math::
   :label: lm-drag

   R_\mathrm{m} =
   \begin{cases}
     \dfrac{\beta^2\,\mathrm{Ha_s}}{\mathrm{Re}}
       \left(1 + \dfrac{c_\mathrm{w}}{1 + c_\mathrm{w}}\,\mathrm{Ha_s}\right)
       & \textit{ncase} = 1 , \\[2ex]
     \dfrac{\mathrm{Ha_w^2}}{\mathrm{Re}} & \textit{ncase} = 2 ,
   \end{cases}
   \qquad
   \beta = \frac{h_0}{b}, \quad
   c_\mathrm{w} = \frac{\sigma_\mathrm{w} t_\mathrm{w}}{\sigma_\mathrm{e} b} ,

where :math:`\sigma_\mathrm{w}` and :math:`t_\mathrm{w}` are the
*sigma_w* and *tw* values.  An insulating wall has
:math:`c_\mathrm{w} = 0`.  The drag coefficient is the only term
that *ncase* changes.  With *ncase* 1, the form for a strip bounded by
side walls, *Bw* is not used.  With *ncase* 2, called the axisymmetric
case in the source, *Bs*, *sigma_w* and *tw* do not enter the
equations, and the default *Bw* = 0 gives no magnetic drag.  The value
is not checked, and any value other than 1 acts as 2.

The boundary conditions are no slip at the wall, zero shear at the
free surface, no heat flux into the wall, and at the free surface

.. math::

   \frac{\partial\Theta}{\partial y}\bigg|_{y=h} = \frac{q_\mathrm{s}}{q_\mathrm{ss}} ,

where :math:`q_\mathrm{s}` is the net heat flux into the film.  At the
inlet :math:`U = 1`, :math:`h = 1` and :math:`\Theta = 0`.

The lithium properties are fixed constants that do not vary with the
film temperature.  The source gives them as values near 350
degrees C.

.. list-table::
   :header-rows: 1
   :widths: 14 56 30

   * - Symbol
     - Property
     - Value
   * - :math:`\nu`
     - kinematic viscosity
     - 8.5e-7 m\ :sup:`2`/s
   * - :math:`\rho`
     - mass density
     - 485 kg/m\ :sup:`3`
   * - :math:`\sigma_\mathrm{e}`
     - electrical conductivity
     - 3.09e6 S/m
   * - :math:`c_p`
     - specific heat
     - 4200 J/(kg K)
   * - :math:`\kappa`
     - thermal conductivity
     - 49.6 W/(m K)
   * - :math:`H_\mathrm{vap}`
     - latent heat of vaporization
     - 145920 J/mol
   * - :math:`m_\mathrm{Li}`
     - mass of a lithium atom
     - 1.1526e-26 kg

Evaporation
^^^^^^^^^^^

With *evap yes* the film loses mass and heat by evaporation.  The
evaporation flux :math:`\Gamma_\mathrm{evap}(T_\mathrm{s})` at the
surface temperature :math:`T_\mathrm{s}` is the Hertz-Knudsen flux with
the Antoine vapour pressure, Eqs. :eq:`evap-pvap` and :eq:`evap-flux`
of :doc:`compute surface/chemical/evaporation <compute_surface_chemical_evaporation>`,
with an evaporation coefficient of 1.  It is zero below 25 degrees C.
The surface recedes at the dimensionless velocity

.. math::

   V_\mathrm{evap} = \frac{\Gamma_\mathrm{evap}\, m_\mathrm{Li}}{\rho\, U_0} ,

with the same atom mass :math:`m_\mathrm{Li}` as in the flux, and
the heat flux into the film is the wall heat flux
:math:`q_\mathrm{w}` less the evaporative cooling,

.. math::
   :label: lm-qnet

   q_\mathrm{s} = q_\mathrm{w} - \frac{H_\mathrm{vap}}{N_\mathrm{A}}\,
      \Gamma_\mathrm{evap}(T_\mathrm{s}) ,

where :math:`N_\mathrm{A}` is the Avogadro constant.
:math:`|q_\mathrm{s}|` is limited to :math:`5\,|q_\mathrm{w}|`, which
keeps an overshoot of the temperature in one iteration from
destabilizing the next.  With *evap no*,
:math:`V_\mathrm{evap} = 0` and :math:`q_\mathrm{s} = q_\mathrm{w}`.

Numerical solution
^^^^^^^^^^^^^^^^^^

Eq. :eq:`lm-film` is advanced in pseudo-time with step *dt* on a grid
of *Nx* stations along the strip and *Ny* points across the film, in
the coordinate :math:`y/h`.  Each iteration marches from the inlet to
the outlet.  At every station the momentum and energy equations are
implicit in :math:`y` and are solved with the Thomas algorithm, and
:math:`V` follows from continuity.  The new velocity and thickness
are then mixed with the old ones with the weight *relax*, and
:math:`q_\mathrm{s}` is updated from Eq. :eq:`lm-qnet`.

The iteration stops when the largest change of the dimensionless
thickness between two iterations is below *eps*, or after *max_iter*
iterations.  Every 25 iterations the temperature field is checked.
If it is not finite or :math:`|\Theta|` exceeds 50, the solve is
declared diverged.  The fix then prints a warning and leaves the
per-surf values of the previous solve in place.

The solver prints a warning when Re is below 10\ :sup:`3`, when
:math:`\mathrm{Ha_s}` is below 5 x 10\ :sup:`4`, when Pr is outside
10\ :sup:`-3` to 10\ :sup:`2`, or when the strip is longer than
300 :math:`h_0`.  These lie outside the range in which the solver has
been validated.

Mapping between wall and strip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each surface element of the group is assigned to one strip station by
its arc length along the wall.  For all sources except *solps_b2pl*,
the element centres are ordered by their second box coordinate and
then by their first, and the arc length is the cumulative distance
between consecutive centres.  The first element in this order is the
inlet.  With *solps_b2pl* each element centre is projected onto the
target polyline of the file, and the arc length is that of the file,
measured from its first row.

The wall heat flux at the stations is obtained as follows.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Source
     - Heat flux at a station
   * - *c_ID*, *f_ID*
     - mean of the per-surf values of the elements assigned to the
       station
   * - Q
     - the constant
   * - *plasma*
     - mean over the assigned elements of the heat flux that the
       plasma/fields compute returns at the element centre, times
       *hf_scale*
   * - *background*
     - mean over the assigned elements of
       :math:`(q_\parallel^2 + q_\perp^2)^{1/2}`, times *hf_scale*.
       The values are those of the plasma cell adjacent to the wall
       element if the plasma file maps wall elements to cells, and are
       otherwise interpolated at the element centre.
   * - *target*, *solps_b2pl*
     - the profile of the file, interpolated linearly in arc length at
       the station, times *hf_scale*

For the sources that are read on the surface elements (*c_ID*,
*f_ID*, *plasma*, *background*), a station to which no element is
assigned receives zero heat flux, so *Nx* should not exceed the number
of surface elements in the group by much.  Negative or
non-finite values from the *plasma* and *background* sources are set
to zero.  *hf_scale* is not applied to the *c_ID*, *f_ID* and constant
sources.

A *solps_b2pl* file is a text table.  Lines that start with "#" are
skipped.  The fix reads column 1, the arc length (m), columns 2 to 4,
:math:`n_e` (m\ :sup:`-3`), :math:`T_e` and :math:`T_i` (eV), column
5, the total heat flux Wtot (W/m\ :sup:`2`), and columns 15 and 16,
the :math:`R` and :math:`Z` coordinates of the target points (m).
The D\ :sup:`+` flux is computed as

.. math::

   \Gamma_\mathrm{D} = n_e \sqrt{\frac{T_e + T_i}{m_i}} ,

with :math:`m_i` the mass of a deuteron, 2.014 amu.  The file needs at
least four data rows of 16 columns.

Output
^^^^^^

After each solve the fix writes three per-surf custom attributes on
the surface elements of its group.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Content
   * - *Tsurf_lm*
     - surface temperature (degrees C).  The name can be changed with
       the *custom* keyword.
   * - *h_lm*
     - film thickness (m)
   * - *Gamma_D_lm*
     - incident D\ :sup:`+` flux (m\ :sup:`-2` s\ :sup:`-1`),
       interpolated from the file with the *target* and *solps_b2pl*
       sources and zero with the others

The attributes can be used as *s_name* by computes and dumps.  They
are read by
:doc:`compute surface/chemical/evaporation <compute_surface_chemical_evaporation>`
and :doc:`compute surface/chemical/adatom <compute_surface_chemical_adatom>`,
whose fluxes in turn feed
:doc:`fix surface/emit/source <fix_surface_emit_source>` to launch
lithium atoms.  When the fix is first initialized, the elements of its
group are set to *Tin*, *h0* and zero.  Elements outside the group are
not written.

Several fix surface/state/lm commands can share the same attributes,
since each writes only the elements of its own group.  The second and
third examples model an outer and an inner divertor leg in this way.

With *static no* the strip is solved at the end of every
:math:`N`-th timestep, which suits a heat flux that the run itself
updates.  With *static yes* it is solved once at the start of each
:doc:`run <run>` and the per-surf values are then kept, which suits a
prescribed heat flux.

The *csv* keyword writes one line per station with the columns arc
length *s_m*, arc length in the coordinate of the *solps_b2pl* file
*L_Lsep_m*, *R_m*, *Z_m*, *Tsurf_C*, *h_m*, *Wtot_Wm2*, *Gamma_D_m2s*
and *Gamma_evap_m2s*.  The file is written by processor 0 at the start
of a run.  It holds the solved profile only with *static yes*.  The
coordinates are filled only with the *solps_b2pl* source, and the
heat flux and D\ :sup:`+` flux only with the *target* and *solps_b2pl*
sources.

Restrictions
------------

This command is part of the OPENEDGE package.

It requires explicit surface elements and cannot be used with implicit
surfaces.  The *plasma* source requires a compute plasma/fields.  The
*background* source requires a :doc:`fix background <fix_background>`
whose plasma file contains the parallel and perpendicular heat fluxes.
The *target* source requires HDF5.

Each processor builds the mapping and solves the strip from the
surface elements it owns.  Sources that are read on the surface
elements (*c_ID*, *f_ID*, *plasma*, *background*) are not combined
across processors.

*Ny* must not exceed 1023.

The fix has no per-surf vector or array of its own.  Use the custom
attributes for output.

Related commands
----------------

:doc:`compute surface/chemical/evaporation <compute_surface_chemical_evaporation>`,
:doc:`compute surface/chemical/adatom <compute_surface_chemical_adatom>`,
:doc:`fix surface/emit/source <fix_surface_emit_source>`,
:doc:`fix background <fix_background>`, :doc:`custom <custom>`

Default
-------

The option defaults are h0 = 0.005, U0 = 8.0, Bs = 5.0, Bw = 0.0,
alpha = 43.0, width = 1.67, Tin = 350, sigma_w = 0, tw = 25e-6, qss =
1.0e6, Nx = 201, Ny = 51, ncase = 1, max_iter = 3000, eps = 5e-8,
relax = 1.0, dt = 0.5, evap = yes, hf_scale = 1.0, static = no, custom
= Tsurf_lm, no csv file.
