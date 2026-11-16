.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix particulate/charge

fix particulate/charge command
==============================

Syntax
------

.. parsed-literal::

   fix ID particulate/charge Nevery background bg-ID keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* particulate/charge = style name of this fix command
* Nevery = update the charge every this many timesteps
* background = required literal keyword
* bg-ID = ID of the :doc:`fix background <fix_background>` that
  supplies the plasma
* zero or more keyword/value pairs may be appended
* keyword = *model* or *mixture* or *material* or *ion_mass_amu* or
  *ion_charge_state* or *fixed_charge* or *thermionic* or
  *richardson_A* or *work_function_eV* or *see* or *see_model* or
  *see_angular* or *validity* or *radius* or *mass* or *temp*

  * *model* value = *dustt2005* or *dis2021* = charging closure
  * *mixture* value = mix-ID = act only on species of this mixture
  * *material* value = NAME = :doc:`material <material>` that supplies
    the thermionic, secondary-emission and tensile properties
  * *ion_mass_amu* value = mass of the background ions (amu)
  * *ion_charge_state* value = charge state of the background ions
    (> 0)
  * *fixed_charge* value = Zd = assign this charge number and skip the
    current balance
  * *thermionic* value = *yes* or *no* = include thermionic emission
  * *richardson_A* value = Richardson constant
    (A m\ :sup:`-2` K\ :sup:`-2`)
  * *work_function_eV* value = work function (eV)
  * *see* value = *yes* or *no* = secondary-electron emission of model
    *dustt2005*
  * *see_model* value = *none* or *dustt2005* or *kollath_smirnov2007*
    or *young_dekker* = secondary-electron closure
  * *see_angular* value = *yes* or *no* = average the secondary yield
    over the angle of incidence
  * *validity* value = *warn* or *error* or *no* = response when the
    particulate radius exceeds the Debye length
  * *radius* value = radius given to a particle that has none (m)
  * *mass* value = mass given to a particle that has none (kg, > 0)
  * *temp* value = temperature given to a particle that has none (K)

Examples
--------

.. code-block:: text

   fix pd background file input/plasma.h5 static yes
   fix qd particulate/charge 1 background pd model dustt2005 &
          mixture particulates material Li validity warn
   fix qd particulate/charge 1 background pd model dis2021 &
          mixture particulates material Li thermionic yes &
          see_model kollath_smirnov2007 see_angular yes

Description
-----------

Compute the charge number :math:`Z_d` of each particulate from the
local background plasma.  The surface potential :math:`\phi_d` is the
floating potential at which the currents to the grain sum to zero, and

.. math::
   :label: pcharge-zd

   Z_d = \frac{4\pi\varepsilon_0\, r_d\, \phi_d}{e} ,

the charge of an isolated sphere of radius :math:`r_d`.  The result is
stored in the custom per-particle attribute *particulate_charge* and
is read by :doc:`fix particulate/drag <fix_particulate_drag>` and
:doc:`fix particulate/thermal <fix_particulate_thermal>`.  The model
is described in [Diaw2026]_.

The grain temperature :math:`T_d` is in K and
:math:`k_\mathrm{B}` is written out with it.  The plasma temperatures
:math:`T_e` and :math:`T_i` are in energy units.  Other symbols are
defined on the :ref:`notation <notation>` page.  This page also uses

.. math::

   \chi = -\frac{e\phi_d}{T_e} , \qquad
   v_{T_i} = \sqrt{\frac{2T_i}{m_i}} , \qquad
   u = \frac{|u_\parallel\hat{\mathbf{b}} - \mathbf{v}|}{v_{T_i}} , \qquad
   \bar{v}_e = \sqrt{\frac{8T_e}{\pi m_e}} ,

so that :math:`\chi > 0` for a negatively charged grain and :math:`u`
is the speed of the grain relative to the ion flow in units of the ion
thermal speed.  The ion mass :math:`m_i` and charge state :math:`Z_i`
are set by *ion_mass_amu* and *ion_charge_state*.  *dustt2005*
converts *ion_mass_amu* to kg with the proton mass and *dis2021* with
the atomic mass unit, a difference of 0.7 % in :math:`m_i`.

The *model* keyword selects the expressions for the currents,
*dustt2005* (the default) or *dis2021*.  Use the same model on the
charge, drag and thermal fixes.

Model dustt2005
^^^^^^^^^^^^^^^

The currents are those of the DUSTT model [Pigarov2005]_.  The
electron current is the orbital-motion-limited (OML) current to a
repelling sphere, reduced by secondary emission,

.. math::
   :label: pcharge-ie-dustt

   I_e = -e\,\pi r_d^2\, n_e\, \bar{v}_e\,
         \left(1 - \bar{\delta}\right) e^{-\chi} .

The ion current depends on the flow,

.. math::
   :label: pcharge-ii-dustt

   I_i = e\,\pi r_d^2\, n_i\, v_{T_i}\, \max\!\left[F(u,a),\,0\right],
   \qquad
   F(u,a) = \left(u + \frac{1}{2u} + \frac{a}{u}\right)\mathrm{erf}\,u
            + \frac{e^{-u^2}}{\sqrt{\pi}} ,
   \qquad
   a = \chi\,\frac{T_e}{T_i} .

For :math:`u < 10^{-3}` the limit
:math:`F = 2(1+a)/\sqrt{\pi}` is used, which is the OML current of a
plasma at rest.  The ion charge state does not enter this model.

With *thermionic yes* the Richardson-Dushman current is added,

.. math::
   :label: pcharge-ith-dustt

   I_\mathrm{th} = 4\pi r_d^2\, A_\mathrm{R}\, T_d^2\,
      \exp\!\left(-\frac{W_\mathrm{eff}}{k_\mathrm{B}T_d}\right),
   \qquad
   W_\mathrm{eff} = \max\!\left(0,\;
      W_\mathrm{f} - \frac{e^2}{16\pi\varepsilon_0 r_d}\right),

where :math:`A_\mathrm{R}` is the Richardson constant and
:math:`W_\mathrm{f}` the work function.  This current does not depend
on :math:`\phi_d`.  It is zero for a particle without a positive
temperature.

The secondary-electron term is switched on by *see yes* or,
equivalently, *see_model dustt2005*.  The yield for an electron of
energy :math:`E` is the Sternglass form

.. math::
   :label: pcharge-sternglass

   \delta(E) = 7.4\,\delta_\mathrm{m}\,\frac{E}{E_\mathrm{m}}\,
               \exp\!\left(-2\sqrt{E/E_\mathrm{m}}\right),

with the maximum yield :math:`\delta_\mathrm{m}` and its energy
:math:`E_\mathrm{m}` taken from the material.  It is averaged over the
incident Maxwellian flux,

.. math::

   \bar{\delta} = \min\!\left[1,\;
      \frac{1}{T_e^2}\int_0^{30T_e} \delta(E)\, E\, e^{-E/T_e}\,
      \mathrm{d}E \right],

by a 256-point midpoint rule.  Without a material that has positive
:math:`\delta_\mathrm{m}` and :math:`E_\mathrm{m}`,
:math:`\bar{\delta} = 0`.  This term is experimental.  It covers
electron impact only, Eq. :eq:`pcharge-ie-dustt` has no branch for a
positive grain, and the emitted electrons carry no energy in the heat
balance of the thermal fix.  A warning is printed when it is selected,
and another the first time :math:`\bar{\delta}` exceeds 0.95.

The floating potential is the root of

.. math::
   :label: pcharge-balance-dustt

   I_e(\phi_d) + I_i(\phi_d) + I_\mathrm{th} = 0 .

Model dis2021
^^^^^^^^^^^^^

The currents follow the Dust Injection Simulator model of Nespoli et
al. (Phys. Plasmas 28, 073704, 2021) and hold for either sign of the
grain potential.  With :math:`X = Z_i\,\chi\,T_e/T_i`,

.. math::
   :label: pcharge-ie-dis

   I_e = -e\,\pi r_d^2\, n_e\, \bar{v}_e\, F_e(\chi), \qquad
   F_e(\chi) =
   \begin{cases}
      e^{-\chi}, & \chi \ge 0, \\
      1 - \chi, & \chi < 0,
   \end{cases}

.. math::
   :label: pcharge-ii-dis

   I_i = Z_i e\,\pi r_d^2\, n_i\, v_{T_i}\, F_i(u,X) ,

.. math::

   F_i(u,X) =
   \begin{cases}
      \dfrac{1}{2u}\left[\left(1 + 2u^2 + 2X\right)\mathrm{erf}\,u
         + \dfrac{2u}{\sqrt{\pi}}\,e^{-u^2}\right], & X \ge 0, \\[10pt]
      \dfrac{1}{4u}\left[\left(1 + 2u^2 + 2X\right)
         \left(\mathrm{erf}\,u_+ + \mathrm{erf}\,u_-\right)
         + \dfrac{2}{\sqrt{\pi}}\left(u_+ e^{-u_-^2} + u_- e^{-u_+^2}\right)
         \right], & X < 0,
   \end{cases}

with :math:`u_\pm = u \pm \sqrt{-X}`.  For :math:`u < 10^{-3}` the
limits :math:`2(1+X)/\sqrt{\pi}` and :math:`2e^{X}/\sqrt{\pi}` are
used.  The first branch is the function :math:`F` of Eq.
:eq:`pcharge-ii-dustt`.

With *thermionic yes*,

.. math::
   :label: pcharge-ith-dis

   I_\mathrm{th} = 4\pi r_d^2\, A_\mathrm{R}\, T_d^2\,
      \exp\!\left(-\frac{W_\mathrm{f}}{k_\mathrm{B}T_d}\right)
      f_\mathrm{th}(\chi), \qquad
   f_\mathrm{th} =
   \begin{cases}
      1, & \chi \ge 0, \\
      (1 - y)\,e^{y}, \quad y = \dfrac{\chi T_e}{k_\mathrm{B}T_d},
         & \chi < 0 .
   \end{cases}

The factor :math:`f_\mathrm{th}` is the fraction of the emitted
electrons that escape a positive grain.

With *see_model kollath_smirnov2007* the secondary-electron current is

.. math::
   :label: pcharge-isee-dis

   I_\mathrm{see} = |I_e|\;\bar{\delta}\; f_\mathrm{esc}, \qquad
   f_\mathrm{esc} = \frac{1 + 3y_\mathrm{s}}{(1+y_\mathrm{s})^3}, \qquad
   y_\mathrm{s} = \max\!\left(-\frac{\chi T_e}{W_\mathrm{f}},\,0\right).

The yield :math:`\delta(E)` is the Kollath law used in
[Smirnov2007]_, which is Eq. :eq:`pcharge-sternglass` with the
coefficient written as :math:`2.72^2`.  Its average over the Maxwellian
flux of the electrons that reach the surface is evaluated in closed
form,

.. math::

   \bar{\delta} = 2.72^2\,\delta_\mathrm{m}\,
      \frac{\alpha_\mathrm{m}}{1+\kappa}\;
      2\,e^{\kappa+\alpha_\mathrm{m}}
      \int_{\sqrt{\kappa}+\sqrt{\alpha_\mathrm{m}}}^{\infty}
      \left(t - \sqrt{\alpha_\mathrm{m}}\right)^5 e^{-t^2}\,\mathrm{d}t ,
   \qquad
   \alpha_\mathrm{m} = \frac{T_e}{E_\mathrm{m}}, \quad
   \kappa = \max(-\chi,\,0) .

With *see_angular yes*, :math:`\bar{\delta}` is multiplied by 1.25,
the average of :math:`(\cos\alpha)^{-0.4}` over a cosine distribution
of the angle of incidence :math:`\alpha`.  A warning is printed when
*see_model kollath_smirnov2007* is used and *see_angular* is not
given.  The yield is not clipped.

The floating potential is the root of

.. math::
   :label: pcharge-balance-dis

   I_i + I_e + I_\mathrm{th} + I_\mathrm{see} = 0 .

*see_model young_dekker* is recognized but not available, and
selecting it stops the run with an error.

Solution and update
^^^^^^^^^^^^^^^^^^^

Eq. :eq:`pcharge-balance-dustt` or :eq:`pcharge-balance-dis` is solved
for each particulate by bisection.  The bracket is
:math:`-80 \le \phi_d/T_\mathrm{max} \le 20` with :math:`\phi_d` in V
and :math:`T_\mathrm{max} = \max(T_e, T_i)` in eV, and is doubled up
to 12 times if it holds no sign change.  The relative tolerance is
10\ :sup:`-10` for *dustt2005* and 10\ :sup:`-12` for *dis2021*.

The charge is not advanced in time.  On every step that is a multiple
of :math:`N` the fix sets :math:`Z_d` to the equilibrium value of Eq.
:eq:`pcharge-zd`, once at the start of the step and once at the end,
after the particle move.  Between these steps a particle keeps its
last value.  This assumes that the charging time is short compared
with :math:`\Delta t_N`.

The plasma is sampled at the particle position.  A particle is left
unchanged where :math:`T_e`, :math:`T_i`, :math:`n_e` or :math:`n_i`
is not positive.  If the root search fails, model *dustt2005* leaves
the particle unchanged and model *dis2021* stops with an error.

The *fixed_charge* keyword writes the given charge number to every
processed particle and skips the current balance.  It is meant for
tests of the forces.

Electrostatic breakup
^^^^^^^^^^^^^^^^^^^^^

With model *dustt2005* and a material of positive tensile strength
:math:`F_\mathrm{t}`, a particulate breaks up when
:math:`|\phi_d|` exceeds

.. math::
   :label: pcharge-breakup

   \phi^* = 0.1\ \mathrm{V}\;
      \sqrt{\frac{10\,F_\mathrm{t}}{1\ \mathrm{Pa}}}\;
      \frac{r_d}{1\ \mu\mathrm{m}} .

The factor 10 converts :math:`F_\mathrm{t}` to dyn/cm\ :sup:`2`.  The
particulate is replaced by two fragments at the same position.  Each
has half the mass, the radius :math:`2^{-1/3} r_d`, the temperature
of the parent and half its charge number.  The fragments separate
with velocities of :math:`\pm 0.1` m/s relative to the parent, in a
random direction in the plane of the first two velocity components.
Model *dis2021* has no breakup.

Other keywords
^^^^^^^^^^^^^^

The *mixture* keyword restricts the fix to the species of the named
mixture.  Without it every particle with a positive radius is
processed.

The *material* keyword names a :doc:`material <material>`.  Its
*work_function_eV* and *richardson_A* are used for thermionic emission
unless the keywords of the same name are given on the fix.  Its
*see_delta_m* and *see_E_m_eV* are used for secondary emission and its
*tensile_Pa* for breakup.

The *validity* keyword sets what happens when
:math:`r_d > \lambda_\mathrm{D}`, where the OML currents are outside
their range.  With *warn* a warning is printed the first time this
occurs, with *error* the run stops, and with *no* nothing is reported.
The charge is computed from the same expressions in every case.

The *radius*, *mass* and *temp* keywords fill in the radius, mass or
temperature of a particle that has none.  Values already present are
not replaced.

Restrictions
------------

This command is part of the OPENEDGE package.  It requires a
:doc:`fix background <fix_background>`.

Model *dis2021* requires a *material* defined by a
:doc:`material <material>` command with a *provenance* string.  With
*thermionic yes* the material command must give *work_function_eV*
and *richardson_A* explicitly, and with *see_model
kollath_smirnov2007* it must give *work_function_eV*, *see_delta_m*
and *see_E_m_eV*.  Every species the fix acts on must have a positive
temperature in the species file unless the *temp* keyword is given.
Model *dis2021* rejects *see yes* and *see_model dustt2005*.

Model *dustt2005* accepts only *see_model none* or *dustt2005*.

*see_model kollath_smirnov2007* requires a material with positive
*see_delta_m* and *see_E_m_eV*.  *see_model young_dekker* is not
available.

Related commands
----------------

:doc:`material <material>`,
:doc:`fix particulate/drag <fix_particulate_drag>`,
:doc:`fix particulate/thermal <fix_particulate_thermal>`,
:doc:`fix particulate/emit <fix_particulate_emit>`,
:doc:`fix background <fix_background>`

Default
-------

The option defaults are model = dustt2005, ion_mass_amu = 2.0,
ion_charge_state = 1, thermionic = no, richardson_A = 1.2e6,
work_function_eV = 2.9, see = no, see_model = none, see_angular = no,
validity = warn, no mixture filter, no material and no fixed_charge.
