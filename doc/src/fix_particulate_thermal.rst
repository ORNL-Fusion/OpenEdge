.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix particulate/thermal

fix particulate/thermal command
===============================

Syntax
------

.. parsed-literal::

   fix ID particulate/thermal Nevery mix-ID background bg-ID keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* particulate/thermal = style name of this fix command
* Nevery = apply the thermal update every this many timesteps
* mix-ID = ID of the mixture that contains the particulate species
* background = required literal keyword
* bg-ID = ID of the :doc:`fix background <fix_background>` that
  supplies the plasma
* zero or more keyword/value pairs may be appended
* keyword = *model* or *material* or *heating* or *charge* or
  *heatflux/scale* or *ion_mass_amu* or *ion_charge_state* or *twall_K*
  or *alpha_e* or *rocket_eta* or *emit_into* or *termination* or
  *dis_mass_tol*

  * *model* value = *dustt2005* or *dis2021* = heating closure and
    termination rule
  * *material* value = NAME = :doc:`material <material>` that supplies
    the properties
  * *heating* value = *flux* or *oml* or *auto*

    * *flux* = heat flux density from the heat-flux data of the
      background, Eq. :eq:`pthermal-qflux`
    * *oml* = heating by the collected electrons and ions
    * *auto* = *oml* where the Debye length is defined, with an error
      if the particulate radius exceeds it

  * *charge* value = fix-ID = ID of the matching
    :doc:`fix particulate/charge <fix_particulate_charge>`
  * *heatflux/scale* value = multiplier on the plasma heat flux
    density (>= 0)
  * *ion_mass_amu* value = mass of the background ions (amu)
  * *ion_charge_state* value = charge state of the background ions
    (> 0)
  * *twall_K* value = wall temperature for the radiative exchange (K)
  * *alpha_e* value = evaporation coefficient, in (0,1]
  * *rocket_eta* value = efficiency of the evaporation recoil, in [0,1]
  * *emit_into* value = mix-ID = create vapour particles of this
    mixture
  * *termination* value = *dis* or *dustt* = override the termination
    rule of the model
  * *dis_mass_tol* value = relative mass at which the *dis* rule
    removes the particulate, in (0,1)

Examples
--------

.. code-block:: text

   fix heat particulate/thermal 1 particulates background pd &
            model dustt2005 material Li heating auto alpha_e 1.0 &
            twall_K 300 emit_into lithium_vapor
   fix heat particulate/thermal 1 particulates background pd &
            model dis2021 material Li heating oml charge qd

Description
-----------

Evolve the temperature, radius and mass of the particulates of
mixture mix-ID under plasma heating, thermal radiation and
evaporation or sublimation, with melting.  Optionally create the
evaporated atoms as simulation particles and apply an evaporation
recoil force.  The model is described in [Diaw2026]_.

The grain temperature :math:`T_d` and the wall temperature
:math:`T_\mathrm{wall}` are in K and :math:`k_\mathrm{B}` is written
out with them.  The plasma temperatures :math:`T_e` and :math:`T_i`
are in energy units.  Other symbols are defined on the
:ref:`notation <notation>` page.  The material properties
:math:`\rho_d`, :math:`c_p`, :math:`A_d`, :math:`H_\mathrm{vap}`,
:math:`a`, :math:`b`, :math:`\epsilon`, :math:`T_\mathrm{m}` and
:math:`H_\mathrm{m}` are those of the :doc:`material <material>` named
by the *material* keyword, and
:math:`m_\mathrm{a} = A_d\,\mathrm{amu}` is the mass of one atom of
the material.

Evaporation and energy balance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The saturated vapour pressure follows the law stored in the material,

.. math::
   :label: pthermal-psat

   \log_{10}\!\left(\frac{p_\mathrm{sat}}{1\ \mathrm{atm}}\right)
      = a + \frac{b}{T_d} ,

and the flux density of atoms leaving the surface is the
Hertz-Knudsen flux with the evaporation coefficient
:math:`\alpha_e` of the *alpha_e* keyword,

.. math::
   :label: pthermal-gamma

   \Gamma_\mathrm{ev} = \alpha_e\,
      \frac{p_\mathrm{sat}(T_d)}
           {\sqrt{2\pi\, m_\mathrm{a}\, k_\mathrm{B} T_d}} .

The code evaluates it as
:math:`\Gamma_\mathrm{ev} = \alpha_e\; 3.513\times10^{26}\;
p_\mathrm{sat}/\sqrt{A_d T_d}` m\ :sup:`-2` s\ :sup:`-1` with
:math:`p_\mathrm{sat}` in mmHg, :math:`A_d` in amu and :math:`T_d` in
K.  There is no recondensation term.

The radius and temperature of a sphere of uniform temperature obey

.. math::
   :label: pthermal-drdt

   \frac{\mathrm{d}r_d}{\mathrm{d}t}
      = -\frac{m_\mathrm{a}\,\Gamma_\mathrm{ev}}{\rho_d} ,

.. math::
   :label: pthermal-dtdt

   \frac{\mathrm{d}T_d}{\mathrm{d}t}
      = \frac{3}{\rho_d\, c_{p,\mathrm{eff}}\, r_d}
        \left[\, q_\mathrm{pl}
        - \epsilon\,\sigma_\mathrm{SB}
          \left(T_d^4 - T_\mathrm{wall}^4\right)
        - \Gamma_\mathrm{ev}\,\frac{H_\mathrm{vap}}{N_\mathrm{A}}
        \right] ,

where :math:`q_\mathrm{pl}` is the plasma heat flux density onto the
surface, :math:`\sigma_\mathrm{SB}` the Stefan-Boltzmann constant and
:math:`N_\mathrm{A}` the Avogadro constant.  The second term is the
grey-body exchange with a wall at the temperature of the *twall_K*
keyword.  The third is the latent heat carried by the evaporated
atoms.  The mass follows from the radius,

.. math::
   :label: pthermal-mass

   m_d = \frac{4}{3}\pi\,\rho_d\, r_d^3 ,

so the mass-loss rate is
:math:`\mathrm{d}m_d/\mathrm{d}t = -4\pi r_d^2\, m_\mathrm{a}
\Gamma_\mathrm{ev}`.

Melting is treated with an apparent heat capacity.  The specific heat
is *cp_solid* below :math:`T_\mathrm{m}`, if the material defines a
positive value, and *cp* otherwise.  Within 20 K of
:math:`T_\mathrm{m}` the latent heat of fusion is added,

.. math::
   :label: pthermal-cpeff

   c_{p,\mathrm{eff}} = c_p(T_d)
      + \frac{H_\mathrm{m}}{A_d\times10^{-3}\ \mathrm{kg/mol}}\;
        \frac{1}{2\,\Delta T_\mathrm{m}}
   \quad\text{for}\quad |T_d - T_\mathrm{m}| < \Delta T_\mathrm{m}
      = 20\ \mathrm{K} ,

so that the grain absorbs or releases :math:`H_\mathrm{m}` while it
crosses the band.  No melt fraction is stored.

Plasma heating
^^^^^^^^^^^^^^

The *heating* keyword selects :math:`q_\mathrm{pl}`.

With *flux* (the default) it is taken from the parallel and
perpendicular heat flux densities :math:`q_\parallel` and
:math:`q_\perp` of the background,

.. math::
   :label: pthermal-qflux

   q_\mathrm{pl} = \frac{1}{4}\sqrt{q_\parallel^2 + q_\perp^2} .

With *oml* it is the energy carried by the electrons and ions that the
grain collects in the orbital-motion-limited (OML) approximation.
The *model* keyword selects the expressions, which are given below.
It is zero where :math:`n_e` or :math:`T_e` is not positive.

With *auto* the *oml* value is used where :math:`n_e` and :math:`T_e`
are positive and the *flux* value elsewhere.  The run stops with an
error if :math:`r_d/\lambda_\mathrm{D} > 1`, because no collection
model for that range is available.

With model *dustt2005* a negative :math:`q_\mathrm{pl}` is set to
zero.  With model *dis2021* the signed value enters Eq.
:eq:`pthermal-dtdt`.  In both models :math:`q_\mathrm{pl}` is then
multiplied by the value of *heatflux/scale*.

OML heating with model dustt2005
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With :math:`\chi = -e\phi_d/T_e` the normalized surface potential,

.. math::
   :label: pthermal-qoml-dustt

   q_\mathrm{pl} = \Gamma_e\,(2.5\,T_e)
      + \Gamma_i\left(2.5\,T_i + \chi T_e + 13.6\ \mathrm{eV}\right) ,

.. math::

   \Gamma_e = \frac{n_e}{4}\sqrt{\frac{8T_e}{\pi m_e}}\;e^{-\chi} ,
   \qquad
   \Gamma_i = \frac{n_i}{4}\sqrt{\frac{8T_i}{\pi m_i}}
      \left(1 + \frac{\chi T_e}{T_i}\right) .

The ion energy includes the fall through the sheath potential and the
recombination energy of a hydrogenic ion.  The potential is
:math:`\phi_d = Z_d e/(4\pi\varepsilon_0 r_d)` with :math:`Z_d` from
the *particulate_charge* attribute of
:doc:`fix particulate/charge <fix_particulate_charge>`.  If
:math:`\phi_d < 0`, :math:`\chi` is taken from it and limited to at
most 20.  If no charge fix is defined or :math:`\phi_d \ge 0`,
:math:`\chi = 2.5` is used.  The ion flux is zero where :math:`T_i`
is not positive.  The ion charge state does not enter.

OML heating with model dis2021
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The heat flux density is assembled from the currents :math:`I_e`,
:math:`I_i`, :math:`I_\mathrm{th}` and :math:`I_\mathrm{see}` of the
charge fix named by the *charge* keyword, Eqs.
:eq:`pcharge-ie-dis` to :eq:`pcharge-isee-dis`, so that every emission
current has its cooling term.  With :math:`A = 4\pi r_d^2`,
:math:`v_{T_i} = (2T_i/m_i)^{1/2}`, :math:`X = Z_i\chi T_e/T_i`,
:math:`u` the speed of the grain relative to the ion flow in units of
:math:`v_{T_i}`, and currents divided by :math:`e` so that the
products are powers,

.. math::
   :label: pthermal-qoml-dis

   q_\mathrm{pl} = q_i + q_e + q_\mathrm{n} + q_\mathrm{sh}
      - q_\mathrm{th} - q_\mathrm{see} ,

.. math::

   q_i &= \tfrac{1}{4}\, n_i v_{T_i}\, T_i\; G_i(u,X) , \\
   q_e &= \frac{|I_e|}{eA}\, T_e\; G_e(\chi) , \qquad
   G_e = \begin{cases}
            2 + \chi, & \chi \ge 0, \\
            (2 - \chi)/(1 - \chi), & \chi < 0,
         \end{cases} \\
   q_\mathrm{n} &= \frac{I_i}{Z_i\, eA}\;(13.6\ \mathrm{eV}) , \\
   q_\mathrm{sh} &= \frac{I_i + I_e}{eA}\;\chi T_e , \\
   q_\mathrm{th} &= \frac{I_\mathrm{th}^0}{eA}
      \left[W_\mathrm{f}\, f_\mathrm{th}
            + 2k_\mathrm{B}T_d\;
              \tfrac{1}{2}\left(2 + 2\beta + \beta^2\right)e^{-\beta}
      \right] , \qquad
      \beta = \max\!\left(-\frac{\chi T_e}{k_\mathrm{B}T_d},\,0\right) , \\
   q_\mathrm{see} &= \frac{I_\mathrm{see}}{eA}\;
      3W_\mathrm{f}\,
      \frac{(1+y_\mathrm{s})(1+2y_\mathrm{s})}{1+3y_\mathrm{s}} , \qquad
      y_\mathrm{s} = \max\!\left(-\frac{\chi T_e}{W_\mathrm{f}},\,0\right) .

Here :math:`I_\mathrm{th}^0` is the thermionic current of Eq.
:eq:`pcharge-ith-dis` without the factor :math:`f_\mathrm{th}`, and
:math:`W_\mathrm{f}` is the work function.  The six terms are the
kinetic energy of the collected ions and electrons, the
neutralization energy of the ions, the energy exchanged with the
sheath potential, and the cooling by thermionic and secondary
electrons.  The ion energy moment is

.. math::

   G_i(u,X) =
   \begin{cases}
      \dfrac{1}{4}\left[\dfrac{2}{\sqrt{\pi}}
         \left(5 + 2u^2 + 2X\right)e^{-u^2}
         + \dfrac{P(u,X)}{u}\,\mathrm{erf}\,u\right], & X \ge 0, \\[12pt]
      \dfrac{1}{8}\left\{\dfrac{2}{\sqrt{\pi}}\left[
         \left(5 + 2u^2 - c\right)e^{-u_+^2}
         + \left(5 + 2u^2 + c\right)e^{-u_-^2}\right]
         + \dfrac{P(u,X)}{u}
           \left(\mathrm{erf}\,u_+ + \mathrm{erf}\,u_-\right)\right\},
         & X < 0,
   \end{cases}

with :math:`P = 3 + 12u^2 + 4u^4 + 2X(1 + 2u^2)`,
:math:`u_\pm = u \pm \sqrt{-X}` and
:math:`c = (3 + 2u^2)\sqrt{-X}/u`.  For :math:`u < 10^{-3}` the limits
:math:`2(2+X)/\sqrt{\pi}` and :math:`2e^{X}(2-X)/\sqrt{\pi}` are used.

Numerical method
^^^^^^^^^^^^^^^^

The fix acts on every step that is a multiple of :math:`N`, once at
the start of the step and once at the end, after the particle move.
Each application advances Eqs. :eq:`pthermal-drdt` and
:eq:`pthermal-dtdt` over :math:`\Delta t_N/2` by explicit Euler
substeps.  A substep is shortened so that :math:`T_d` changes by at
most 25 K, or by at most half its value if that is smaller while the
grain cools, and so that :math:`r_d` decreases by at most 2 %.  The
run stops with an error after 10000 substeps in one application.
The mass is then set from Eq. :eq:`pthermal-mass`.

With model *dustt2005*, :math:`q_\mathrm{pl}` is evaluated once per
application.  With model *dis2021* and OML heating, the current
balance of the charge fix is solved again in every substep at the
current :math:`r_d` and :math:`T_d`, because the thermionic current
changes quickly with temperature.

Termination
^^^^^^^^^^^

The *model* keyword also sets the rule that ends the life of a
particulate.  Let :math:`r_0` be the radius in the
:doc:`species <species>` definition.

With *dustt2005* (the default) the particulate is removed when
:math:`r_d = 0.1\,r_0`.  At that point 99.9 % of the atoms of the
initial sphere have evaporated.  The remaining 0.1 % is not counted
as vapour and is reported separately as the terminal remainder.

With *dis2021* the evaporation is followed down to the relative mass
:math:`m_d/m_0` given by *dis_mass_tol*, that is to
:math:`r_d = r_0\,(\mathit{dis\_mass\_tol})^{1/3}`.  The remainder is
then counted as evaporated, so the atom tally equals the initial
number of atoms.

The last substep is shortened so that the radius lands on the
terminal value.  The *termination* keyword overrides the rule of the
model with *dustt* or *dis*.  It is meant for verification runs.

Vapour source and recoil
^^^^^^^^^^^^^^^^^^^^^^^^

With *emit_into*, the atoms evaporated in one application are created
as simulation particles at the position of the particulate.  If
:math:`\Delta N` atoms left the particulate, the number of new
particles is drawn from a Poisson distribution of mean
:math:`w_d\,\Delta N / F_\mathrm{num}`.  Here :math:`w_d` is the
number of physical particulates represented by the simulation
particle, set by the *nweight* keyword of
:doc:`fix particulate/emit <fix_particulate_emit>` and 1 otherwise.
The species of each new particle is drawn from the fractions of the
*emit_into* mixture.  Each velocity component is drawn from a normal
distribution of standard deviation
:math:`(k_\mathrm{B}T_d/m)^{1/2}`, with :math:`m` the mass of the
vapour species and :math:`T_d` the mean of the grain temperatures
before and after the application.  The velocity of the particulate is
not added.

The *rocket_eta* keyword applies a recoil from asymmetric
evaporation.  In each application the velocity of a surviving
particulate changes by

.. math::
   :label: pthermal-rocket

   \Delta\mathbf{v} = -\eta\,
      \frac{\Delta N\, m_\mathrm{a}}{m_d}\,
      \sqrt{\frac{8k_\mathrm{B}T_d}{\pi m_\mathrm{a}}}\;
      \frac{\nabla T_e}{|\nabla T_e|} ,

with :math:`\eta` the value of *rocket_eta*, :math:`m_d` the mass
after the application and :math:`T_d` the mean temperature as above.
The recoil points down the electron temperature gradient of the
background.  A value of 0 turns it off.

Other keywords
^^^^^^^^^^^^^^

The *charge* keyword names the matching charge fix.  Model *dis2021*
requires it with *heating oml* or *auto*.

The *ion_mass_amu* and *ion_charge_state* keywords set :math:`m_i`
and :math:`Z_i` for the OML heating.

Restart, output info
--------------------

No information about this fix is written to
:doc:`binary restart files <restart>`.

This fix computes a global scalar and a global vector of length 2,
which can be accessed by any command that uses global values from a
fix, such as :doc:`stats_style <stats_style>`.  All are cumulative,
summed over processors, and multiplied by :math:`w_d`.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Output
     - Meaning
   * - scalar
     - number of physical atoms evaporated
   * - vector element 1
     - the same tally as the scalar
   * - vector element 2
     - terminal remainder of the *dustt* termination rule, in atoms

The fix writes these custom per-particle attributes:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Attribute
     - Meaning
   * - *droplet_heating_q*
     - :math:`q_\mathrm{pl}` of the last substep, including
       *heatflux/scale* (W/m\ :sup:`2`)
   * - *droplet_a_over_lambdaD*
     - :math:`r_d/\lambda_\mathrm{D}`
   * - *droplet_oml_weight*
     - 1 if OML heating was used, 0 otherwise
   * - *particulate_dis_q_ion*
     - :math:`q_i` (W/m\ :sup:`2`), model *dis2021* only
   * - *particulate_dis_q_electron*
     - :math:`q_e`, model *dis2021* only
   * - *particulate_dis_q_neutralization*
     - :math:`q_\mathrm{n}`, model *dis2021* only
   * - *particulate_dis_q_sheath*
     - :math:`q_\mathrm{sh}`, model *dis2021* only
   * - *particulate_dis_q_thermionic*
     - :math:`q_\mathrm{th}`, model *dis2021* only
   * - *particulate_dis_q_secondary*
     - :math:`q_\mathrm{see}`, model *dis2021* only

Restrictions
------------

This command is part of the OPENEDGE package.  It requires a
:doc:`fix background <fix_background>`.  *heating flux* and *auto*
require that background to carry the parallel and perpendicular heat
flux data.

The material must define positive *rho*, *cp*, *mass_amu* and
*hvap_J_mol* and a negative *antoine_b*.  Every species of the mixture
must have a positive radius in the species file.

Model *dis2021* requires a :doc:`material <material>` command with a
*provenance* string and explicit *rho*, *cp*, *cp_solid*, *mass_amu*,
*hvap_J_mol*, *antoine_a*, *antoine_b*, *emissivity*, *tmelt_K* and
*hmelt_J_mol*.  Every species of the mixture must have a positive
radius and a mass equal to :math:`\tfrac{4}{3}\pi\rho_d r_d^3` to a
relative accuracy of 10\ :sup:`-6`.  With *heating oml* or *auto* it
requires the *charge* keyword.  The named fix must be a
:doc:`fix particulate/charge <fix_particulate_charge>` with model
*dis2021*, the same background fix, the same material and the same
*ion_mass_amu* and *ion_charge_state*.  It must be defined before
this fix.

Related commands
----------------

:doc:`fix particulate/charge <fix_particulate_charge>`,
:doc:`fix particulate/drag <fix_particulate_drag>`,
:doc:`fix particulate/emit <fix_particulate_emit>`,
:doc:`material <material>`,
:doc:`fix background <fix_background>`

Default
-------

The option defaults are model = dustt2005, material = Li, heating =
flux, heatflux/scale = 1.0, ion_mass_amu = 2.0, ion_charge_state = 1,
twall_K = 300, alpha_e = 1.0, rocket_eta = 0, dis_mass_tol = 1e-12,
termination set by the model, and no emit_into.
