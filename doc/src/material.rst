.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: material

material command
================

Syntax
------

.. parsed-literal::

   material NAME keyword value ...

* NAME = name of the material (at most 15 characters).  *Li* and *B*
  are built in.
* one or more keyword/value pairs must be appended
* keyword = *provenance* or *rho* or *cp* or *cp_solid* or *mass_amu*
  or *hvap_J_mol* or *antoine_a* or *antoine_b* or *emissivity* or
  *work_function_eV* or *richardson_A* or *tmelt_K* or *hmelt_J_mol* or
  *tensile_Pa* or *see_delta_m* or *see_E_m_eV*

  * *provenance* value = ID = string that records where the data came
    from (at most 95 characters)
  * *rho* value = mass density (kg/m\ :sup:`3`)
  * *cp* value = specific heat of the liquid, or of a single-phase
    material (J/kg/K)
  * *cp_solid* value = specific heat of the solid (J/kg/K).  With 0,
    *cp* is used for both phases.
  * *mass_amu* value = atomic mass (amu)
  * *hvap_J_mol* value = enthalpy of vaporization or sublimation (J/mol)
  * *antoine_a* value = coefficient :math:`a` of the vapour-pressure
    law, Eq. :eq:`material-psat`
  * *antoine_b* value = coefficient :math:`b` of the vapour-pressure
    law (K, negative)
  * *emissivity* value = total emissivity for thermal radiation (0 to 1)
  * *work_function_eV* value = work function (eV)
  * *richardson_A* value = Richardson constant
    (A m\ :sup:`-2` K\ :sup:`-2`)
  * *tmelt_K* value = melting temperature (K)
  * *hmelt_J_mol* value = enthalpy of fusion (J/mol)
  * *tensile_Pa* value = tensile strength (Pa).  With 0 there is no
    electrostatic breakup.
  * *see_delta_m* value = maximum secondary-electron yield
  * *see_E_m_eV* value = electron energy at the maximum yield (eV)

Examples
--------

.. code-block:: text

   material B emissivity 0.75
   material MyStuff rho 1850 cp 1800 mass_amu 9.0 hvap_J_mol 5e5 &
            antoine_a 9.1 antoine_b -28000
   material Li provenance "Handbook-2019" rho 534 cp 4200 cp_solid 3580 &
            mass_amu 6.94 hvap_J_mol 1.47e5 antoine_a 5.055 &
            antoine_b -8023 emissivity 0.10 tmelt_K 453.7 hmelt_J_mol 3.0e3

Description
-----------

Define a particulate material, or change properties of an existing
one.  A material holds the bulk properties of the dust grains or
droplets that the particulate fixes need.  It is selected with the
*material* keyword of
:doc:`fix particulate/thermal <fix_particulate_thermal>`,
:doc:`fix particulate/drag <fix_particulate_drag>` and
:doc:`fix particulate/charge <fix_particulate_charge>`.  The
particulate model is described in [Diaw2026]_.

A material command with a new NAME creates the material with the
defaults of the table below.  A material command with an existing NAME,
including the built-in *Li* and *B*, changes only the listed
properties.  The command prints the resulting property set to the
screen.

Properties
^^^^^^^^^^

Particulate temperatures :math:`T_d` are in K on this page and
:math:`k_\mathrm{B}` is written out.  The symbols in the second column
are those used on the pages of the particulate fixes.

.. list-table::
   :header-rows: 1
   :widths: 20 44 16 20

   * - Property
     - Meaning
     - Unit
     - Default for a new material
   * - *rho*
     - mass density :math:`\rho_d`
     - kg/m\ :sup:`3`
     - unset, must be given
   * - *cp*
     - specific heat :math:`c_p` of the liquid, or of a single-phase
       material
     - J/kg/K
     - unset, must be given
   * - *cp_solid*
     - specific heat of the solid, used for
       :math:`T_d < T_\mathrm{m}`
     - J/kg/K
     - 0 (*cp* is used for both phases)
   * - *mass_amu*
     - atomic mass :math:`A_d`
     - amu
     - unset, must be given
   * - *hvap_J_mol*
     - enthalpy of vaporization or sublimation :math:`H_\mathrm{vap}`
     - J/mol
     - unset, must be given
   * - *antoine_a*
     - coefficient :math:`a` in Eq. :eq:`material-psat`
     -
     - 0
   * - *antoine_b*
     - coefficient :math:`b` in Eq. :eq:`material-psat`, negative
     - K
     - 0
   * - *emissivity*
     - total hemispherical emissivity :math:`\epsilon`
     -
     - 0
   * - *work_function_eV*
     - work function :math:`W_\mathrm{f}`
     - eV
     - unset
   * - *richardson_A*
     - Richardson constant :math:`A_\mathrm{R}`
     - A m\ :sup:`-2` K\ :sup:`-2`
     - unset
   * - *tmelt_K*
     - melting temperature :math:`T_\mathrm{m}`
     - K
     - unset
   * - *hmelt_J_mol*
     - enthalpy of fusion :math:`H_\mathrm{m}`
     - J/mol
     - 0
   * - *tensile_Pa*
     - tensile strength :math:`F_\mathrm{t}`
     - Pa
     - 0 (no breakup)
   * - *see_delta_m*
     - maximum secondary-electron yield :math:`\delta_\mathrm{m}`
     -
     - 0
   * - *see_E_m_eV*
     - electron energy :math:`E_\mathrm{m}` at the maximum yield
     - eV
     - 0

The saturated vapour pressure used for evaporation is

.. math::
   :label: material-psat

   \log_{10}\!\left(\frac{p_\mathrm{sat}}{1\ \mathrm{atm}}\right)
      = a + \frac{b}{T_d} ,

with :math:`T_d` in K.  The emissivity enters the grey-body exchange
with the wall temperature in
:doc:`fix particulate/thermal <fix_particulate_thermal>`.

Built-in materials
^^^^^^^^^^^^^^^^^^

Two materials can be used without a material command.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Property
     - *Li*
     - *B*
   * - *rho*
     - 534
     - 2340
   * - *cp*
     - 4200
     - 2000
   * - *cp_solid*
     - 3580
     - 0
   * - *mass_amu*
     - 6.94
     - 10.81
   * - *hvap_J_mol*
     - 1.47e5
     - 5.65e5
   * - *antoine_a*
     - 5.055
     - 8.64
   * - *antoine_b*
     - -8023
     - -32030
   * - *emissivity*
     - 0.10
     - 0.80
   * - *work_function_eV*
     - 2.9
     - 4.45
   * - *richardson_A*
     - 1.2e6
     - 1.2e6
   * - *tmelt_K*
     - 453.7
     - 2349
   * - *hmelt_J_mol*
     - 3.0e3
     - 5.02e4
   * - *tensile_Pa*
     - 0
     - 1.0e9
   * - *see_delta_m*
     - 0.5
     - 1.2
   * - *see_E_m_eV*
     - 85
     - 150

Use of the properties
^^^^^^^^^^^^^^^^^^^^^

* :doc:`fix particulate/thermal <fix_particulate_thermal>` uses *rho*,
  *cp*, *cp_solid*, *mass_amu*, *hvap_J_mol*, *antoine_a*, *antoine_b*,
  *emissivity*, *tmelt_K* and *hmelt_J_mol*.  It stops with an error
  unless *rho*, *cp*, *mass_amu* and *hvap_J_mol* are positive and
  *antoine_b* is negative.
* :doc:`fix particulate/drag <fix_particulate_drag>` uses *rho*.
* :doc:`fix particulate/charge <fix_particulate_charge>` uses
  *work_function_eV* and *richardson_A* for thermionic emission,
  *tensile_Pa* for electrostatic breakup, and *see_delta_m* and
  *see_E_m_eV* for secondary-electron emission.

The command records which properties were given explicitly.  Model
*dis2021* of the particulate fixes requires that every property it
uses was set by a material command and that a *provenance* string was
given.  The built-in values do not count as explicit.  Model
*dustt2005* accepts the built-in values.

Particulate models
^^^^^^^^^^^^^^^^^^

A particulate is a simulation particle with positive per-particle
*radius*, *mass* and *temp* attributes.  It represents one condensed
object: a solid grain, a molten droplet, or an object that changes
phase during the run.  The attributes are initialized from the
:doc:`species <species>` definition and evolved by the following
commands.

* :doc:`material <material>`: density, heat capacity, phase-change,
  vapour-pressure, emission and strength properties
* :doc:`fix particulate/emit <fix_particulate_emit>`: injection from
  selected surfaces
* :doc:`fix particulate/charge <fix_particulate_charge>`: local charge
  number :math:`Z_d`
* :doc:`fix particulate/drag <fix_particulate_drag>`: ion drag, neutral
  drag, electric force and gravity
* :doc:`fix particulate/thermal <fix_particulate_thermal>`:
  temperature, phase change, evaporation, mass and radius
* :doc:`surf_collide particulate/bounce <surf_collide_particulate_bounce>`:
  sticking or restitution at a wall

Define the particulate species with positive radius, mass and
temperature, put them in their own :doc:`mixture <mixture>`, and pass
that mixture to the particulate fixes.  Atomic ions can then be
advanced by the Boris or guiding-centre pusher while the particulates
bypass it:

.. code-block:: text

   global pusher mode hybrid plasma pd skip particulates

   fix emit particulate/emit particulates injector ...
   fix charge particulate/charge 1 background pd model dustt2005 &
              mixture particulates material Li
   fix drag particulate/drag 1 2.0 1 background pd &
            model dustt2005 mixture particulates material Li
   fix heat particulate/thermal 1 particulates background pd &
            model dustt2005 material Li heating auto

The charge, drag and thermal fixes each take a *model* keyword with
the values *dustt2005* (the default) and *dis2021*.  Select the same
model on all three.  With *dis2021* the drag and thermal fixes check
the model of the charge fix they refer to.  The three fixes act at the
start and at the end of a timestep, so keep their *Nevery* values
equal unless a convergence study supports another choice.

The two models are different published closure sets.  *dustt2005*
follows the DUSTT model [Pigarov2005]_.  *dis2021* follows the Dust
Injection Simulator of Nespoli et al. (Phys. Plasmas 28, 073704,
2021), with the ion collection drag of [Smirnov2007]_.  For a
negatively charged grain without electron emission in a plasma of
singly charged ions the two current balances are the same equation.
The floating potentials then differ by about 0.1 %, because
*dustt2005* converts *ion_mass_amu* with the proton mass and *dis2021*
with the atomic mass unit.  The charge state of the ions enters only
*dis2021*, so the potentials separate for
:math:`Z_i > 1`.  The drag and heating differ: *dustt2005* uses
the ion-friction expression of [Pigarov2005]_ and a fixed energy of
:math:`2.5\,T` per collected particle, and *dis2021* uses collection
drag and energy moments that depend on the flow velocity.  The
treatment of electron emission and the rule that ends the evaporation
of a grain also differ.  The case in
*examples/verification/particulates/dis2021/uniform_plasma_comparison*
applies one timestep of each model to a boron grain of radius 1
micrometre at rest in a uniform deuterium plasma, with thermionic and
secondary emission off.  Its script checks the charge, velocity and
heat flux of each run against the equations of that model and then
compares the two runs.  For that state it gives floating potentials
0.1 % apart, a velocity change 2.8 % larger with *dis2021* and a heat
flux 27 % larger with *dustt2005*.

Direction vectors given by the user, the *dir* vector of
:doc:`fix particulate/emit <fix_particulate_emit>` and the *gravity*
vector of :doc:`fix particulate/drag <fix_particulate_drag>`, are
:math:`(R, Z, \phi)` components.  They are mapped onto the coordinate
order of a two-dimensional planar or axisymmetric run.  In a
three-dimensional run the three numbers are taken as the
:math:`(x, y, z)` components.

The charging, drag and heating closures have different ranges of
validity, and a run that completes does not show that a closure holds
for the particulate size in question.  The orbital-motion-limited
(OML) charging and heating expressions assume
:math:`r_d \ll \lambda_\mathrm{D}`.  Use *validity error* on
:doc:`fix particulate/charge <fix_particulate_charge>` for runs that
must stay inside that limit.

Restrictions
------------

This command is part of the OPENEDGE package.

NAME is limited to 15 characters and the *provenance* string to 95
characters.  All property values must be numeric.

Related commands
----------------

:doc:`fix particulate/thermal <fix_particulate_thermal>`,
:doc:`fix particulate/drag <fix_particulate_drag>`,
:doc:`fix particulate/charge <fix_particulate_charge>`,
:doc:`fix particulate/emit <fix_particulate_emit>`,
:doc:`surf_collide particulate/bounce <surf_collide_particulate_bounce>`,
:doc:`species <species>`

Default
-------

The defaults of a new material are listed in the first table above:
*cp_solid*, *hmelt_J_mol*, *tensile_Pa*, *see_delta_m*, *see_E_m_eV*,
*antoine_a*, *antoine_b* and *emissivity* are 0, and the other
properties are unset and must be given before they are used.  A new
material has no *provenance* string.  The built-in *Li* and *B* values
are listed in the second table.
