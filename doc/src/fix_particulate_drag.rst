.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix particulate/drag

fix particulate/drag command
============================

Syntax
------

.. parsed-literal::

   fix ID particulate/drag Nevery A_bg Z_bg background bg-ID keyword value ...

* ID is documented in the :doc:`fix <fix>` command
* particulate/drag = style name of this fix command
* Nevery = apply the forces every this many timesteps
* A_bg = mass of the background ions (amu)
* Z_bg = charge state of the background ions (integer)
* background = required literal keyword
* bg-ID = ID of the :doc:`fix background <fix_background>` that
  supplies the plasma
* zero or more keyword/value pairs may be appended
* keyword = *model* or *mixture* or *material* or *coulomb/self* or
  *charge* or *coulomb/chi* or *coulomb/delta* or *coulomb/lnlambda* or
  *neutrals* or *efield* or *gravity* or *radius* or *mass* or *temp*

  * *model* value = *dustt2005* or *dis2021* = ion-drag closure
  * *mixture* value = mix-ID = act only on species of this mixture
  * *material* value = NAME = :doc:`material <material>` that supplies
    the particulate density
  * *coulomb/self* value = *yes* or *no* = take the normalized
    potential, :math:`T_i/T_e` and the Coulomb logarithm from the local
    plasma and the particulate charge
  * *charge* value = fix-ID = ID of the matching
    :doc:`fix particulate/charge <fix_particulate_charge>`
  * *coulomb/chi* value = normalized surface potential :math:`\chi`
    used with *coulomb/self no*
  * *coulomb/delta* value = :math:`T_i/T_e` used with *coulomb/self no*
    (> 0)
  * *coulomb/lnlambda* value = Coulomb logarithm used with
    *coulomb/self no* (>= 0)
  * *neutrals* value = *yes* or *no* = neutral drag
  * *efield* value = *yes* or *no* = electric force
  * *gravity* values = gR gZ gphi = gravitational acceleration as
    :math:`(R, Z, \phi)` components (m/s\ :sup:`2`)
  * *radius* value = radius given to a particle that has none (m)
  * *mass* value = mass given to a particle that has none (kg)
  * *temp* value = temperature given to a particle that has none (K)

Examples
--------

.. code-block:: text

   fix fd particulate/drag 1 2.0 1 background pd &
          model dustt2005 mixture particulates material Li &
          coulomb/self yes neutrals yes efield yes &
          gravity 0.0 -9.80665 0.0
   fix fd particulate/drag 1 2.0 1 background pd model dis2021 &
          mixture particulates material Li charge qd

Description
-----------

Apply the ion drag, the neutral drag, the electric force and gravity
to particulates.  The velocity :math:`\mathbf{v}` of a particulate of
mass :math:`m_d` obeys

.. math::
   :label: pdrag-eom

   \frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t}
      = -\nu_i\left(\mathbf{v} - \mathbf{u}_i\right)
        - \nu_n\,\mathbf{v} + \mathbf{a} ,
   \qquad
   \mathbf{a} = \mathbf{g} + \frac{Z_d e}{m_d}\,\mathbf{E} ,

where :math:`\mathbf{u}_i = u_\parallel\hat{\mathbf{b}}` is the flow
velocity of the background ions, :math:`\nu_i` and :math:`\nu_n` are
the ion and neutral drag rates, and the neutrals are at rest.  The
model is described in [Diaw2026]_.

The plasma temperatures :math:`T_e`, :math:`T_i` and the neutral
temperature :math:`T_n` are in energy units.  Symbols are defined on
the :ref:`notation <notation>` page.  The background ions have mass
:math:`m_i = A_\mathrm{bg}\,\mathrm{amu}` and charge state
:math:`Z_i = Z_\mathrm{bg}`, and :math:`\rho_d` is the mass density of
the particulate material.  This page also uses

.. math::

   v_{T_i} = \sqrt{\frac{2T_i}{m_i}} , \qquad
   u = \frac{|\mathbf{u}_i - \mathbf{v}|}{v_{T_i}} , \qquad
   \chi = -\frac{e\phi_d}{T_e} , \qquad
   \phi_d = \frac{Z_d e}{4\pi\varepsilon_0 r_d} , \qquad
   \tau = \frac{T_i}{T_e} .

Ion drag
^^^^^^^^

The ion drag force is
:math:`\mathbf{F}_i = m_d\,\nu_i\,(\mathbf{u}_i - \mathbf{v})` with

.. math::
   :label: pdrag-nui

   \nu_i = \nu_0\,\zeta , \qquad
   \nu_0 = \frac{3}{4}\,\frac{n_i m_i v_{T_i}}{\rho_d\, r_d} .

For a sphere of density :math:`\rho_d`,
:math:`\nu_0 = \pi r_d^2\, n_i m_i v_{T_i}/m_d`.  The rate is computed
from :math:`\rho_d` and :math:`r_d`, not from the per-particle mass.
The dimensionless factor :math:`\zeta` is the sum of a part from the
ions collected by the grain and a part from the ions scattered in its
Coulomb field.  The *model* keyword selects its form.  The scattering
part is common to both models,

.. math::
   :label: pdrag-scat

   \zeta_\mathrm{sc}(u,X) = X^2\,\ln\Lambda\;
      \frac{\mathrm{erf}\,u - 2u\,e^{-u^2}/\sqrt{\pi}}{u^3} ,

which tends to :math:`4X^2\ln\Lambda/(3\sqrt{\pi})` as
:math:`u \to 0`.  That limit is used for :math:`u < 10^{-3}`.  The
argument :math:`X` is the grain potential in units of the ion
temperature and is defined differently in the two models.

With model *dustt2005* [Pigarov2005]_, :math:`X = \chi/\tau` and
:math:`\zeta = \zeta_\mathrm{coll} + \zeta_\mathrm{sc}` with

.. math::
   :label: pdrag-coll-dustt

   \zeta_\mathrm{coll} = \frac{1}{2\sqrt{\pi}\,u^3}
      \left\{ u\left(2u^2 + 1 + 2X\right)e^{-u^2}
      + \frac{\sqrt{\pi}}{2}\left[4u^4 + 2u^2 - 1
        - 2\left(1 - 2u^2\right)X\right]\mathrm{erf}\,u \right\} .

For :math:`u < 10^{-3}` the limit
:math:`\zeta_\mathrm{coll} = (5 + 4X)/(3\sqrt{\pi})` is used.  The ion
charge state does not enter this model.  The model treats negatively
charged grains only: :math:`\chi` is limited to at most 20 and is set
to 0 for a grain with :math:`\phi_d \ge 0`.

With model *dis2021*, :math:`\chi` keeps its sign,
:math:`X = Z_i\,\chi/\tau`, and
:math:`\zeta = \zeta_\mathrm{coll} + \zeta_\mathrm{sc}`.  The
collection part is that of [Smirnov2007]_.  For :math:`X \ge 0`,

.. math::
   :label: pdrag-coll-dis

   \zeta_\mathrm{coll} = \frac{1}{2u^2}\left[
      \left(1 + 2u^2 + 2X\right)\frac{e^{-u^2}}{\sqrt{\pi}}
      + u\,Q(u,X)\,\mathrm{erf}\,u \right], \qquad
   Q = 1 + 2u^2 + 2X - \frac{1 - 2u^2 + 2X}{2u^2} ,

with the limit :math:`(8 + 4X)/(3\sqrt{\pi})` used for
:math:`u < 10^{-3}`.  For :math:`X < 0`, with
:math:`u_\pm = u \pm \sqrt{-X}` and
:math:`c = (1 - 2u^2)\sqrt{-X}/u`,

.. math::

   \zeta_\mathrm{coll} = \frac{1}{4u^2}\left\{
      \frac{\left(1 + 2u^2 + c\right)e^{-u_+^2}
            + \left(1 + 2u^2 - c\right)e^{-u_-^2}}{\sqrt{\pi}}
      + u\,Q(u,X)
        \left(\mathrm{erf}\,u_+ + \mathrm{erf}\,u_-\right)\right\} .

For :math:`X < 0` and :math:`u < 10^{-2}` the expansion of this
expression to second order in :math:`u` is used,

.. math::

   \zeta_\mathrm{coll} = \frac{e^{X}}{\sqrt{\pi}}\left[
      \frac{4\,(2 - X)}{3}
      + u^2\left(\frac{8X^2}{15} - \frac{4X}{5} + \frac{8}{15}\right)
      \right] .

Potential and Coulomb logarithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With *coulomb/self yes* (the default), :math:`\phi_d` is computed
from the *particulate_charge* attribute written by
:doc:`fix particulate/charge <fix_particulate_charge>`,
:math:`\tau` is the local :math:`T_i/T_e`, and

.. math::
   :label: pdrag-lnl

   \ln\Lambda = \frac{1}{2}\,\ln\!\left[
      \frac{b_{90}^2 + (\eta\lambda_s)^2}{b_{90}^2 + r_d^2}\right] ,

.. math::

   b_{90} = \frac{r_d\,|\chi|\,T_e}{T_i\,(3 + 2u^2)} , \qquad
   \lambda_s = \lambda_\mathrm{D}
      \left[1 + \frac{3T_e}{T_i\,(3 + 2u^2)}\right]^{-1/2} , \qquad
   \eta = 1 + \frac{r_d}{\lambda_s}
      \left(1 + \sqrt{\frac{T_e}{6T_i}}\right) .

Here :math:`b_{90}` is the impact parameter for scattering through
a right angle and :math:`\lambda_s` the screening length including
the ions.  :math:`\ln\Lambda` is set to 0 where the expression is
negative, where :math:`\chi = 0` and where :math:`n_e` is not
positive.

With *coulomb/self no* the values of *coulomb/chi*, *coulomb/delta*
and *coulomb/lnlambda* are used for :math:`\chi`, :math:`\tau` and
:math:`\ln\Lambda` on every particle.  The same values are used with
*coulomb/self yes* when no *particulate_charge* attribute exists or
where :math:`T_e` is not positive.  With model *dustt2005* and no
charge fix a warning is printed, and with the default
*coulomb/chi* of 0 only the collection part of the drag remains.

The *charge* keyword names the charge fix.  Model *dis2021* requires
it with *coulomb/self yes* and then checks that the two fixes use the
same model, background fix, material and ion parameters.

Neutral drag
^^^^^^^^^^^^

With *neutrals yes* the drag on neutrals at rest is

.. math::
   :label: pdrag-nun

   \nu_n = \frac{3}{4}\,\frac{n_n m_i v_{T_n}}{\rho_d\, r_d}\;\zeta_n(s) ,
   \qquad
   v_{T_n} = \sqrt{\frac{2T_n}{m_i}} , \qquad
   s = \frac{v}{v_{T_n}} ,

.. math::

   \zeta_n(s) = \frac{1}{s}\left[
      \left(1 + s^2 - \frac{1}{4s^2}\right)\mathrm{erf}\,s
      + \left(s + \frac{1}{2s}\right)\frac{e^{-s^2}}{\sqrt{\pi}}\right] ,

where :math:`n_n` and :math:`T_n` are the neutral density and
temperature of the background.  The neutrals are given the mass of
the background ions.  For :math:`s < 10^{-3}` the limit
:math:`\zeta_n = 8/(3\sqrt{\pi})` is used.  The term is zero where the
background has no positive :math:`n_n` and :math:`T_n`.

Electric force and gravity
^^^^^^^^^^^^^^^^^^^^^^^^^^

With *efield yes* the acceleration :math:`Z_d e\,\mathbf{E}/m_d` is
added, with :math:`\mathbf{E}` the electric field of the background at
the particle position and :math:`Z_d` the *particulate_charge*
attribute.  The term is zero if no charge fix is defined or the
background has no electric field.

The *gravity* keyword adds the constant acceleration
:math:`\mathbf{g}`.  Its three values are the :math:`(R, Z, \phi)`
components.  They are mapped onto the coordinate order of a
two-dimensional planar or axisymmetric run.  In a three-dimensional
run they are taken as the :math:`(x, y, z)` components.

Time integration
^^^^^^^^^^^^^^^^

The fix acts on every step that is a multiple of :math:`N`, once at
the start of the step and once at the end, after the particle move.
Each application advances Eq. :eq:`pdrag-eom` over
:math:`\Delta t_N/2` with the rates and the acceleration held fixed,
using the exact solution

.. math::
   :label: pdrag-update

   \mathbf{v}' = \mathbf{u}_\mathrm{eff} + \frac{\mathbf{a}}{\nu}
      + \left(\mathbf{v} - \mathbf{u}_\mathrm{eff}
              - \frac{\mathbf{a}}{\nu}\right) e^{-\nu\,\Delta t_N/2} ,
   \qquad
   \nu = \nu_i + \nu_n , \quad
   \mathbf{u}_\mathrm{eff} = \frac{\nu_i}{\nu}\,\mathbf{u}_i .

The update is stable for any :math:`\nu\,\Delta t_N`.  Where
:math:`\nu = 0` the velocity changes by
:math:`\mathbf{a}\,\Delta t_N/2`.  The ion drag is zero where
:math:`n_i` or :math:`T_i` is not positive.

Other keywords
^^^^^^^^^^^^^^

The *mixture* keyword restricts the fix to the species of the named
mixture.  Without it the fix acts on every particle.

The *material* keyword takes :math:`\rho_d` from a
:doc:`material <material>`.  Without it the density of liquid lithium,
534 kg/m\ :sup:`3`, is used.

The *radius*, *mass* and *temp* keywords fill in the radius, mass or
temperature of a particle that has none.  Values already present are
not replaced.

Restrictions
------------

This command is part of the OPENEDGE package.  It requires a
:doc:`fix background <fix_background>`.

Model *dis2021* requires a *material* defined by a
:doc:`material <material>` command with a *provenance* string and an
explicit *rho*.  With *coulomb/self yes* it also requires the *charge*
keyword.  The named fix must be a
:doc:`fix particulate/charge <fix_particulate_charge>` with model
*dis2021*, the same background fix, the same material, and
*ion_mass_amu* and *ion_charge_state* equal to A_bg and Z_bg.  It must
be defined before this fix.

Related commands
----------------

:doc:`fix particulate/charge <fix_particulate_charge>`,
:doc:`fix particulate/thermal <fix_particulate_thermal>`,
:doc:`material <material>`,
:doc:`fix background <fix_background>`

Default
-------

The option defaults are model = dustt2005, coulomb/self = yes,
coulomb/chi = 0, coulomb/delta = 1, coulomb/lnlambda = 10, neutrals =
yes, efield = yes, no gravity, no mixture filter, and a density of
534 kg/m\ :sup:`3` (no material).
