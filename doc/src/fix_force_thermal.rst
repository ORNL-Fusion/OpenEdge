.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix force/thermal

fix force/thermal command
=========================

Syntax
------

.. parsed-literal::

   fix ID force/thermal Nevery bsource keyword values ...

* ID is documented in the :doc:`fix <fix>` command
* force/thermal = style name of this fix command
* Nevery = apply the force every this many timesteps
* bsource = *background* or *bfield*

  * *background* fix-ID = ID of a :doc:`fix background <fix_background>`
    that supplies :math:`\mathbf{B}`, :math:`\nabla T_i` and
    :math:`\nabla T_e` at the particle position
  * *bfield* BxSRC BySRC BzSRC = sources for the magnetic field
    components (T) in the coordinate order of the simulation box.  Each
    source is *c_ID[col]*, column col of a per-grid compute, or
    *p_name*, a per-particle custom vector.

* one or more keyword/value pairs must be appended
* keyword = *ion_thermal* or *elec_thermal* or *ion_mass_amu*

  * *ion_thermal* values = [*yes* or *no*] [gradTiR_SRC gradTiZ_SRC]

    * *yes* or *no* = turn the ion thermal force on or off
    * gradTiR_SRC, gradTiZ_SRC = sources for the :math:`R` and :math:`Z`
      components of :math:`\nabla T_i` (eV/m).  Required with *bfield*,
      not allowed with *background*.

  * *elec_thermal* values = [*yes* or *no*] [gradTeR_SRC gradTeZ_SRC],
    the same for the electron thermal force and :math:`\nabla T_e`
  * *ion_mass_amu* value = mass of the background ions (amu)

Examples
--------

.. code-block:: text

   fix pd background file plasma.h5 static yes
   fix fth force/thermal 1 background pd ion_thermal elec_thermal
   fix fth force/thermal 20 background pd ion_thermal yes elec_thermal no
   fix fth force/thermal 1 bfield c_cpl[1] c_cpl[2] c_cpl[3] &
       elec_thermal c_cpl[12] c_cpl[13]

Description
-----------

Apply the ion and electron thermal forces to charged simulation
particles.  Both act along the magnetic field and are proportional to
the parallel temperature gradients of the background plasma:

.. math::
   :label: thermal-force

   F_\parallel = \beta_i\,\nabla_\parallel T_i + \alpha_e Z^2\,\nabla_\parallel T_e ,
   \qquad
   \nabla_\parallel = \hat{\mathbf{b}}\cdot\nabla .

Here :math:`Z` is the charge state of the particle and the
temperatures are in energy units.  Other symbols are defined on the
:ref:`notation <notation>` page.

The ion coefficient is the one used in DIVIMP (options CIOPTN = 1 and
3) and given by Stangeby [Stangeby2000]_,

.. math::
   :label: thermal-beta

   \beta_i = \frac{3\left[\mu_m + 5\sqrt{2}\,Z^2
             \left(1.1\,\mu_m^{5/2} - 0.35\,\mu_m^{3/2}\right) - 1\right]}
             {2.6 - 2\mu_m + 5.4\,\mu_m^2} ,
   \qquad
   \mu_m = \frac{m}{m + m_i} ,

where :math:`m` is the mass of the particle and :math:`m_i` the mass of
the background ions.  For a heavy impurity, :math:`\mu_m \to 1` and
:math:`\beta_i \to 2.65\,Z^2`.  The electron coefficient is
:math:`\alpha_e = 0.71`, the Braginskii value for
:math:`Z_\mathrm{eff} = 1` [Braginskii1965]_.  Both terms push an
impurity ion towards higher temperature, which in a divertor usually
means away from the target.

The *ion_thermal* and *elec_thermal* keywords select the terms.  A
keyword given without *yes* or *no* turns its term on.  At least one
term must be on.

Background-ion mass
^^^^^^^^^^^^^^^^^^^

With a *background* source and a plasma file that has a single ion
species, :math:`m_i` is read from the *ion_species/mass_amu* entry of
the file.  If the file has several ion species or no species metadata,
the deuterium mass, 2.01410177811 amu, is used.  The *ion_mass_amu*
keyword overrides both.

Time integration
^^^^^^^^^^^^^^^^

The force is applied as two half-kicks, one at the start of the step
and one at the end, so the particle push between them stays
time-centred.  On a step where the fix acts, each half-kick is

.. math::

   \Delta\mathbf{v} = \frac{F_\parallel}{m}\,\frac{\Delta t_N}{2}\,\hat{\mathbf{b}} ,
   \qquad \Delta t_N = N\,\Delta t ,

so the impulse over :math:`N` steps equals the force integrated over
that interval.  For a particle advanced by the guiding-centre pusher
the same impulse is added to its parallel velocity.

A larger :math:`N` saves field and gradient lookups, but each kick
grows in proportion.  Where the gradients are steep, near the
separatrix or the X-point, keep :math:`N` small or raise the
*subcycles* setting of the :doc:`global pusher <global>` command.

Field and gradient sources
^^^^^^^^^^^^^^^^^^^^^^^^^^

With a *background* source, :math:`\mathbf{B}` is sampled at the
particle position.  The temperature gradients come from the per-cell
gradient datasets of the plasma file if it has them, and otherwise
from bilinear interpolation of the gradient fields on the regular
:math:`(R,Z)` grid.  A native three-dimensional background supplies
the full gradient vector.

With a *bfield* source, the field and gradient components are read
from the listed sources.  A per-grid compute is read in the particle's
cell, and the fix invokes the compute itself.  Gradient sources are
always the :math:`R` and :math:`Z` components.  The toroidal component
is taken as zero.

Neutral particles are not affected, and a particle is skipped where
:math:`B = 0`.

Restrictions
------------

This command is part of the OPENEDGE package.

Related commands
----------------

:doc:`fix cross_field_diffusion <fix_cross_field_diffusion>`,
:doc:`fix coulomb/background <fix_coulomb_background>`,
:doc:`fix background <fix_background>`,
:doc:`global <global>`

Default
-------

Neither term is applied unless its keyword appears.  *ion_mass_amu*
defaults to the ion mass stored in the plasma file, or to the
deuterium mass as described above.
