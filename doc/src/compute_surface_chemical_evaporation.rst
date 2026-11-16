.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: compute surface/chemical/evaporation

compute surface/chemical/evaporation command
============================================

Syntax
------

.. parsed-literal::

   compute ID surface/chemical/evaporation group-ID tsurf name sticking value

* ID is documented in the :doc:`compute <compute>` command
* surface/chemical/evaporation = style name of this compute command
* group-ID = ID of the surface group to evaluate
* *tsurf* name = name of the per-surf custom attribute that holds the
  surface temperature (degrees C)
* *sticking* value = evaporation coefficient :math:`\alpha_\mathrm{s}`
  (> 0)

Examples
--------

.. code-block:: text

   compute cevap surface/chemical/evaporation divertor tsurf Tsurf_lm sticking 1.0
   fix femit surface/emit/source LiSource divertor cevap perspecies no &
       normal yes nlaunch_total 200 model thermal_tsurf Tsurf_lm

Description
-----------

Compute the evaporation flux of lithium atoms from each surface
element of the group as a function of its surface temperature.  This
is the evaporation law of the liquid-lithium model of [Diaw2026]_.

The temperature is read from the per-surf custom attribute *name* in
degrees C, usually *Tsurf_lm* written by
:doc:`fix surface/state/lm <fix_surface_state_lm>`, and converted to
:math:`T = T_\mathrm{C} + 273.15` in K.  The saturated vapour pressure
of lithium follows the two-coefficient Antoine form

.. math::
   :label: evap-pvap

   p_\mathrm{v}(T) = p_\mathrm{atm}\; 10^{\,A - B/T} ,
   \qquad A = 5.055, \quad B = 8023\ \mathrm{K},
   \quad p_\mathrm{atm} = 101325\ \mathrm{Pa} ,

so that :math:`p_\mathrm{v}` is 1.1 Pa at 800 K and 109 Pa at 1000 K.
This is the law of the built-in Li entry of the
:doc:`material <material>` command, so a lithium surface and a lithium
particulate evaporate at the same rate at a given temperature.  The
flux of atoms leaving the surface is the Hertz-Knudsen expression

.. math::
   :label: evap-flux

   \Gamma_\mathrm{evap}(T) = \alpha_\mathrm{s}\,
      \frac{p_\mathrm{v}(T)}{\sqrt{2\pi\, m_\mathrm{Li}\, k_\mathrm{B} T}} ,

with :math:`m_\mathrm{Li} = 1.1526\times10^{-26}` kg the mass of a
lithium atom, :math:`k_\mathrm{B}` the Boltzmann constant and
:math:`\alpha_\mathrm{s}` the *sticking* value.  The flux is the gross
evaporation into vacuum.  No return flux from the vapour is
subtracted.  The flux is set to zero below 298.15 K and on surface
elements outside the group.

The values are meant as the emission source of lithium atoms for
:doc:`fix surface/emit/source <fix_surface_emit_source>`, or for
output with :doc:`dump surf <dump>`.

The same attribute can be given to the *tsurf* keyword of
:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`
and to *thermal_tsurf* of
:doc:`fix surface/emit/source <fix_surface_emit_source>`, which also
read it in degrees C.

Output info
-----------

This compute calculates a per-surf vector with one value per surface
element owned by the processor.  The value is the evaporation flux
:math:`\Gamma_\mathrm{evap}` in m\ :sup:`-2` s\ :sup:`-1`.  It is zero
for surface elements that are not in the group.

The vector can be accessed by any command that uses per-surf values
from a compute as input, for example :doc:`dump surf <dump>`,
:doc:`fix ave/surf <fix_ave_surf>` or
:doc:`fix surface/emit/source <fix_surface_emit_source>`.

Restrictions
------------

This command is part of the OPENEDGE package.

The custom attribute named by *tsurf* must exist when the compute is
initialized.  It is created by
:doc:`fix surface/state/lm <fix_surface_state_lm>`.

Both keyword/value pairs must be given.  The command is rejected with
fewer than six arguments, so *sticking* cannot be left out even for
the value 1.0.

Related commands
----------------

:doc:`fix surface/state/lm <fix_surface_state_lm>`,
:doc:`compute surface/chemical/adatom <compute_surface_chemical_adatom>`,
:doc:`fix surface/emit/source <fix_surface_emit_source>`

Default
-------

The internal default is sticking = 1.0, but the keyword must be given,
as described above.
