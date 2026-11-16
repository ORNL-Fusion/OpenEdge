.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: compute surface/chemical/adatom

compute surface/chemical/adatom command
=======================================

Syntax
------

.. parsed-literal::

   compute ID surface/chemical/adatom group-ID tsurf name dp_flux source keyword value ...

* ID is documented in the :doc:`compute <compute>` command
* surface/chemical/adatom = style name of this compute command
* group-ID = ID of the surface group to evaluate
* *tsurf* name = name of the per-surf custom attribute that holds the
  surface temperature (degrees C)
* *dp_flux* source = source of the incident D\ :sup:`+` flux
  (m\ :sup:`-2` s\ :sup:`-1`)

  * *c_ID* or *c_ID[N]* = per-surf vector, or column N of the per-surf
    array, of a compute
  * *f_ID* or *f_ID[N]* = per-surf vector, or column N of the per-surf
    array, of a fix
  * *s_name* = per-surf custom attribute

* zero or more keyword/value pairs may be appended
* keyword = *Yad* or *Yad_Yps* or *f_ad* or *A_arr* or *E_eff*

  * *Yad* value = ad-atom yield :math:`Y_\mathrm{ad}` for D on Li
  * *Yad_Yps* value = ratio :math:`r_\mathrm{ad}` of the ad-atom yield
    to the physical-sputtering yield
  * *f_ad* value = neutral fraction :math:`f_\mathrm{ad}` of the
    released ad-atoms
  * *A_arr* value = Arrhenius prefactor :math:`A`
  * *E_eff* value = effective binding energy
    :math:`E_\mathrm{eff}` (eV)

Examples
--------

.. code-block:: text

   compute cadat surface/chemical/adatom divertor tsurf Tsurf_lm dp_flux s_Gamma_D_lm
   compute cadat surface/chemical/adatom divertor tsurf Tsurf_lm &
       dp_flux c_dflux[2] Yad 1e-3 E_eff 0.9
   fix femit surface/emit/source LiSource divertor cadat perspecies no &
       normal yes nlaunch_total 200 model thermal_tsurf Tsurf_lm

Description
-----------

Compute the flux of lithium ad-atoms released from each surface
element of the group by the incident deuterium ions.  The release is
thermally activated, so the flux depends on the surface temperature
and on the D\ :sup:`+` flux.  This is the ad-atom source of the
liquid-lithium model of [Diaw2026]_.

The temperature is read from the per-surf custom attribute *name* in
degrees C, usually *Tsurf_lm* written by
:doc:`fix surface/state/lm <fix_surface_state_lm>`, and converted to
:math:`T = T_\mathrm{C} + 273.15` in K.  The ion flux
:math:`\Gamma_\mathrm{D}` is taken from the *dp_flux* source.  A
suitable custom attribute is *Gamma_D_lm*, which fix surface/state/lm
fills when its heat-flux source carries the ion flux.

The released flux is

.. math::
   :label: adatom-flux

   \Gamma_\mathrm{ad}(T, \Gamma_\mathrm{D}) =
      f_\mathrm{ad}\;
      \frac{r_\mathrm{ad}}
           {1 + A\,\exp\!\left(\dfrac{E_\mathrm{eff}}{k_\mathrm{B} T}\right)}\;
      Y_\mathrm{ad}\; \Gamma_\mathrm{D} ,

with :math:`k_\mathrm{B}` the Boltzmann constant and
:math:`E_\mathrm{eff}` converted from eV to J.  At low temperature the
exponential dominates and the flux is proportional to the Arrhenius
factor :math:`\exp(-E_\mathrm{eff}/k_\mathrm{B}T)`.  At high temperature it
saturates at
:math:`f_\mathrm{ad}\, r_\mathrm{ad}\, Y_\mathrm{ad}\, \Gamma_\mathrm{D}/(1+A)`.
The flux is zero where :math:`T < 1` K and on surface elements
outside the group.

A compute given as the *dp_flux* source is invoked by this compute
when needed.  A fix source is read as it stands.

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
element owned by the processor.  The value is the ad-atom flux
:math:`\Gamma_\mathrm{ad}` in m\ :sup:`-2` s\ :sup:`-1`.  It is zero
for surface elements that are not in the group.

The vector can be accessed by any command that uses per-surf values
from a compute as input, for example :doc:`dump surf <dump>`,
:doc:`fix ave/surf <fix_ave_surf>` or
:doc:`fix surface/emit/source <fix_surface_emit_source>`.

Restrictions
------------

This command is part of the OPENEDGE package.

Both *tsurf* and *dp_flux* are required.  The custom attributes named
by *tsurf* and by an *s_name* source must exist when the compute is
initialized.  A *c_ID* or *f_ID* source must produce per-surf values.
A column index must be 1 or larger and cannot be given with an
*s_name* source.

Related commands
----------------

:doc:`fix surface/state/lm <fix_surface_state_lm>`,
:doc:`compute surface/chemical/evaporation <compute_surface_chemical_evaporation>`,
:doc:`fix surface/emit/source <fix_surface_emit_source>`

Default
-------

The option defaults are Yad = 1e-3, Yad_Yps = 1.0, f_ad = 1.0, A_arr
= 1e-7, E_eff = 0.9.
