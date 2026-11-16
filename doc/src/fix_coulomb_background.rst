.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix coulomb/background

fix coulomb/background command
==============================

Syntax
------

.. parsed-literal::

   fix ID coulomb/background Nevery source A_bg Z_bg bg-sources

* ID is documented in the :doc:`fix <fix>` command
* coulomb/background = style name of this fix command
* Nevery = apply the collision operator every this many timesteps
* source = *background* or *plasma*

  * *background* fix-ID = ID of a :doc:`fix background <fix_background>`
    that supplies :math:`T_e`, :math:`n_e`, :math:`T_i`, :math:`n_i`,
    :math:`u_\parallel` and :math:`\mathbf{B}` at the particle position
  * *plasma* TeSRC NeSRC = sources for the electron temperature (eV) and
    electron density (m\ :sup:`-3`) used in the Coulomb logarithm

* A_bg = mass of the background species (amu)
* Z_bg = charge state of the background species (negative for electrons)
* bg-sources = TiSRC NiSRC VparSRC BxSRC BySRC BzSRC, given only with
  *plasma*

  * TiSRC, NiSRC = background temperature (eV) and density (m\ :sup:`-3`)
  * VparSRC = background parallel flow velocity (m/s)
  * BxSRC, BySRC, BzSRC = magnetic field components (T) in the
    coordinate order of the simulation box

* every SRC is *c_ID[col]*, column col of a per-grid compute, or
  *p_name*, a per-particle custom vector

Examples
--------

.. code-block:: text

   fix pd background file plasma.h5 static yes
   fix fcb coulomb/background 5 background pd 2.0 1.0

   fix fcb coulomb/background 1 plasma c_cpl[7] c_cpl[10] 2.0 1.0 &
       c_cpl[8] c_cpl[10] c_cpl[11] c_cpl[1] c_cpl[2] c_cpl[3]

Description
-----------

Collide each charged simulation particle with a fluid background
species.  The background has no particles of its own.  Every
:math:`N` steps a virtual partner of mass
:math:`m_\mathrm{bg} = A_\mathrm{bg}\,\mathrm{amu}` and charge
:math:`q_\mathrm{bg} = Z_\mathrm{bg}\,e` is drawn for each charged
particle, the pair is scattered, and the partner is discarded.  Only
the simulation particle changes velocity, so the background acts as a
reservoir of momentum and energy.  This gives friction against the
background flow, thermalization towards :math:`T_i` and pitch-angle
scattering from one operator.

Symbols are defined on the :ref:`notation <notation>` page.

The partner velocity is sampled from a Maxwellian that drifts along
the magnetic field,

.. math::

   \mathbf{v}_\mathrm{bg} = u_\parallel\,\hat{\mathbf{b}}
      + \sqrt{\frac{T_i}{m_\mathrm{bg}}}\;\boldsymbol{\xi} ,

where :math:`\boldsymbol{\xi}` is a vector of three independent unit
normal numbers.  With
:math:`\mathbf{g} = \mathbf{v}_\mathrm{bg} - \mathbf{v}`, the
scattering parameter is

.. math::

   s = \frac{(q\, q_\mathrm{bg})^2\, n_i\, \ln\Lambda}
            {4\pi\varepsilon_0^2\, m_\mathrm{r}^2\, g^3}\,\Delta t_N ,
   \qquad
   m_\mathrm{r} = \frac{m\, m_\mathrm{bg}}{m + m_\mathrm{bg}} ,

and the scattering angle is sampled exactly as in
:doc:`fix coulomb/binary <fix_coulomb_binary>`, Eq.
:eq:`coulomb-chi`.  The Coulomb logarithm is Eq. :eq:`coulomb-lnl`
evaluated with the local :math:`T_e` and :math:`n_e`.  The particle
velocity changes by

.. math::

   \mathbf{v}' = \mathbf{v} - \frac{m_\mathrm{bg}}{m + m_\mathrm{bg}}\,(\mathbf{g}' - \mathbf{g}) .

With the *background* form every plasma quantity is sampled at the
particle position from the :doc:`fix background <fix_background>`.
With the *plasma* form the quantities are read from the listed
sources.  A per-grid compute source is read in the particle's cell,
and the fix invokes the compute itself on the steps it acts.

A particle advanced by the guiding-centre pusher is scattered on its
reconstructed full velocity, with a gyrophase assigned from the
particle ID and the timestep.  Its parallel velocity and magnetic
moment are then recomputed from the scattered velocity.

Neutral particles are not affected.  A particle is skipped where
:math:`n_i` or :math:`T_i` is not positive.  The operator runs at the
end of the timestep, after the particle move.

Restrictions
------------

This command is part of the OPENEDGE package.

One fix handles one background species.  The density and temperature
that enter :math:`s` are the ion values :math:`n_i` and :math:`T_i` of
the source, so *A_bg* and *Z_bg* must describe that species.

Related commands
----------------

:doc:`fix coulomb/binary <fix_coulomb_binary>`,
:doc:`fix background <fix_background>`,
:doc:`fix neutral/collide/background <fix_neutral_collide_background>`

Default
-------

none
