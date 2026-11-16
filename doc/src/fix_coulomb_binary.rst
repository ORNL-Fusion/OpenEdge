.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: fix coulomb/binary

fix coulomb/binary command
==========================

Syntax
------

.. parsed-literal::

   fix ID coulomb/binary Nevery source

* ID is documented in the :doc:`fix <fix>` command
* coulomb/binary = style name of this fix command
* Nevery = apply the collision operator every this many timesteps
* source = *background* or *plasma*

  * *background* fix-ID = ID of a :doc:`fix background <fix_background>`
    that supplies :math:`T_e` and :math:`n_e` at the particle position
  * *plasma* TeSRC NeSRC = sources for the electron temperature (eV) and
    electron density (m\ :sup:`-3`).  Each source is *c_ID[col]*, column
    col of a per-grid compute, or *p_name*, a per-particle custom vector.

Examples
--------

.. code-block:: text

   fix pd background file plasma.h5 static yes
   fix fcp coulomb/binary 1 background pd

   fix fcp coulomb/binary 2 plasma c_cpl[7] c_cpl[10]

Description
-----------

Collide the charged simulation particles with each other, cell by
cell.  The operator is the cumulative small-angle scattering model of
Nanbu [Nanbu1997]_ with the random pairing of Takizuka and Abe
[Takizuka1977]_.  Collisions with the fluid background plasma are a
separate operator, :doc:`fix coulomb/background <fix_coulomb_background>`,
and the two can be used together.

Symbols are defined on the :ref:`notation <notation>` page.

Pairing
^^^^^^^

Every :math:`N` steps the :math:`N_c` charged particles of a cell are
put in random order and paired as (1,2), (3,4), and so on.  If
:math:`N_c` is odd, the last particle is paired with a randomly chosen
earlier one, which then scatters twice.  Cells with fewer than two
charged particles are skipped, and so are neutral particles.

Scattering
^^^^^^^^^^

For a pair :math:`(a,b)` with relative velocity
:math:`\mathbf{g} = \mathbf{v}_b - \mathbf{v}_a`, the cumulative effect
of the small-angle collisions over the interval
:math:`\Delta t_N = N\,\Delta t` is set by the dimensionless parameter

.. math::
   :label: coulomb-s

   s = \frac{(q_a q_b)^2\, n\, \ln\Lambda}
            {4\pi\varepsilon_0^2\, m_\mathrm{r}^2\, g^3}\,\Delta t_N ,
   \qquad
   m_\mathrm{r} = \frac{m_a m_b}{m_a + m_b} .

The partner density is that of the charged simulation particles in the
cell,

.. math::

   n = \frac{N_c\, F_\mathrm{num}}{V_\mathrm{cell}} .

The scattering angle :math:`\chi` is then sampled from Nanbu's
distribution:

.. math::
   :label: coulomb-chi

   \cos\chi =
   \begin{cases}
     1 + s \ln U, & s \le 0.01, \\[4pt]
     \dfrac{1}{A}\,\ln\!\left(e^{-A} + 2U\sinh A\right), & 0.01 < s \le 6, \\[8pt]
     2U - 1, & s > 6,
   \end{cases}

where :math:`A(s)` is the tabulated solution of
:math:`\coth A - 1/A = e^{-s}`.  The last branch is isotropic
scattering.  The azimuthal angle is uniform on :math:`[0, 2\pi)`.

With :math:`\mathbf{g}'` the rotated relative velocity, the two
particles are updated as

.. math::

   \mathbf{v}_a' = \mathbf{v}_a - \frac{m_b}{m_a + m_b}\,(\mathbf{g}' - \mathbf{g}),
   \qquad
   \mathbf{v}_b' = \mathbf{v}_b + \frac{m_a}{m_a + m_b}\,(\mathbf{g}' - \mathbf{g}),

which conserves the momentum and kinetic energy of the pair to
round-off.

Coulomb logarithm
^^^^^^^^^^^^^^^^^

The plasma *source* enters only through the Coulomb logarithm,

.. math::
   :label: coulomb-lnl

   \ln\Lambda = \max\!\left[\,2,\; \ln\!\left(12\pi\, n_e \lambda_\mathrm{D}^3\right)\right],
   \qquad
   \lambda_\mathrm{D} = \sqrt{\frac{\varepsilon_0 T_e}{n_e e^2}} ,

with :math:`T_e` and :math:`n_e` averaged over the two partners.  It is
set to 2 where :math:`T_e` or :math:`n_e` is not positive.  A per-grid
compute source is read in the particle's cell, and the fix invokes the
compute itself when it needs it.

The operator runs at the end of the timestep, after the particle move.

Restrictions
------------

This command is part of the OPENEDGE package.

The partner density :math:`n` uses the global :math:`F_\mathrm{num}` of
the :doc:`global <global>` command and ignores per-particle weights
:math:`w`.  With :doc:`fix particle/weight <fix_particle_weight>` the
collision rate is therefore that of equally weighted particles.

Related commands
----------------

:doc:`fix coulomb/background <fix_coulomb_background>`,
:doc:`fix background <fix_background>`

Default
-------

none
