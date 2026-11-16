.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: compute grid/weighted

compute grid/weighted command
=============================

Syntax
------

.. parsed-literal::

   compute ID grid/weighted group-ID mix-ID value1 value2 ... keyword arg

* ID is documented in the :doc:`compute <compute>` command
* grid/weighted = style name of this compute command
* group-ID = group ID for which grid cells to perform calculation on
* mix-ID = mixture ID to perform calculation on
* one or more values must be appended
* value = *n_w* or *nrho_w* or *mass_w* or *massrho_w* or *nfrac_w* or
  *massfrac_w* or *u_w* or *v_w* or *w_w* or *usq_w* or *vsq_w* or
  *wsq_w* or *ke_w* or *temp_w* or *erot_w* or *evib_w* or *trot_w* or
  *tvib_w* or *pxrho_w* or *pyrho_w* or *pzrho_w* or *kerho_w* or
  *pradtot_w* or *prad_w*

  * *n_w* = number of real particles in the cell
  * *nrho_w* = number density
  * *mass_w* = mean particle mass
  * *massrho_w* = mass density
  * *nfrac_w* = number fraction of the group in the cell
  * *massfrac_w* = mass fraction of the group in the cell
  * *u_w*, *v_w*, *w_w* = x, y, z components of the mass-averaged
    velocity
  * *usq_w*, *vsq_w*, *wsq_w* = mass-averaged squares of the x, y, z
    velocity components
  * *ke_w* = mean kinetic energy per particle
  * *temp_w* = temperature
  * *erot_w* = mean rotational energy per particle
  * *evib_w* = mean vibrational energy per particle
  * *trot_w* = rotational temperature
  * *tvib_w* = vibrational temperature
  * *pxrho_w*, *pyrho_w*, *pzrho_w* = x, y, z components of the
    momentum density
  * *kerho_w* = kinetic energy density
  * *pradtot_w* = power radiated by the particles in the cell
  * *prad_w* = radiated power density

* zero or one keyword/arg pair may be given anywhere in the value list
* keyword = *adas*

  * *adas* arg = fix-ID = ID of the
    :doc:`fix volume/chem/adas <fix_volume_chem_adas>` that holds the
    radiation tables.  Required by *pradtot_w* and *prad_w*.

Examples
--------

.. code-block:: text

   fix pw particle/weight
   mixture LiAll Li Li+ Li2+ Li3+
   compute cgw grid/weighted all LiAll nrho_w u_w v_w temp_w nfrac_w
   fix 1 ave/grid all 10 100 1000 c_cgw[*]
   dump 1 grid all 1000 tmp.grid id f_1[*]

   fix fchem volume/chem/adas 1 W auto
   mixture allW W W+ W2+ W3+ group wsum
   compute cprad grid/weighted all allW prad_w pradtot_w adas fchem
   compute Prad reduce sum c_cprad[2]

Description
-----------

Compute per-grid-cell moments of the particle distribution in which
each simulation particle counts with its own statistical weight.  The
values are the counterparts of those of
:doc:`compute grid <compute_grid>`, which weights every particle by
the global :math:`F_\mathrm{num}`.  The *_w* suffix marks the
weighting.

The weight :math:`w_i` of particle :math:`i` is the number of real
particles it represents.  It is the custom attribute *pweight* created
by :doc:`fix particle/weight <fix_particle_weight>` and set at launch
by the weighted emission commands, for example
:doc:`fix surface/emit/source <fix_surface_emit_source>`.  A particle
that no emitter has weighted carries :math:`w_i = F_\mathrm{num}`.
The compute applies no further factor of :math:`F_\mathrm{num}`.  If
all particles are unweighted, every value therefore equals the
corresponding value of :doc:`compute grid <compute_grid>`.  If the
weights differ between particles, only the values of this compute are
correctly normalized.

Symbols are defined on the :ref:`notation <notation>` page.  In
addition, :math:`m_i`, :math:`\mathbf{v}_i = (v_{x,i}, v_{y,i},
v_{z,i})`, :math:`\varepsilon_{\mathrm{rot},i}` and
:math:`\varepsilon_{\mathrm{vib},i}` are the mass, velocity,
rotational energy and vibrational energy of particle :math:`i`,
:math:`V_c` is the volume of the grid cell, and :math:`W_c` is the
weight of the cell set by the *weight* keyword of the
:doc:`global <global>` command (1 if cell weighting is off).

Sums
^^^^

A value is computed for each group of the mixture.  For one group
:math:`g` and one cell, the sums over the particles of the group in
the cell are

.. math::
   :label: gridw-sums

   S_0 = \sum_{i \in g} w_i , \qquad
   S_m = \sum_{i \in g} w_i m_i , \qquad
   \mathbf{S}_p = \sum_{i \in g} w_i m_i \mathbf{v}_i , \qquad
   S_E = \sum_{i \in g} w_i m_i v_i^2 .

Densities
^^^^^^^^^

The extensive values are normalized by the cell volume:

.. math::
   :label: gridw-dens

   \begin{aligned}
   \mathit{n\_w} &= S_0 , &
   \mathit{nrho\_w} &= \frac{W_c}{V_c}\, S_0 , \\
   \mathit{massrho\_w} &= \frac{W_c}{V_c}\, S_m , &
   (\mathit{pxrho\_w}, \mathit{pyrho\_w}, \mathit{pzrho\_w})
       &= \frac{W_c}{V_c}\, \mathbf{S}_p , \\
   \mathit{kerho\_w} &= \frac{W_c}{V_c}\, \frac{S_E}{2} . &&
   \end{aligned}

For unweighted particles :math:`S_0 = F_\mathrm{num} N_g`, where
:math:`N_g` is the number of simulation particles of the group in the
cell, and *nrho_w* is the *nrho* value of
:doc:`compute grid <compute_grid>`.

Means
^^^^^

The intensive values are ratios of the sums of Eq. :eq:`gridw-sums`:

.. math::
   :label: gridw-mean

   \begin{aligned}
   \mathit{mass\_w} &= \frac{S_m}{S_0} , &
   (\mathit{u\_w}, \mathit{v\_w}, \mathit{w\_w})
       &= \frac{\mathbf{S}_p}{S_m} , \\
   \mathit{usq\_w} &= \frac{\sum_{i \in g} w_i m_i v_{x,i}^2}{S_m} , &
   \mathit{ke\_w} &= \frac{S_E}{2\, S_0} , \\
   \mathit{temp\_w} &= \frac{S_E}{3 k_\mathrm{B}\, S_0} , &&
   \end{aligned}

and likewise *vsq_w* and *wsq_w* with :math:`v_{y,i}` and
:math:`v_{z,i}`.  The temperature *temp_w* is formed from the full
particle velocities.  The mean flow velocity is not subtracted, as in
:doc:`compute grid <compute_grid>`.

The internal-energy values are

.. math::
   :label: gridw-int

   \mathit{erot\_w} = \frac{\sum_{i \in g} w_i\,
                      \varepsilon_{\mathrm{rot},i}}{S_0} ,
   \qquad
   \mathit{trot\_w} = \frac{2}{k_\mathrm{B}}\,
      \frac{\sum_{i \in g} w_i\, \varepsilon_{\mathrm{rot},i}}
           {\sum_{i \in g} w_i\, \zeta_{\mathrm{rot},i}} ,

where :math:`\zeta_{\mathrm{rot},i}` is the number of rotational
degrees of freedom of the species of particle :math:`i`.  The values
*evib_w* and *tvib_w* are the same with the vibrational energy and the
vibrational degrees of freedom.

In units other than SI the kinetic terms of Eqs. :eq:`gridw-dens` and
:eq:`gridw-mean` and the numerators of *trot_w* and *tvib_w* carry the
conversion factor from mass times velocity squared to energy of the
:doc:`units <units>` command.

Group fractions
^^^^^^^^^^^^^^^

The fractions divide a group sum by the same sum over the particles of
all groups of the mixture in the cell:

.. math::

   \mathit{nfrac\_w} = \frac{S_0}{\sum_{g'} S_0^{(g')}} ,
   \qquad
   \mathit{massfrac\_w} = \frac{S_m}{\sum_{g'} S_m^{(g')}} .

Radiated power
^^^^^^^^^^^^^^

The radiated-power values sum the radiation of every particle at its
own position:

.. math::
   :label: gridw-prad

   \mathit{pradtot\_w} = \sum_{i \in g} w_i\, n_{e,i}\, L_i ,
   \qquad
   \mathit{prad\_w} = \frac{W_c}{V_c} \sum_{i \in g} w_i\, n_{e,i}\, L_i .

Here :math:`n_{e,i}` and :math:`T_{e,i}` are the electron density and
temperature at particle :math:`i`, read from the per-particle plasma
cache.  The power coefficient
:math:`L_i = L^\mathrm{PLT}(T_{e,i}, n_{e,i}) +
L^\mathrm{PRB}(T_{e,i}, n_{e,i})` is the sum of the ADAS line
radiation coefficient (PLT) and the recombination and bremsstrahlung
coefficient (PRB) of the charge state of the particle, in W
m\ :sup:`3`.  Both are interpolated from the tables of the
:doc:`fix volume/chem/adas <fix_volume_chem_adas>` named by the *adas*
keyword, with :math:`T_e` and :math:`n_e` clamped to the table range.
A species that takes part in no active reaction of that fix
contributes zero.

In a cell with uniform plasma, *prad_w* equals :math:`n_e` times the
sum over charge states of *nrho_w* times :math:`L`.  Where the plasma
varies inside a cell, each particle uses its local values.  The
radiation is a diagnostic.  It is not removed from the background
plasma.

Output
^^^^^^

The compute produces a per-grid array.  The number of columns is the
number of values times the number of groups in the mixture.  The
columns of the first group come first, in the order the values were
given, followed by the columns of the second group, and so on.  With
the values *nrho_w* and *u_w*, columns 1 and 2 are *nrho_w* and *u_w*
of the first group and columns 3 and 4 are those of the second group.

The array can be used by any command that takes per-grid values from a
compute, for example :doc:`dump grid <dump>` and
:doc:`fix ave/grid <fix_ave_grid>`.  See
:ref:`Section 4.8 <howto_8>` for an overview of output options.
:doc:`fix ave/grid <fix_ave_grid>` averages the sums of Eq.
:eq:`gridw-sums` and forms the ratios from the averaged sums.

A value whose denominator is zero in a cell is set to zero there.
Cells that are not in the grid group hold zeros.  So do split cells,
whose particles are tallied in their sub-cells.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Values
     - Unit (SI)
   * - *n_w*
     - real particles
   * - *nrho_w*
     - m\ :sup:`-3`
   * - *mass_w*
     - kg
   * - *massrho_w*
     - kg m\ :sup:`-3`
   * - *nfrac_w*, *massfrac_w*
     - dimensionless
   * - *u_w*, *v_w*, *w_w*
     - m/s
   * - *usq_w*, *vsq_w*, *wsq_w*
     - m\ :sup:`2`/s\ :sup:`2`
   * - *ke_w*, *erot_w*, *evib_w*
     - J
   * - *temp_w*, *trot_w*, *tvib_w*
     - K
   * - *pxrho_w*, *pyrho_w*, *pzrho_w*
     - kg m\ :sup:`-2` s\ :sup:`-1`
   * - *kerho_w*
     - J m\ :sup:`-3`
   * - *pradtot_w*
     - W
   * - *prad_w*
     - W m\ :sup:`-3`

.. note::

   The temperatures of this compute are in K, the temperature unit of
   the :doc:`units <units>` command, and the energies in J.  They are
   not in eV.

Restrictions
------------

This command is part of the OPENEDGE package.

A :doc:`fix particle/weight <fix_particle_weight>` must be defined so
that the *pweight* attribute exists.  The compute stops with an error
otherwise.  The number of groups in the mixture must not change after
the compute is defined.

The values *pradtot_w* and *prad_w* require the *adas* keyword and the
per-particle plasma cache, which exists when the *pusher* keyword of
the :doc:`global <global>` command names a plasma provider.  The cache
is filled inside the timestep, before the particles move, with the
plasma at the particle positions at that time.  It is refreshed every
N steps, where N is the smallest *Nevery* of the fixes that read it.
A particle that has not been through a refresh has
:math:`n_{e,i} = 0` and contributes nothing, so both values are zero
at step 0 of a run that starts with new particles.  They are not
available with the KOKKOS variant of this compute.

Related commands
----------------

:doc:`fix particle/weight <fix_particle_weight>`,
:doc:`compute grid <compute_grid>`,
:doc:`fix ave/grid <fix_ave_grid>`,
:doc:`fix surface/emit/source <fix_surface_emit_source>`,
:doc:`fix volume/chem/adas <fix_volume_chem_adas>`

Default
-------

none
