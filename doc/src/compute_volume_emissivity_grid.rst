.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: compute volume/emissivity/grid

compute volume/emissivity/grid command
======================================

Syntax
------

.. parsed-literal::

   compute ID volume/emissivity/grid group-ID mix-ID keyword values ...

* ID is documented in the :doc:`compute <compute>` command
* volume/emissivity/grid = style name of this compute command
* group-ID = group ID for which grid cells to perform the calculation on
* mix-ID = mixture ID whose species groups are tallied
* the *pec* and *plasma_compute* keywords are required
* keyword = *pec* or *plasma_compute* or *pec_units*

  * *pec* values = elem pec_id line = selector of the photon emissivity
    coefficient table */volume/pec/elem/pec_id/line* in the process
    database
  * *plasma_compute* value = ID of a
    :doc:`compute plasma/fields <compute_plasma_fields>` that supplies
    :math:`T_e` and :math:`n_e`
  * *pec_units* value = *cm3s* or *m3s* = units of the stored
    coefficient, cm\ :sup:`3`/s or m\ :sup:`3`/s

Examples
--------

.. code-block:: text

   mixture liplot Li Li+ Li2+ Li3+ group SELF
   compute cpl plasma/fields all file plasma.h5
   compute cex volume/emissivity/grid all liplot pec li 96 li0_6710_pec &
       plasma_compute cpl
   compute crc volume/emissivity/grid all liplot pec li 96 &
       li0_6710_recom_pec plasma_compute cpl
   fix aem ave/grid all 1 1000 1000 c_cex[1] c_crc[2]
   dump 1 grid all 1000 emis.*.dat id f_aem[*]

Description
-----------

Compute the volumetric emissivity of one spectral line in every grid
cell of the group,

.. math::
   :label: emis-line

   \varepsilon_g = n_e\, n_g\, \mathrm{PEC}(T_e, n_e) ,

where :math:`n_g` is the number density of species group :math:`g` of
the mixture and PEC is the photon emissivity coefficient of the line
(photons m\ :sup:`3`/s).  The result is the number of photons emitted
per unit volume and time into the full solid angle, in photons
m\ :sup:`-3` s\ :sup:`-1`.  The compute applies no solid-angle
factor.  A radiance along a line of sight, in photons
m\ :sup:`-2` s\ :sup:`-1` sr\ :sup:`-1`, is the path integral of
:math:`\varepsilon_g/4\pi`.

There is one column per species group.  The compute evaluates
Eq. :eq:`emis-line` with the same coefficient for every group and
does not check which species a group holds, so only the column of the
species that the table refers to is the emission of the line.  For an
excitation coefficient that species is the emitting charge state.  For
a recombination coefficient it is the next higher charge state, which
recombines into the emitting one.  In the example above the groups
are the charge states, the first compute is used in column 1 (Li) and
the second in column 2 (Li\ :sup:`+`), and the two contributions add
to the Li I 670.8 nm emission.

Other symbols are defined on the :ref:`notation <notation>` page.

The group density is formed from the particles of the group in the
cell,

.. math::

   n_g = \frac{f_c}{V} \sum_{p \in g} w_p ,

where :math:`w_p` is the *pweight* attribute of particle :math:`p`,
the number of real particles it stands for
(:doc:`fix particle/weight <fix_particle_weight>`), :math:`V` is the
cell volume and :math:`f_c` the grid-cell weight set by the
:doc:`global weight <global>` command.

The coefficient table is read from the process database,
*database/processes.h5*, under */volume/pec/elem/pec_id/line*.  It
holds :math:`\log_{10}\mathrm{PEC}` on a grid of
:math:`\log_{10} T_e` (eV) and :math:`\log_{10} n_e`
(m\ :sup:`-3`), ingested from open-ADAS adf15 files by
*database/ingest/add_pec_tables.py*.  The script names the tables
*<ion>_<wavelength>_pec* for excitation and
*<ion>_<wavelength>_recom_pec* for recombination, with the wavelength
in Angstrom, and keeps the coefficient in the ADAS unit of
cm\ :sup:`3`/s.  The compute
interpolates bilinearly in these logarithms and holds the value at
the nearest edge outside the grid.  With *pec_units cm3s*, the ADAS
convention, the coefficient is multiplied by 10\ :sup:`-6` to convert
it to m\ :sup:`3`/s.  Use *m3s* if the table is already in SI units.

:math:`T_e` and :math:`n_e` are queried from the plasma/fields compute
at the cell centre.  The emissivity is zero in a cell that has no
particle of the group, or where :math:`T_e` or :math:`n_e` is not
positive.

The compute tallies :math:`\sum w_p` when it is invoked and applies
the plasma factors afterwards.  When it is time averaged by
:doc:`fix ave/grid <fix_ave_grid>`, the accumulated weights are
divided by the number of samples before Eq. :eq:`emis-line` is
evaluated, so the average is the emissivity of the mean density.

Output info
-----------

This compute calculates a per-grid array with one column per species
group of the mixture.  The values are in photons
m\ :sup:`-3` s\ :sup:`-1`.  They can be used by any command that uses
per-grid values from a compute as input, such as
:doc:`dump grid <dump>` or :doc:`fix ave/grid <fix_ave_grid>`.

Restrictions
------------

This command is part of the OPENEDGE package.

It requires :doc:`fix particle/weight <fix_particle_weight>`, which
provides the *pweight* attribute, and a
:doc:`compute plasma/fields <compute_plasma_fields>`.  The process
database must contain the requested table.  Otherwise the compute
stops with an error.

Related commands
----------------

:doc:`compute plasma/fields <compute_plasma_fields>`,
:doc:`fix particle/weight <fix_particle_weight>`,
:doc:`fix volume/chem/adas <fix_volume_chem_adas>`,
:doc:`fix ave/grid <fix_ave_grid>`

Default
-------

The option default is pec_units = cm3s.
