.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. _notation:

Notation and units
==================

The OpenEdge command pages share one set of symbols.  A page defines
any symbol that is not listed here.

Units
-----

OpenEdge runs in SI units (:doc:`units si <units>`).  Lengths are in
m, times in s, masses in kg, densities in m\ :sup:`-3` and magnetic
fields in T.  Two exceptions follow fusion practice:

* Temperatures and energies are given in eV in input scripts, plasma
  files and output.  In the equations a temperature :math:`T` stands
  for :math:`k_\mathrm{B}T`, an energy.
* Species masses are given in amu in the species file and in keywords
  such as *ion_mass_amu*.

Charge states are in units of the elementary charge :math:`e`.

Coordinates
-----------

Cylindrical coordinates are :math:`(R, \phi, Z)` with :math:`Z` along
the symmetry axis of the device.  How they map onto the simulation box
depends on the :doc:`dimension <dimension>` of the run.  The mapping is
given in :ref:`Section 4.2 <howto_2>` and in the
:ref:`coordinate table <background_coords>` of the
:doc:`fix background <fix_background>` page.  The normalized poloidal
flux :math:`\psi_N` is 0 on the magnetic axis and 1 on the separatrix.

Vectors are bold, :math:`\mathbf{v}`, and their magnitudes are italic,
:math:`v`.  The unit vector along the magnetic field is
:math:`\hat{\mathbf{b}} = \mathbf{B}/B`.  Subscripts :math:`\parallel`
and :math:`\perp` denote components along and across
:math:`\hat{\mathbf{b}}`.

Symbols
-------

.. list-table::
   :header-rows: 1
   :widths: 18 62 20

   * - Symbol
     - Meaning
     - Unit
   * - :math:`\Delta t`
     - timestep set by the :doc:`timestep <timestep>` command
     - s
   * - :math:`N`
     - the *Nevery* argument of a fix; the fix acts every :math:`N` steps
     -
   * - :math:`\Delta t_N = N\,\Delta t`
     - time covered by one application of a fix
     - s
   * - :math:`m,\; q = Ze,\; \mathbf{x},\; \mathbf{v}`
     - mass, charge, position and velocity of a simulation particle
     - kg, C, m, m/s
   * - :math:`Z`
     - charge state of a simulation particle
     -
   * - :math:`w`
     - weight of a simulation particle, the number of real particles
       it represents.  It equals :math:`F_\mathrm{num}` unless
       :doc:`fix particle/weight <fix_particle_weight>` is used.
     -
   * - :math:`F_\mathrm{num}`
     - number of real particles per simulation particle, the *fnum*
       setting of the :doc:`global <global>` command
     -
   * - :math:`n_e,\; T_e`
     - electron density and temperature of the background plasma
     - m\ :sup:`-3`, eV
   * - :math:`n_i,\; T_i,\; m_i,\; Z_i`
     - density, temperature, mass and charge state of the background ions
     - m\ :sup:`-3`, eV, kg
   * - :math:`u_\parallel`
     - parallel flow velocity of the background ions
     - m/s
   * - :math:`\mathbf{B},\; \mathbf{E}`
     - magnetic and electric field
     - T, V/m
   * - :math:`\psi,\; \psi_N`
     - poloidal flux per radian and its normalized value,
       :math:`\psi_N = (\psi - \psi_\mathrm{axis})/(\psi_\mathrm{sep} - \psi_\mathrm{axis})`
       with :math:`\psi_\mathrm{axis}` on the magnetic axis and
       :math:`\psi_\mathrm{sep}` on the separatrix
     - T m\ :sup:`2`, --
   * - :math:`B_R,\; B_Z,\; B_\phi`
     - cylindrical components of the magnetic field
     - T
   * - :math:`\Omega = qB/m`
     - gyrofrequency
     - rad/s
   * - :math:`\rho_\mathrm{L} = v_\perp/|\Omega|`
     - Larmor radius
     - m
   * - :math:`\mu = m v_\perp^2/(2B)`
     - magnetic moment
     - J/T
   * - :math:`\mathbf{g},\; g`
     - relative velocity of two collision partners and its magnitude
     - m/s
   * - :math:`m_\mathrm{r}`
     - reduced mass of two collision partners
     - kg
   * - :math:`\chi`
     - scattering angle in the centre-of-mass frame, on the collision
       pages.  On the particulate pages :math:`\chi = -e\phi_d/T_e` is
       the normalized surface potential of a grain.
     - rad, --
   * - :math:`\ln\Lambda`
     - Coulomb logarithm
     -
   * - :math:`\lambda_\mathrm{D}`
     - electron Debye length,
       :math:`(\varepsilon_0 T_e / n_e e^2)^{1/2}`
     - m
   * - :math:`c_s`
     - ion sound speed, :math:`[(T_e + T_i)/m_i]^{1/2}`
     - m/s
   * - :math:`N_\mathrm{sub},\; \delta t`
     - number of pusher subcycles and the substep
       :math:`\delta t = \Delta t/N_\mathrm{sub}`
     - --, s
   * - :math:`\mathbf{X},\; v_\parallel`
     - guiding-centre position and parallel velocity
     - m, m/s
   * - :math:`\Phi`
     - potential drop across the sheath and magnetic presheath
     - V
   * - :math:`D_\perp`
     - cross-field diffusivity
     - m\ :sup:`2`/s
   * - :math:`\Gamma`
     - particle flux density onto or from a surface
     - m\ :sup:`-2` s\ :sup:`-1`
   * - :math:`E_0,\; \theta`
     - impact energy and impact angle, measured from the surface normal
     - eV, degrees
   * - :math:`Y(E_0,\theta)`
     - sputter yield, atoms removed per incident particle
     -
   * - :math:`R_N`
     - particle reflection coefficient
     -
   * - :math:`E_\mathrm{s}`
     - surface binding energy
     - eV
   * - :math:`\sigma_s`
     - areal density of species :math:`s` on a surface element
     - m\ :sup:`-2`
   * - :math:`U`
     - random number uniform on (0,1)
     -
   * - :math:`\xi`
     - random number from a normal distribution of zero mean and unit
       variance
     -
   * - :math:`r_d,\; m_d,\; T_d,\; Z_d`
     - radius, mass, temperature and charge number of a particulate
       (dust grain or droplet)
     - m, kg, K, --
   * - :math:`\phi_d`
     - surface potential of a particulate relative to the plasma
     - V
   * - :math:`k_j,\; \lambda_j`
     - rate coefficient of reaction :math:`j` and its expected number
       of events per particle in :math:`\Delta t_N`
     - m\ :sup:`3`/s, --

Physical constants (:math:`e`, :math:`\varepsilon_0`,
:math:`k_\mathrm{B}`, :math:`m_e`) have their usual meaning and CODATA
values.
