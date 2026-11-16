.. hand-edited: this page is maintained here, not generated from doc/*.txt

.. index:: global

global command
==============

Syntax
------

.. parsed-literal::

   global keyword values ...

* one or more keyword/value pairs

* keyword = *fnum* or *nrho* or *vstream* or *temp* or *field* or *efield* or *bfield* or *surfs* or *surfgrid* or *surfmax* or *splitmax* or *surftally* or *gridcut* or *comm/sort* or *comm/style* or *weight* or *particle/reorder* or *mem/limit* or *optmove* or *pusher* or *phi_track*

  .. parsed-literal::

       *fnum* value = ratio
         ratio = Fnum ratio of physical particles to simulation particles
       *nrho* value = density
         density = number density of background gas (# per length^3 units)
       *vstream* values = Vx Vy Vz
         Vx,Vy,Vz = streaming velocity of background gas (velocity units)
       *temp* values = thermal
         thermal = temperature of background gas (temperature units)
       *field* values = fstyle args
         fstyle = *none* or *constant* or *particle* or *grid*
           *field* arg = none
           *constant* args = mag ex ey ez
             mag = magnitude of field acceleration (acceleration units)
             ex,ey,ez = direction vector which the field acts along
           *particle* arg = fixID
             fixID = ID of fix that computes per particle field components
           *grid* args = fixID Nfreq
             fixID = ID of fix that computes per grid cell field components
             Nfreq = update field values every this many timesteps
       *surfs* value = explicit or explicit/distributed or implicit
         explicit = surfs defined in read_surf file, each proc owns copy of all surfs
         explicit/distributed = surfs defined in read_surf file, each proc owns
                                only the surfs for its owned_ghost grid cells
         implicit = surfs defined in read_isurf file, each proc owns
                                only the surfs for its owned+ghost grid cells
       *surfgrid* value = *percell* or *persurf* or *auto*
         percell = loop over my cells and check every surf
         persurf = loop over my surfs and cells they overlap
         auto = choose percell or persurf based on surface element and proc count
       *surfmax* value = Nsurf
         Nsurf = max # of surface elements allowed in single grid cell
       *splitmax* value = Nsplit
         Nsplit = max # of sub-cells one grid cell can be split into by surface elements
       *surftally* value = *reduce* or *rvous* or *auto*
         reduce = tally surf collision info via MPI_Allreduce operations
         rvous = tally via a rendezvous algorithm
         auto = choose reduce or rvous based on surface element and proc count
       *gridcut* value = cutoff
         cutoff = acquire ghost cells up to this far away (distance units)
       *comm/sort* value = yes or no
         yes/no = sort incoming messages by proc ID if yes, else no sort
       *comm/style* value = neigh or all
         neigh = setup particle comm with subset of near-neighbor processor
         all = allow particle comm with potentially any processor
       *weight* value = *wstyle* *mode*
         wstyle = *cell*
         mode = *none* or *volume* or *radius* or *radius/only*
       *particle/reorder* value = *nsteps*
         nsteps = reorder the particles every this many timesteps
       *mem/limit* value = *grid* or bytes
         grid = limit extra memory for load-balancing, particle reordering, and restart file read/write to grid cell memory
         bytes = limit extra particle memory to this amount (in MBytes)
       *optmove* value = yes or no
         yes/no = use optimized particle move if yes, else use regular move

  * *efield* values = efstyle args

    * efstyle = *none* or *particle* or *grid*
    * *none* args = none
    * *particle* args = fixID Nfreq

      * fixID = ID of a :doc:`fix efield/particle <fix_efield_particle>`
        that computes the electric field components per particle
      * Nfreq = read but not used.  The field is recomputed every
        timestep.

    * *grid* args = fixID Nfreq

      * fixID = ID of a fix that computes the electric field
        components per grid cell
      * Nfreq = recompute the field every this many timesteps (0 =
        once at the start of a run)

  * *bfield* values = bfstyle args, the same as *efield* for the
    magnetic field, with a
    :doc:`fix bfield/particle <fix_bfield_particle>` for the
    *particle* style
  * *pusher* values = one or more pusher keyword/value pairs
  * pusher keyword = *mode* or *plasma* or *skip* or *subcycles* or
    *gca_switch* or *gca_integrator* or *boris_near* or *gc_wall* or
    *switch_log* or *dump* or *dump_every* or *bad_dt_check* or
    *bad_dt_limit* or *sheath*

    * *mode* value = *boris* or *hybrid* or *gca*
    * *plasma* value = ID of a :doc:`fix background <fix_background>`
      or a :doc:`compute plasma/fields <compute_plasma_fields>`
    * *skip* value = mix-ID = mixture whose particles bypass the
      pusher
    * *subcycles* value = N = substeps per particle move (N > 0)
    * *gca_switch* value = F = ratio of field scale length to Larmor
      radius above which a particle is advanced as a guiding centre
      (F > 0)
    * *gca_integrator* value = *rk4* or *rk2* or *simple*
    * *boris_near* values = M [*rhoL*]

      * M = use the Boris scheme within this distance of the wall
        (m), M >= 0, 0 = off
      * *rhoL* = optional word.  M is then a multiple of the local
        Larmor radius.

    * *gc_wall* value = *a0* or *flux*
    * *switch_log* value = file = base name of the per-processor log
      files of scheme changes
    * *dump* value = *yes* or *no*
    * *dump_every* value = N = print pusher diagnostics every N
      timesteps (N > 0)
    * *bad_dt_check* value = *yes* or *no*
    * *bad_dt_limit* value = X = warn if
      :math:`|q/m|\,B\,\delta t` exceeds X (X > 0)
    * *sheath* values = smode sargs

      * smode = *off* or *kick* or *spatial* or *boundary*
      * sargs = zero or more of *geom* ID, *mD_amu* M, *dmax* D,
        *waveform* attr freq
      * *geom* ID = ID of a
        :doc:`compute nearest_surf/grid <compute_nearest_surf_grid>`,
        required unless smode is *off*
      * *mD_amu* M = mass of the background ions (amu)
      * *dmax* D = distance from the wall within which the sheath
        acts (m), D >= 0, 0 = automatic
      * *waveform* attr freq = custom per-surf attribute with the
        bias of each surface element, and its frequency (Hz)

  * *phi_track* value = *yes* or *no* = accumulate the toroidal angle
    travelled by each particle if *yes*

Examples
--------

.. parsed-literal::

   global fnum 1.0e20
   global vstream 100.0 0 0 fnum 5.0e18
   global temp 1000
   global weight cell radius
   global mem/limit 100
   global field constant 9.8 0 0 1

.. code-block:: text

   global efield particle fe 1
   global bfield particle fb 0

   global pusher mode boris plasma pd subcycles 20
   global pusher plasma pd mode gca gca_integrator rk2
   global pusher mode hybrid plasma pd skip particulates gca_switch 2.5 &
          gc_wall flux sheath boundary geom cgeom mD_amu 2.0
   global pusher mode boris plasma pd subcycles 50 sheath spatial geom cgeom
   global phi_track yes

Description
-----------

Define global properties of the system.

The *fnum* keyword sets the ratio of real, physical molecules to
simulation particles.  E.g. a value of 1.0e20 means that one particle
in the simulation represents 1.0e20 molecules of the particle species.

The *nrho* keyword sets the number density of the background gas.  For
3d simulations the units are #/volume.  For 2d, the units are
effectively #/area since the z dimension is treated as having a length
of 1.0.

Assuming your simulation is populated by particles from the background
gas, the *fnum* and *nrho* settings can determine how many particles
will be present in your simulation, when using the
:doc:`create_particles <create_particles>` or :doc:`fix emit <fix_emit_face>` command variants.

The *vstream* keyword sets the streaming velocity of the background
gas.

The *temp* keyword sets the thermal temperature of the background gas.
This is a Gaussian velocity distribution superposed on top of the
streaming velocity.

The *field* keyword adds an additional external field term which can
included in the motion of particles.  The *fstyle* argument can be
*none* or *constant*, *particle*, or *grid*.  Note that only one of
these can be set by the global command.  If the *field* keyword is
specified multiple times, only the last one has an effect.

The *none* setting turns off any external field setting previously
specified.  It is the default.

The *constant* setting is for a field that has no spatial or time
dependence; the same field vector acts on all particles.  Gravity is
an example of a constant external field.  The *mag* arguement sets the
magnitude of the field.  The (ex,ey,ez) components specify the
direction the field acts in.  The components do not need to be a unit
vector; the code converts them into a unit vector internally.  Note
that a z-component cannot be used for 2d simulations.

The *particle* setting is for a field that is computed on a per
particle basis, depending on the position or other attributes of each
particle.  A spatially- or time-dependent magnetic field, acting on
the magnetic moment of each particle, is an example of a variable
external field.  The fixID argument is the ID of a fix which computes
the components of the field vector for each particle.  These may alter
both the position and velocity of each particle when it is advected
each timestep.

The *grid* setting is for a field that is computed on a per grid cell
basis and applied to all the particles in the grid cell.  A spatially-
or time-dependent magnetic field which is coarsened to act at the
resolution of grid cells is an example of a per grid cell external
field.  The fixID argument is the ID of a fix which computes the
components of the field vector for each grid cell.  These may alter
both the position and velocity or particles in the grid cell when they
are advected each timestep.  The Nfreq argument specifies how often to
re-compute the per grid cell field vectors.  For a field that has no
time dependence you should set Nfreq to zero; the field will only be
computed once at the beginning of each simulation run.  For a field
with time-dependence you can choose how often to recompute the field,
depending on how fast it varies.

Note that there is a tradeoff between the *particle* and *grid*
options of the *field* keyword.  For the *particle* option the field
must be computed every timestep for all particles; the field values
are accurately computed at precisely each particle's position but it
is an expensive operation.  For the *grid* option the field is only
computed once at the beginning of a run or once every Nfreq
timesteps.  Even if it is computed every timestep, the number of grid
cells is typically much smaller than the number of particles.
However the accuracy of the field applied to each particle is more
approximate than for the *particle* option.  This is because the
field applied to each particle is the value it has at the center of
the particle's grid cell.

----------

The *surfs* keyword determines what kind of surface elements OpenEdge
uses and how they are distributed across processors.  Possible values
are *explicit*, *explicit/distributed*, and *implicit*.  See the
:ref:`Howto 6.13 <howto_13>` section of the manual for an
explantion of explicit versus implicit surfaces.  The distributed
option can be important for models with huge numbers of surface
elements.  Each processor stores copies of only the surfaces that
overlap grid cells it owns or has ghost copies of.  Implicit surfaces
are always distributed.  The *explicit* setting is the default and
means each processor stores a copy of all the defined surface
elements.  Note that a surface element requires about 100 bytes of
storage, so storing a million on a single processor requires about 100
MBytes.

The *surfgrid* keyword determines what algorithm is used to enumerate
the overlaps (intersections) between grid cells and surface elements
(lines in 2d, triangles in 3d).  The possible settings are *percell*,
*persurf*, and *auto*.  The *auto* setting is the default and will
choose between a *percell* or *persurf* algorithm based on the number
of surface elements and processor count.  If there are more processors
than surface elements, the *percell* algorithm is used.  Otherwise the
*persurf* algorithm is used.  The *percell* algorithm loops over the
subset of grid cells each processor owns.  All the surface elements
are tested for overlap with each owned grid cell.  The *persurf*
algorithm loops over a 1/P fraction of surface elements on each
processor.  The bounding box around each surface is used to find all
grid cells it possibly overlaps.  For large numbers of surface
elements or processors, the *persurf* algorithm is generally faster.

The *surfmax* keyword determines the maximum number of surface
elements (lines in 2d, triangles in 3d) that can overlap a single grid
cell.  The default is 100, which should be large enough for any
simulation, unless you define very coarse grid cells relative to the
size of surface elements they contain.

The *splitmax* keyword determines the maximum number of sub-cells a
single grid cell can be split into as a result of its intersection
with multiple surface elements (lines in 2d, triangles in 3d).  The
default is 10, which should be large enough for any simulation, unless
you embed a complex-shaped surface object into one or a very few grid
cells.

The *surftally* keyword determines what algorithm is used to combine
tallies of surface collisions across processors that own portions of
the same surface element.  The possible settings are *reduce*,
*rvous*, and *auto*.  The *auto* setting is the default and will
choose between a *reduce* or *rvous* algorithm based on the number of
surface elements and processor count.  If there are more processors
than surface elements, the *reduce* algorithm is used.  Otherwise the
*rvous* algorithm is used.  The *reduce* algorithm is suitable for
relatively small surface elememt counts.  It creates a copy of a
vector or array of length the global number of surface elements.  Each
processor sums its tally contributions into the vector or array.  An
MPI_Allreduce() is performed to sum it across all processors.  Each
processor than extracts values for the N/P surfaces it owns.  The
*rvous* algorithm is faster for large surface element counts.  A
rendezvous style of communication is performed where every processor
sends its tally contributions directly to the processor which owns the
element as one of its N/P elements.

----------

The *gridcut* keyword determines the cutoff distance at which ghost
grid cells will be stored by each processor.  Assuming the processor
owns a compact clump of grid cells (see below), it will also store
ghost cell information from nearby grid cells, up to this distance
away.  If the setting is -1.0 (the default) then each processor owns a
copy of ghost cells for all grid cells in the simulation.  This can
require too much memory for large models.  If the cutoff is 0.0,
processors own a minimal number of ghost cells.  This saves memory but
may require multiple passes of communication each timestep to move all
the particles and migrate them to new owning processors.  Typically a
cutoff the size of 2-3 grid cell diameters is a good compromise that
requires only modest memory to store ghost cells and allows all
particle moves to complete in only one pass of communication.

An example of the *gridcut* cutoff applied to a clumped assignment is
shown in this zoom-in of a 2d hierarchical grid with 5 levels, refined
around a tilted ellipsoidal surface object (outlined in pink).  One
processor owns the grid cells colored orange.  A bounding rectangle
around the orange cells, extended by a short cutoff distance, is drawn
as a purple rectangle.  The rectangle contains only a few ghost grid
cells owned by other processors.

.. image:: JPG/partition_zoom_cutoff.jpg
   :align: center

.. warning::

   Using the *gridcut* keyword with a cutoff >= 0.0 is
   only allowed if the grid cells owned by each processor are "clumped".
   If each processor's grid cells are "dispersed", then ghost cells
   cannot be created with a *gridcut* cutoff >= 0.0.  Whenever ghost
   cells are generated, a warning to this effect will be triggered.  At a
   later point when surfaces are read in or a simulation is performed, an
   error will result.  The solution is to use the
   :doc:`balance_grid <balance_grid>` command to change to a clumped grid
   cell assignment.  See :ref:`Section 4.10 <howto_12>` of the
   manual for an explanation of clumped and dispersed grid cell
   assignments and their relative performance trade-offs.

.. warning::

   If grid cells have already been defined via the
   :doc:`create_grid <create_grid>`, :doc:`read_grid <read_grid>`, or
   :doc:`read_restart <read_restart>` commands, when the *gridcut* cutoff
   is specified, then any ghost cell information that is currently stored
   will be erased.  As discussed in the preceeding paragraph, a
   :doc:`balance_grid <balance_grid>` command must then be invoked to
   regenerate ghost cell information.  If this is not done before
   surfaces are read in or a simulation is performed, an error will
   result.

The *comm/sort* keyword determines whether the messages a proc
receives for migrating particles (every step) and ghost grid cells (at
setup and after re-balance) are sorted by processor ID.  Doing this
requires a bit of overhead, but can make it easier to debug in
parallel, because simulations should be reproducible when run on the
same number of processors.  Without sorting, messages may arrive in a
randomized order, which means lists of particles and grid cells end up
in a different order leading to statistical differences between runs.

The *comm/style* keyword determines the style of particle
communication that is performed to migrate particles every step.  The
most efficient method is typically for each processor to exchange
messages with only the processors it has ghost cells for, which is the
method used by the *neigh* setting.  The *all* setting performs a
relatively cheap, but global communication operation to determine the
exact set of neighbors that need to be communicated with at each step.
For small processor counts there is typically little difference.  On
large processor counts the *neigh* setting can be significantly
faster.  However, if the flow is streaming in one dominant direction,
there may be no particle migration needed to upwind processors, so the
*all* method can generate smaller counts of neighboring processors.

Note that the *neigh* style only has an effect (at run time) when the
grid is decomposed by the RCB option of the :doc:`balance <fix_balance>` or
:doc:`fix balance <fix_balance>` commands.  If that is not the case,
OpenEdge performs the particle communication as if the *all* setting
were in place.

The *weight* keyword determines whether particle weighting is used.
Currently the only style allowed, as specified by wstyle = *cell*, is
per-cell weighting.  This is a mechanism for inducing every grid cell
to contain roughly the same number of particles (even if cells are of
varying size), so as to minimize the total number of particles used in
a simulation while preserving accurate time and spatial averages of
flow quantities.  The cell weights also affect how many particles per
cell are created by the :doc:`create_particles <create_particles>` and
:doc:`fix emit <fix_emit_face>` command variants.

If the mode is set to *none*, per-cell weighting is turned off if it
was previously enabled.  For mode = *volume* or *radius* or
*radius/only*, per-cell weighting is enabled, which triggers two
computations.  First, at the time this command is issued, each grid
cell is assigned a "weight" which is calculated based either on the
cell *volume* or *radius*, as specified by the *mode* setting.  For
the *volume* setting, the weight of a cell is its 3d volume for a 3d
model, and the weight is its 2d area for a 2d model.  For an
axi-symmetric model, the weight is the 3d volume of the 2d
axi-symmetric cell, i.e. the volume the area sweeps out when rotated
around the y=0 axis of symmetry.  The *radius* and *radius/only*
settings are only allowed for axisymmetric systems.  For the *radius*
option, the weight is the distance the cell midpoint is from the y=0
axis of symmetry, multiplied by the length of the cell in the x
direction.  This mode attempts to preserve a uniform number of
particles in each cell, regardless of the cell area, for a uniform
targeted density.  For the *radius/only* option, the weight is just the
distance the cell midpoint is from the y=0 axis of symmetry. This mode attempts to preserve a uniform distribution of particles
per unit area, for a uniform targeted density.  See :ref:`Section 6.2 <howto_2>` for more details on axi-symmetric
models.

Second, when a particle moves from an initial cell to a final cell,
the initial/final ratio of the two cell weights is calculated.  If the
ratio > 1, then additional particles may be created in the final cell,
by cloning the attributes of the incoming particle.  E.g. if the ratio
= 3.4, then two extra particle are created, and a 3rd is created with
probability 0.4.  If the ratio < 1, then the incoming particle may be
deleted.  E.g. if the ratio is 0.7, then the incoming particle is
deleted with probability 0.3.

Note that the first calculation of weights is performed whenever the
*global weight* command is issued.  If particles already exist, they
are not cloned or destroyed by the new weights.  The second
calculation only happens when a simulation is run.

The *particle/reorder* keyword determines how often the list of
particles on each processor is reordered to store particles in the same
grid cell contiguously in memory. This operation is performed every
*nsteps* as specified. A value of 0 means no reordering is ever done.
This option is only available when using the KOKKOS package and can
improve performance on certain hardware such as GPUs, but is typically
slower on CPUs except when running on thousands of nodes. Reordering
requires sorting the particles, which is done automatically when
collisions are enabled. If collisions are not enabled, then sorting
will also be performed in addition to reordering.

The *mem/limit* keyword limits the amount of memory allocated for
several operations: load balancing, reordering of particles, and restart
file read/write. This should only be necessary for very large
simulations where the memory footprint for particles and grid cells is a
significant fraction of available memory. In this case, these operations
can trigger a memory error due to the additional memory they require.
Setting a limit on the memory size will perform these operations more
incrementally so that memory errors do not occur.

A load-balance operation can use as much as 3x more memory than the
memory used to store particles (reported by OpenEdge when a simulation
begins). Particle reordering temporarily doubles the memory needed to
store particles because it is performed out-of-place by default. Reading
and writing restart files also requires temporary buffers to hold grid
cells and particles and can double the memory required.

Specifying the value for *mem/limit* as *grid*, will allocate extra
memory limited to the size of memory for storing grid cells on each
processor. For most simulations this is typically much smaller than the
memory used to store particles. Specifying a numeric value for *bytes*
will allocate extra memory limited to that many MBytes on each
processor. *Bytes* can be specified as a floating point value or an
integer, e.g. 0.5 if you want to use 1/2 MByte of extra memory or 100
for a 100 MByte buffer. Specifying a value of 0 (the default) means no
limit is used. The value used for *mem/limit* must not exceed 2GB or an
error will occur.

For load-balancing, the communication of grid and particle data to new
processors will then be performed in multiple passes (if necessary) so
that only a portion of grid cells and their particles which fit into the
extra memory are migrated in each pass. Similarly for particle
reordering, multiple passes are performed using the extra memory to
reorder the particles nearly in-place. For reading/writing restart
files, multiple passes are used to read from or write to the restart
file as well. For reading restart files, this option is ignored unless
reading from multiple files (i.e. a "%" character was used in the
command to write out the restart) and the number of MPI ranks is greater
than the number of files.

Note that for these operations if the extra memory is too small,
performance will suffer due to the large number of multiple passes
required.

If the *optmove* keyword is set to *yes* then an optimized move
algorithm will be used when possible. Normally, as particles advect
through the mesh each intermediate grid cell crossing must be found
since the particle may encounter a box boundary or surface element.
However, if there are no surfaces and the grid is uniform (only a
single level for all cells without further refinement) and the
*optmove* keyword is set to *yes* then the particle will be moved to
its final position in a single step, skipping all intermediate grid
cell crossings, which can improve performance. If a particle hits a
box boundary or leaves the owning proc's subdomain (including the
ghost cell region), then the normal (non-optimized) move algorithm
will be used for that specific particle on that timestep. The *optmove
yes* option cannot be used when surfaces are defined, the grid is not
uniform, or when fix adapt is enabled, otherwise an error will result.

----------

The *efield*, *bfield*, *pusher* and *phi_track* keywords belong to
OpenEdge and are described in the following sections.

External electric and magnetic fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *efield* and *bfield* keywords give the charged-particle pusher an
electric and a magnetic field that are computed by a fix.  They are
meant for verification cases with analytic fields.  A production run
takes :math:`\mathbf{E}` and :math:`\mathbf{B}` from the plasma
provider named by the *plasma* value of the *pusher* keyword, and the
pusher falls back on the *efield* and *bfield* fields only where that
provider returns no field.

The *none* style removes a field set earlier.  The *particle* style
names a :doc:`fix efield/particle <fix_efield_particle>` or
:doc:`fix bfield/particle <fix_bfield_particle>`, which evaluates
particle-style variables at each particle.  These fields are
recomputed on every timestep, whatever the value of Nfreq.  The *grid*
style names a fix that provides the field components per grid cell.
The pusher then uses the value of the particle's cell.  A per-grid
field is computed once at the start of a run if Nfreq = 0 and every
Nfreq timesteps otherwise.

The field components are in the coordinate order of the simulation
box.  Only the components that the fix defines are used.  The others
are zero.

Charged-particle pusher
^^^^^^^^^^^^^^^^^^^^^^^

The *pusher* keyword sets how charged simulation particles are
advanced in the electric and magnetic fields during the particle move.
This is the collisionless part of the impurity transport model of
OpenEdge [Diaw2025]_.  Collisions, diffusion and the thermal force are
separate fixes, for example
:doc:`fix coulomb/background <fix_coulomb_background>`,
:doc:`fix cross_field_diffusion <fix_cross_field_diffusion>` and
:doc:`fix force/thermal <fix_force_thermal>`.

All pusher sub-keywords follow the word *pusher*.  They can be split
over several *global pusher* commands, and a later setting replaces an
earlier one.  The pusher sub-keywords end at the first word that is
not one of them, which is then read as an ordinary *global* keyword.

A particle with charge :math:`q = Ze` and mass :math:`m` obeys

.. math::
   :label: pusher-lorentz

   \frac{d\mathbf{x}}{dt} = \mathbf{v} , \qquad
   \frac{d\mathbf{v}}{dt} = \frac{q}{m}
      \left( \mathbf{E} + \mathbf{v} \times \mathbf{B} \right) .

The *mode* value selects how Eq. :eq:`pusher-lorentz` is integrated:

* *boris* follows the full gyro-orbit.  Use it where the Larmor radius
  is resolved or the gyrophase matters, for example in front of a
  wall.
* *gca* advances the guiding centre instead of the particle.  It
  needs far fewer steps where the Larmor radius is much smaller than
  the scale length of the field.
* *hybrid* chooses between the two for each particle on each step.

Symbols are defined on the :ref:`notation <notation>` page.

The *plasma* value names the provider of the fields, either a
:doc:`fix background <fix_background>` or a
:doc:`compute plasma/fields <compute_plasma_fields>`.  The magnetic
field is taken from the provider at the particle position.  The
electric field is taken from a :doc:`fix background <fix_background>`
provider only.  Where the provider returns no field, the field of the
*bfield* or *efield* keyword is used if one is defined.  Without a
*plasma* value the *boris* mode uses the *efield* and *bfield* fields
alone.  The *hybrid* and *gca* modes and the sheath model require a
*plasma* value.

Neutral particles move in a straight line,
:math:`\mathbf{x} \leftarrow \mathbf{x} + \mathbf{v}\,\Delta t`.  So
do the particles of the mixture named by *skip*, whatever their
charge.  Use *skip* for particulates, whose forces come from the
particulate fixes.

Boris scheme and subcycling
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The full orbit is integrated with the scheme of Boris [Boris1970]_.
The timestep is divided into :math:`N_\mathrm{sub}` substeps of length
:math:`\delta t = \Delta t / N_\mathrm{sub}`, where
:math:`N_\mathrm{sub}` is the *subcycles* value.  One substep is a
half electric kick, a rotation about :math:`\mathbf{B}`, a second half
kick and a drift:

.. math::
   :label: pusher-boris

   \begin{aligned}
   \mathbf{v}^- &= \mathbf{v} + \frac{q}{m}\mathbf{E}\,\frac{\delta t}{2} , \\
   \mathbf{v}' &= \mathbf{v}^- + \mathbf{v}^- \times \mathbf{t} ,
      \qquad \mathbf{t} = \frac{q}{m}\mathbf{B}\,\frac{\delta t}{2} , \\
   \mathbf{v}^+ &= \mathbf{v}^- + \mathbf{v}' \times \mathbf{s} ,
      \qquad \mathbf{s} = \frac{2\,\mathbf{t}}{1 + t^2} , \\
   \mathbf{v}_\mathrm{new} &= \mathbf{v}^+
      + \frac{q}{m}\mathbf{E}\,\frac{\delta t}{2} , \\
   \mathbf{x}_\mathrm{new} &= \mathbf{x} + \mathbf{v}_\mathrm{new}\,\delta t .
   \end{aligned}

The rotation conserves the kinetic energy exactly.  The gyration is
resolved if :math:`|\Omega|\,\delta t \ll 1`.

The magnetic field is sampled once, at the position at the start of
the move, and held over the substeps.  The electric field is sampled
once in 2d and at every substep in 3d.

How the substeps are joined to the particle move depends on the
geometry:

* In 3d and in planar 2d the position advances with every substep.
  With :math:`N_\mathrm{sub} > 1` each substep segment is tested
  against the surface elements of the particle's grid cell.  If it
  crosses one, or if the particle leaves the cell, the push ends at
  that substep.  On a surface crossing the end point is placed just
  behind the intersection.  The move then traces the straight segment
  from the starting position to that end point with the usual
  cell-crossing and surface-collision logic.  The substeps that were
  not taken are dropped.  The Boris push is not called again for that
  particle on that timestep, so it advances by less than
  :math:`\Delta t`.
* In axisymmetric 2d all :math:`N_\mathrm{sub}` velocity updates of
  Eq. :eq:`pusher-boris` are done at the starting position.  The
  particle then moves along the single straight segment
  :math:`\mathbf{x} + \mathbf{v}_\mathrm{new}\,\Delta t`, and the move
  maps the end point and the velocity back into the :math:`(R,Z)`
  plane.  The straight segment must be short compared with the Larmor
  radius, so here the timestep itself has to resolve the gyration.

With *bad_dt_check yes* each processor prints one warning, the first
time a particle has :math:`|q/m|\,B\,\delta t` above the
*bad_dt_limit* value.  The step is not changed.

Guiding-centre equations
^^^^^^^^^^^^^^^^^^^^^^^^

A guiding-centre particle is described by the position
:math:`\mathbf{X}` of its guiding centre, its parallel velocity
:math:`v_\parallel` and its magnetic moment :math:`\mu`.  When a
particle is first advanced as a guiding centre, the state is built
from its position and velocity with the local field,

.. math::
   :label: pusher-gcinit

   \mathbf{X} = \mathbf{x} + \frac{\mathbf{v} \times \hat{\mathbf{b}}}{\Omega} ,
   \qquad
   v_\parallel = \mathbf{v} \cdot \hat{\mathbf{b}} ,
   \qquad
   \mu = \frac{m v_\perp^2}{2B} ,

with the signed gyrofrequency :math:`\Omega = qB/m`.  The magnetic
moment is then held constant.

With the *rk4* and *rk2* integrators the code advances the
guiding-centre equations of Littlejohn [Littlejohn1983]_,

.. math::
   :label: pusher-gc

   \begin{aligned}
   \frac{d\mathbf{X}}{dt} &= \frac{1}{B^*_\parallel}
      \left[ v_\parallel \mathbf{B}^*
             + \mathbf{E} \times \hat{\mathbf{b}}
             + \frac{\mu}{q}\, \hat{\mathbf{b}} \times \nabla B \right] , \\
   \frac{dv_\parallel}{dt} &= \frac{\mathbf{B}^*}{B^*_\parallel} \cdot
      \left[ \frac{q}{m}\mathbf{E} - \frac{\mu}{m} \nabla B \right] ,
   \end{aligned}

with

.. math::
   :label: pusher-bstar

   \mathbf{B}^* = \mathbf{B}
      + \frac{m v_\parallel}{q}\, \nabla \times \hat{\mathbf{b}} ,
   \qquad
   B^*_\parallel = \hat{\mathbf{b}} \cdot \mathbf{B}^* .

The terms of the first line of Eq. :eq:`pusher-gc` are the motion
along the field, the :math:`\mathbf{E} \times \mathbf{B}` drift and
the :math:`\nabla B` drift.  The curvature drift is contained in
:math:`v_\parallel \mathbf{B}^*` through the
:math:`\nabla \times \hat{\mathbf{b}}` term of Eq.
:eq:`pusher-bstar`.  The second line holds the acceleration by the
parallel electric field and the mirror force.

The *gca_integrator* value selects the time integration of
:math:`(\mathbf{X}, v_\parallel)`:

* *rk4* is the classical fourth-order Runge-Kutta method, with four
  evaluations of Eq. :eq:`pusher-gc` per step.
* *rk2* is the explicit midpoint method, with two evaluations.
* *simple* is a one-stage update of a reduced model without the
  :math:`\nabla \times \hat{\mathbf{b}}` term, so
  :math:`\mathbf{B}^* = \mathbf{B}` and there is no curvature drift:

  .. math::

     \begin{aligned}
     a_\parallel &= \frac{q}{m}\, \hat{\mathbf{b}} \cdot \mathbf{E}
        - \frac{\mu}{m}\, \hat{\mathbf{b}} \cdot \nabla B , \\
     \mathbf{X}_\mathrm{new} &= \mathbf{X} + \Delta t \left[
        \left( v_\parallel + a_\parallel \frac{\Delta t}{2} \right)
        \hat{\mathbf{b}}
        + \frac{\mathbf{E} \times \mathbf{B}}{B^2}
        + \frac{\mu}{qB}\, \hat{\mathbf{b}} \times \nabla B \right] , \\
     v_{\parallel,\mathrm{new}} &= v_\parallel + a_\parallel\, \Delta t .
     \end{aligned}

  Use it for scoping only.

A guiding-centre step covers the whole timestep :math:`\Delta t`.  The
*subcycles* value does not apply to it.  With *rk4* and *rk2* the
fields and their derivatives are sampled again at the position of
every stage.  If a stage position has no field, the fields of the
first stage are kept.

The derivatives :math:`\nabla B` and
:math:`\nabla \times \hat{\mathbf{b}}` come from the plasma provider.
They are available for an equilibrium or a constant magnetic field.  A
field that is defined on the plasma mesh only has no derivatives, and
both are then zero.  The same holds for the three-dimensional
*zones3d* background and for the *analytic* field of
:doc:`compute plasma/fields <compute_plasma_fields>`.  :math:`\nabla \times \hat{\mathbf{b}}` is
computed for an axisymmetric field and is zero for a field source that
is not axisymmetric.

The state :math:`(\mathbf{X}, v_\parallel, \mu)` is kept between
steps in the custom per-particle attributes *gca_x*, *gca_y*, *gca_z*,
*gca_vpar* and *gca_mu*, which can be written by the
:doc:`dump particle <dump>` command.  After a guiding-centre step the
particle position is :math:`\mathbf{X}`.  In 3d the particle velocity
is rebuilt as

.. math::

   \mathbf{v} = v_\parallel \hat{\mathbf{b}}
      + v_\perp \left( \hat{\mathbf{e}}_1 \cos\gamma
                     + \hat{\mathbf{e}}_2 \sin\gamma \right) ,
   \qquad
   v_\perp = \sqrt{\frac{2 \mu B}{m}} ,

with the field at the new position, two unit vectors
:math:`\hat{\mathbf{e}}_1`, :math:`\hat{\mathbf{e}}_2` perpendicular
to :math:`\hat{\mathbf{b}}`, and a gyrophase :math:`\gamma` assigned
from the particle ID and the timestep.  In 2d the particle velocity is
the displacement in the plane divided by :math:`\Delta t`, with a zero
third component, and the physical :math:`v_\parallel` and :math:`\mu`
are those of the attributes.  Commands that change the velocity or the
identity of a particle, such as collisions, ionization and surface
interactions, update or discard the stored state.  A discarded state
is rebuilt at the next step.

Hybrid switch
^^^^^^^^^^^^^

In *hybrid* mode a charged particle is advanced as a guiding centre on
a step if

.. math::
   :label: pusher-switch

   \rho_\mathrm{L} < \frac{L_B}{F_\mathrm{sw}} ,
   \qquad
   L_B = \frac{B}{|\nabla B|} ,

and with the Boris scheme otherwise.  Here :math:`F_\mathrm{sw}` is
the *gca_switch* value and :math:`L_B` the scale length of the field
strength.  The Larmor radius :math:`\rho_\mathrm{L}` is evaluated at
the start of the step, from :math:`\mu` if the particle has a
guiding-centre state and from its velocity otherwise.  A uniform field
has :math:`L_B \to \infty` and always satisfies Eq.
:eq:`pusher-switch`.  The test needs the field derivatives.  Where the
provider has none, a particle in *hybrid* mode stays on the Boris
scheme.

In *gca* mode every charged particle in a non-zero field is advanced
as a guiding centre.  The one exception is a particle that fails Eq.
:eq:`pusher-switch` where the field derivatives are available.  It is
advanced with the Boris scheme, which keeps a particle with a very
large Larmor radius, such as a charged dust grain, from being moved by
the transformation of Eq. :eq:`pusher-gcinit`.

When a guiding-centre particle returns to the Boris scheme, it is
placed on its gyro-orbit, at
:math:`\mathbf{x} = \mathbf{X} - \mathbf{v} \times \hat{\mathbf{b}} / \Omega`
with the rebuilt velocity :math:`\mathbf{v}`.

The *boris_near* value adds a Boris layer in front of the wall.  It
uses the nearest wall element of the particle's cell, taken from the
:doc:`compute nearest_surf/grid <compute_nearest_surf_grid>` named by
the *geom* value of the *sheath* sub-keyword.  The *geom* value can be
given with *sheath off*.  Let :math:`d` be the distance of the
particle from the plane of that element and :math:`d_\mathrm{sw}` the
layer thickness, which is M, or :math:`\mathrm{M}\,\rho_\mathrm{L}`
if the word *rhoL* follows M.  A particle is advanced with the Boris
scheme if one of the following holds:

* :math:`d < d_\mathrm{sw}` at the start of the step.
* The guiding-centre step would end with :math:`d < d_\mathrm{sw}` or
  on the other side of the plane.  The guiding-centre step is then
  discarded and the whole step is repeated with the Boris scheme.
* The particle was on the Boris scheme on the previous step and has
  not yet both gyrated through an angle of :math:`4\pi` on it and
  reached :math:`d \ge 2 d_\mathrm{sw}`.

A value of 0 turns the layer off.

The *gc_wall* value sets the velocity with which a guiding-centre
particle meets a wall.  With *a0* it is the particle velocity of the
previous section, which in 2d is the displacement velocity of the
guiding centre and carries no gyration.  With *flux* the velocity is
rebuilt from :math:`v_\parallel` and :math:`\mu` with a gyrophase
drawn with a probability
proportional to the velocity component towards the wall,
:math:`\max(-\mathbf{v}(\gamma) \cdot \hat{\mathbf{n}}, 0)`, where
:math:`\hat{\mathbf{n}}` is the wall normal.  This weights the phases
as the first passage of a resolved orbit through the wall plane does,
and it is needed to reproduce the impact energy and angle
distributions of the full orbit.  In 3d the phase is drawn this way
when the guiding centre at the end of the step is closer to the plane
of the nearest wall element than
:math:`2\rho_\mathrm{L} + |v_\parallel|\,\Delta t`.  In axisymmetric
2d it is drawn at the moment of the surface collision.  The option has
no effect in planar 2d.  It needs the *geom* value.

The *switch_log* value turns on a log of the changes between the two
schemes.  Each processor writes a CSV file named file.rankN, where N
is the processor number, with the columns timestep, particle ID, old
and new mode (0 = Boris, 1 = guiding centre), reason, distance at the
start and the end of the step, :math:`d_\mathrm{sw}`, kinetic energy
before and after the change, and a flag for a repeated step.

Sheath potential
^^^^^^^^^^^^^^^^

The *sheath* value adds a model of the Debye sheath and the magnetic
presheath in front of the wall, which accelerate ions onto the
surface.  The model is the analytic
potential profile of Borodkina et al., Contrib. Plasma Phys. 56, 640
(2016), with the Debye-sheath solution of Borodkina et al., Russian
Physics Journal 58, 438 (2015).

The potential is a function of the distance :math:`d` from the wall.
:math:`\Phi(d) \ge 0` denotes the potential drop between the plasma
far from the wall and the point at distance :math:`d`, so a positive
ion that reaches :math:`d` has gained the energy :math:`Ze\,\Phi(d)`.
The profile is built from the plasma parameters at the wall element
and the following quantities:

.. math::
   :label: sheath-scales

   c_s = \sqrt{\frac{T_e + T_i}{m_i}} , \qquad
   \rho_s = \frac{c_s\, m_i}{e B} , \qquad
   L_\mathrm{MPS} = 2 \rho_s \sin\alpha , \qquad
   \cos\alpha = |\hat{\mathbf{b}} \cdot \hat{\mathbf{n}}| .

Here :math:`m_i` is the mass of the background ions, set by *mD_amu*,
:math:`\alpha` is the angle between the magnetic field and the wall
normal :math:`\hat{\mathbf{n}}`, and :math:`L_\mathrm{MPS}` is the
width of the magnetic presheath.  The total drop is that of a floating
wall,

.. math::
   :label: sheath-wall

   \frac{e \Phi_\mathrm{w}}{T_e} = \eta_\mathrm{w}
      = \frac{1}{2} \ln \left[ \frac{m_i}{2\pi m_e}\,
        \frac{1}{1 + T_i/T_e} \right] ,

and is set to zero if this expression is negative.  The critical angle
:math:`\alpha^*` is defined by
:math:`\cos\alpha^* = \exp(-\eta_\mathrm{w})`.

For :math:`\alpha \ge \alpha^*`, which is the case at the grazing
field angles of tokamak targets, there is no Debye sheath and the
whole drop occurs in the magnetic presheath:

.. math::
   :label: sheath-phi

   \Phi(d) = \Phi_\mathrm{w} \exp\left( -\frac{2d}{L_\mathrm{MPS}} \right) .

For :math:`\alpha < \alpha^*` the magnetic presheath takes the drop
:math:`\eta_\mathrm{MPS} = -\ln\cos\alpha` and the Debye sheath the
rest.  With :math:`\xi = d/\lambda_\mathrm{D}` and
:math:`\eta = e\Phi/T_e`,

.. math::
   :label: sheath-phi-ds

   \eta(\xi) =
   \begin{cases}
   \eta_\mathrm{w} - Q \left( 1 - e^{-a \xi} \right) ,
      & \xi \le \xi_\mathrm{i} , \\[4pt]
   \eta_\mathrm{MPS} \exp\left[
      -\dfrac{2 (\xi - \xi_\mathrm{i})\, \lambda_\mathrm{D}}{L_\mathrm{MPS}}
      \right] , & \xi > \xi_\mathrm{i} .
   \end{cases}

The constants :math:`a` and :math:`Q` and the position
:math:`\xi_\mathrm{i}` of the interface follow from the slope of the
potential at the interface and at the wall,

.. math::
   :label: sheath-closure

   \begin{aligned}
   s &= \frac{2\, \eta_\mathrm{MPS}\, \lambda_\mathrm{D}}{L_\mathrm{MPS}} , \\
   g_\mathrm{w}^2 &= 2 e^{-\eta_\mathrm{w}}
      + 4 \cos\alpha \sqrt{1 + \frac{2 (\eta_\mathrm{w} - \eta_\mathrm{MPS})}
                                   {1 + T_i/T_e}}
      + s^2 - 6 \cos\alpha , \\
   a &= \frac{g_\mathrm{w} - s}{\eta_\mathrm{w} - \eta_\mathrm{MPS}} ,
   \qquad Q = \frac{g_\mathrm{w}}{a} ,
   \qquad \xi_\mathrm{i} = -\frac{1}{a}
      \ln\left( 1 - \frac{\eta_\mathrm{w} - \eta_\mathrm{MPS}}{Q} \right) .
   \end{aligned}

The potential and the electric field are continuous at the interface.
Eq. :eq:`sheath-closure` is written for a general ratio
:math:`T_i/T_e`.

The sheath acts within the distance :math:`d_\mathrm{max}` of the
wall.  For :math:`\alpha \ge \alpha^*` the automatic value is
:math:`10 \rho_s \sin\alpha`.  For :math:`\alpha < \alpha^*` it is
:math:`\xi_\mathrm{i} \lambda_\mathrm{D} + 10 \rho_s \sin\alpha`, and
at least :math:`20 \lambda_\mathrm{D}`.  A positive *dmax* value
replaces the automatic value, and *dmax* 0 restores it.

The plasma parameters :math:`T_e`, :math:`T_i`, :math:`n_e` and
:math:`\mathbf{B}` come from the *plasma* provider.  With a
:doc:`fix background <fix_background>` they are evaluated once per
wall element, at its midpoint in 2d and its centroid in 3d, and
reused for all particles.  If the provider has no plasma at that
point, points 0.25, 1 and 3 mm from the wall along its normal are
tried.  With a :doc:`compute plasma/fields <compute_plasma_fields>`
they are the values of the particle's cell.  An element has no sheath
where :math:`T_e`, :math:`n_e` or :math:`B` is not positive.

The wall geometry comes from the
:doc:`compute nearest_surf/grid <compute_nearest_surf_grid>` named by
*geom*.  Only the surface elements of the surface group of that
compute carry a sheath.  The wall element of a particle is the
nearest element of its grid cell.  If the cell contains elements of
the group, it is the one of them nearest to the particle.  The
distance :math:`d` is measured along the normal of that element.

Sheath modes
^^^^^^^^^^^^

The first value after *sheath* selects how the potential acts on the
particles.  The charge enters through :math:`|Z|` in all modes.  With
*off* there is no sheath.

With *kick* the sheath has no thickness.  A particle that hits a wall
element receives the full drop in its velocity component
:math:`v_n` along the wall normal, before the surface collision and
reaction models are applied:

.. math::
   :label: sheath-kick

   v_n' = \sqrt{v_n^2 + \frac{2 |Z| e\, \Phi_\mathrm{w}}{m}} ,

directed into the wall.  :math:`\Phi_\mathrm{w}` is Eq.
:eq:`sheath-wall` with the plasma at the point of impact.  The orbit
in front of the wall is not changed, so this mode does not depend on
the substep.  It suits diagnostics of the impact energy and angle.

With *spatial* the potential acts over its width.  After each Boris
substep the kinetic energy of the motion along the wall normal changes
by the exact work of the potential over the substep,

.. math::
   :label: sheath-spatial

   \frac{m}{2} \left( v_n'^2 - v_n^2 \right)
      = |Z| e \left[ \Phi(d_\mathrm{new}) - \Phi(d_\mathrm{old}) \right] ,

where :math:`d_\mathrm{old}` and :math:`d_\mathrm{new}` are the
distances before and after the substep and
:math:`\Phi(d < 0) = \Phi(0)`.  The update is skipped while both
distances exceed :math:`d_\mathrm{max}`.  The sign of :math:`v_n` is
kept.  If an ion that moves away from the wall has too little
:math:`v_n` to climb the potential over the substep, it is turned
around inside the substep and falls back towards the wall.  The net
energy that a particle has received is kept in the custom per-particle
attribute *sheath_bank*.  It cannot exceed
:math:`|Z| e\, \Phi_\mathrm{w}`.  The potential at the end of a move
is kept in the attribute *sheath_phiprev* and is the reference for the
first substep of the next move, so a change of wall element between
two moves is counted as work.  Because Eq. :eq:`sheath-spatial` uses
the potential at the two end points of the substep, the energy gained
over several substeps is the potential difference between the first
and the last point, whatever the substep.  The substep sets how well
the orbit inside the sheath is resolved.

With *boundary* the sheath has no thickness for incoming particles
and is a potential barrier for outgoing ones.  A particle that hits a
wall element receives the kick of Eq. :eq:`sheath-kick`.  At the start
of each push, a particle on the plasma side within
:math:`d_\mathrm{max}` of the wall that moves away from it,
:math:`v_n > 0`, is treated as follows:

* If :math:`m v_n^2/2 < |Z| e\, \Phi_\mathrm{w}`, the particle cannot
  leave the sheath.  Its normal velocity is reversed,
  :math:`v_n' = -v_n`, and it returns to the wall.  This is prompt
  redeposition.
* Otherwise the particle leaves and is slowed down,
  :math:`v_n' = \sqrt{v_n^2 - 2 |Z| e\, \Phi_\mathrm{w} / m}`.  This
  is done once per passage through the layer.  The custom
  per-particle attribute *sheath_paid* records it and is cleared when
  the particle is farther than :math:`d_\mathrm{max}` from the wall.

The *waveform* value adds the bias of a driven surface to the drop in
*boundary* mode.  Its first argument names a custom per-surf
attribute (see the :doc:`custom <custom>` command) of floating-point
type with 3 values per element, :math:`(V_\mathrm{dc}, V_\mathrm{rf}, \varphi_0)`
in V, V and rad.  Its second argument is the frequency :math:`f` in
Hz.  The wall voltage relative to the plasma is

.. math::

   V(t) = V_\mathrm{dc} + V_\mathrm{rf} \sin(2\pi f t + \varphi_0) ,

and the drop used for the kick and the barrier of that element is
:math:`\Phi_\mathrm{w} + \max(-V(t), 0)`.  A positive wall voltage
does not reduce the drop.

In the *spatial* and *boundary* modes the sheath acts on the Boris
steps only.  A particle on a guiding-centre step does not feel the
potential in front of the wall.  It receives the wall kick in the
*kick* and *boundary* modes.  Use *boris_near* in *hybrid* mode to
put particles on the Boris scheme before they enter the sheath.

While a sheath mode is on, the electron density that is stored for
each particle and used by the volume chemistry is reduced inside the
sheath by the Boltzmann factor,
:math:`n_e \exp(-e\Phi(d)/T_e)`.

Pusher diagnostics
^^^^^^^^^^^^^^^^^^

With *dump yes* the pusher prints diagnostics on every timestep that
is a multiple of the *dump_every* value.  Each processor prints one
line per Boris substep for the first particle in its list, with the
fields the particle sees.  In the *spatial* and *boundary* sheath
modes a summary line follows with the number of particles near the
wall, the number of sheath updates or of reflected and escaped
particles, and in *spatial* mode the mean and the largest equivalent
sheath field.  The sheath parameters of the first few wall elements
are printed when they are first evaluated.

Toroidal angle tracking
^^^^^^^^^^^^^^^^^^^^^^^

With *phi_track yes* the toroidal angle travelled by each particle in
an axisymmetric run is accumulated in a custom per-particle attribute
named *phi_unwrap* (rad).  An axisymmetric move displaces a particle
in three dimensions and then rotates it back into the :math:`(R,Z)`
plane.  The angle of each such rotation is added to the attribute, so
the angle is not wrapped to an interval of :math:`2\pi`.  Together
with :math:`R` and :math:`Z` this gives the full three-dimensional
trajectory, for example for a synthetic camera view, without a 3d
grid.  A particle on a guiding-centre step is moved within the plane
and adds nothing.

Restrictions
------------

The global surfmax command must be used before surface elements are
defined, e.g. via the :doc:`read_surf <read_surf>` command.

The *efield*, *bfield*, *pusher* and *phi_track* keywords are part of
the OPENEDGE package.

The *hybrid* and *gca* pusher modes and every sheath mode other than
*off* require the *plasma* value.  Every sheath mode other than *off*
requires the *geom* value, which must name a
:doc:`compute nearest_surf/grid <compute_nearest_surf_grid>`.  The
*waveform* value can only be used with *sheath boundary*.

With the KOKKOS package the *boris*, *hybrid* and *gca* modes and the
*kick*, *spatial* and *boundary* sheath modes run on the device in
2d, axisymmetric and 3d geometry.  Three limits apply there.  The
*plasma* provider must be a :doc:`fix background <fix_background>`
or the Kokkos variant of
:doc:`compute plasma/fields <compute_plasma_fields>`.  A
:doc:`fix background <fix_background>` with *format zones3d* is not
supported.  *sheath kick* reads :math:`T_e` and :math:`T_i` from the
per-particle plasma cache on the device, and the run stops with an
error if the cache is not available.

Related commands
----------------

:doc:`mixture <mixture>`,
:doc:`fix background <fix_background>`,
:doc:`compute plasma/fields <compute_plasma_fields>`,
:doc:`compute nearest_surf/grid <compute_nearest_surf_grid>`,
:doc:`fix efield/particle <fix_efield_particle>`,
:doc:`fix bfield/particle <fix_bfield_particle>`

Default
-------

The keyword defaults are fnum = 1.0, nrho = 1.0, vstream = 0.0 0.0
0.0, temp = 273.15, field = none, efield = none, bfield = none,
surfs = explicit, surfgrid = auto,
surfmax = 100, splitmax = 10, surftally = auto,
gridcut = -1.0, comm/sort = no, comm/style = neigh, weight = cell
none, particle/reorder = 0, mem/limit = 0, optmove = no, phi_track =
no.

The pusher defaults are mode = boris, no plasma provider, no skip
mixture, subcycles = 1, gca_switch = 2.5, gca_integrator = rk4,
boris_near = 0, gc_wall = a0, no switch_log, dump = no, dump_every =
1, bad_dt_check = yes, bad_dt_limit = 0.1, sheath = off, mD_amu =
2.01410177811, dmax = 0 (automatic), no waveform.
