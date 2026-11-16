.. hand-edited: this page is maintained here, not generated from doc/*.txt

3. Commands
===========

This chapter describes the format of an OpenEdge input script and
lists the commands that define a simulation.

| 3.1 :ref:`OpenEdge input script <cmd_1>`
| 3.2 :ref:`Parsing rules <cmd_2>`
| 3.3 :ref:`Input script structure <cmd_3>`
| 3.4 :ref:`Commands listed by category <cmd_4>`
| 3.5 :ref:`Commands listed alphabetically <cmd_5>`

----------

.. _cmd_1:

3.1 OpenEdge input script
-------------------------

OpenEdge executes by reading commands from an input script (text file),
one line at a time.  When the input script ends, OpenEdge exits.  Each
command causes OpenEdge to take some action.  It may set an internal
variable, read in a file, or run a simulation.  Most commands have
default settings, which means you only need to use the command if you
wish to change the default.

In many cases, the ordering of commands in an input script is not
important.  However the following rules apply:

(1) OpenEdge does not read your entire input script and then perform a
simulation with all the settings.  Rather, the input script is read
one line at a time and each command takes effect when it is read.
Thus this sequence of commands:

.. parsed-literal::

   timestep 0.5
   run      100
   run      100

does something different than this sequence:

.. parsed-literal::

   run      100
   timestep 0.5
   run      100

In the first case, the specified timestep (0.5 secs) is used for two
simulations of 100 timesteps each.  In the 2nd case, the default
timestep (1.0 sec) is used for the 1st 100 step simulation and a 0.5
sec timestep is used for the 2nd one.

(2) Some commands are only valid when they follow other commands.  For
example you cannot define the grid overlaying the simulation box until
the box itself has been defined.  Likewise you cannot read in
triangulated surfaces until a grid has been defined to store them.

Many input script errors are detected by OpenEdge and an ERROR or
WARNING message is printed.  :doc:`Section 6 <Section_errors>` gives
more information on what errors mean.  The documentation for each
command lists restrictions on how the command can be used.

----------

.. _cmd_2:

3.2 Parsing rules
-----------------

Each non-blank line in the input script is treated as a command.
OpenEdge commands are case sensitive.  Command names are lower-case, as
are specified command arguments.  Upper case letters may be used in
file names or user-chosen ID strings.

Here is how each line in the input script is parsed by OpenEdge:

(1) If the last printable character on the line is a "&" character
(with no surrounding quotes), the command is assumed to continue on
the next line.  The next line is concatenated to the previous line by
removing the "&" character and newline.  This allows long commands to
be continued across two or more lines.

(2) All characters from the first "#" character onward are treated as
comment and discarded.  See an exception in (6).  Note that a
comment after a trailing "&" character will prevent the command from
continuing on the next line.  Also note that for multi-line commands a
single leading "#" will comment out the entire command.

(3) The line is searched repeatedly for $ characters, which indicate
variables that are replaced with a text string.  See an exception in
(6).

If the $ is followed by curly brackets, then the variable name is the
text inside the curly brackets.  If no curly brackets follow the $,
then the variable name is the single character immediately following
the $.  Thus ${myTemp} and $x refer to variable names "myTemp" and
"x".

How the variable is converted to a text string depends on what style
of variable it is; see the :doc:`variable <variable>` doc page for details.
It can be a variable that stores multiple text strings, and return one
of them.  The returned text string can be multiple "words" (space
separated) which will then be interpreted as multiple arguments in the
input command.  The variable can also store a numeric formula which
will be evaluated and its numeric result returned as a string.

As a special case, if the $ is followed by parenthesis, then the text
inside the parenthesis is treated as an "immediate" variable and
evaluated as an :doc:`equal-style variable <variable>`.  This is a way
to use numeric formulas in an input script without having to assign
them to variable names.  For example, these 3 input script lines:

.. parsed-literal::

   variable X equal (xlo+xhi)/2+sqrt(v_area)
   region 1 block $X 2 INF INF EDGE EDGE
   variable X delete

can be replaced by

.. parsed-literal::

   region 1 block $((xlo+xhi)/2+sqrt(v_area)) 2 INF INF EDGE EDGE

so that you do not have to define (or discard) a temporary variable X.

Note that neither the curly-bracket or immediate form of variables can
contain nested $ characters for other variables to substitute for.
Thus you cannot do this:

.. parsed-literal::

   variable        a equal 2
   variable        b2 equal 4
   print           "B2 = ${b$a}"

Nor can you specify this $($x-1.0) for an immediate variable, but
you could use $(v_x-1.0), since the latter is valid syntax for an
:doc:`equal-style variable <variable>`.

See the :doc:`variable <variable>` command for more details of how
strings are assigned to variables and evaluated, and how they can be
used in input script commands.

(4) The line is broken into "words" separated by whitespace (tabs,
spaces).  Note that words can thus contain letters, digits,
underscores, or punctuation characters.

(5) The first word is the command name.  All successive words in the
line are arguments.

(6) If you want text with spaces to be treated as a single argument,
it can be enclosed in either double or single quotes.  A long single
argument enclosed in quotes can even span multiple lines if the "&"
character is used, as described above.  E.g.

.. parsed-literal::

   print "Volume = $v"
   print 'Volume = $v'
   variable a string "red green blue &
                      purple orange cyan"
   if "${steps} > 1000" then quit

The quotes are removed when the single argument is stored internally.

See the :doc:`dump modify format <dump_modify>` or :doc:`print <print>` or
:doc:`if <if>` commands for examples.  A "#" or "$" character that is
between quotes will not be treated as a comment indicator in (2) or
substituted for as a variable in (3).

.. warning::

   If the argument is itself a command that requires a
   quoted argument (e.g. using a :doc:`print <print>` command as part of an
   :doc:`if <if>` or :doc:`run every <run>` command), then the double and
   single quotes can be nested in the usual manner.  See the doc pages
   for those commands for examples.  Only one level of nesting is
   allowed, but that should be sufficient for most use cases.

----------

.. _cmd_3:

3.3 Input script structure
--------------------------

An input script for a device simulation has five parts:

#. Initialization
#. Problem definition
#. Plasma background
#. Physics settings and output
#. Run

Parts 4 and 5 can be repeated.  Most commands have defaults and are
needed only to change them.  The examples directory contains complete
scripts, which are described in :doc:`Section 5 <Section_example>`.

(1) Initialization

Set the parameters that must be known before the simulation box
exists: :doc:`dimension <dimension>`, :doc:`units <units>`,
:doc:`seed <seed>` and :doc:`boundary <boundary>`.  OpenEdge runs in
SI units, which is the default.  The choice of dimension and boundary
fixes how the box axes map onto the cylindrical coordinates of the
device; see :ref:`Section 4.1 <howto_1>` and
:ref:`Section 4.2 <howto_2>`.

(2) Problem definition

These commands are given in this order:

* :doc:`create_box <create_box>` for the simulation box
* :doc:`create_grid <create_grid>` or :doc:`read_grid <read_grid>` for
  the grid cells
* :doc:`balance_grid <balance_grid>` to distribute the cells over the
  processors
* :doc:`species <species>` and :doc:`mixture <mixture>` for the
  particle species
* :doc:`material <material>` for particulate materials
* :doc:`read_surf <read_surf>` for the wall and other surfaces
* :doc:`surf_collide <surf_collide>` and
  :doc:`surf_react <surf_react>` for the wall models, assigned to the
  surfaces with :doc:`surf_modify <surf_modify>`
* :doc:`read_particles <read_particles>` or
  :doc:`create_particles <create_particles>` for an initial population

The box and the grid are required.  Surfaces are optional, but a
device simulation has at least the wall.  An initial population is
rarely used, because particles are usually injected during the run by
a wall source or an injector.

(3) Plasma background

:doc:`fix background <fix_background>` reads the plasma file or
defines a constant plasma.  Every other OpenEdge command obtains the
plasma quantities and the magnetic field from this fix and refers to
it by its ID, so it is defined before them.
:doc:`compute plasma/fields <compute_plasma_fields>` samples the
background on the grid for output.  See
:ref:`Section 4.3 <howto_3>`.

(4) Physics settings and output

The particle pusher is selected with the *pusher* keyword of the
:doc:`global <global>` command.  The other processes are fixes and
computes:

* :doc:`global pusher <global>` for the Boris, guiding-centre or
  hybrid pusher and the sheath model
* :doc:`fix coulomb/background <fix_coulomb_background>`,
  :doc:`fix force/thermal <fix_force_thermal>` and
  :doc:`fix cross_field_diffusion <fix_cross_field_diffusion>` for
  collisions, thermal forces and anomalous transport of ions
* :doc:`fix store/force <fix_store_force>` to store the contribution
  of one of these processes as a per-particle vector for output
* :doc:`fix volume/chem/adas <fix_volume_chem_adas>` for ionization,
  recombination and charge exchange
* :doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`
  for the flux of the background plasma to the wall and the
  sputtering it causes
* :doc:`fix surface/emit/recycle <fix_surface_emit_recycle>`,
  :doc:`fix surface/emit/source <fix_surface_emit_source>` and
  :doc:`fix surface/emit/puff <fix_surface_emit_puff>` for wall and
  gas sources
* :doc:`fix particulate/emit <fix_particulate_emit>`,
  :doc:`fix particulate/charge <fix_particulate_charge>`,
  :doc:`fix particulate/drag <fix_particulate_drag>` and
  :doc:`fix particulate/thermal <fix_particulate_thermal>` for dust
  grains and droplets
* :doc:`fix balance <fix_balance>` for load balancing during the run
* :doc:`compute <compute>` for diagnostics
* :doc:`stats <stats>` and :doc:`stats_style <stats_style>` for the
  statistics printed to the screen and the log file
* :doc:`dump <dump>` for snapshots of particle, grid and surface data

(5) Run

:doc:`timestep <timestep>` sets the timestep and :doc:`run <run>`
advances the simulation.

The script below has this structure.  It follows tungsten sputtered
from the wall of an axisymmetric device by the oxygen ions of the
background plasma.

.. code-block:: text

   dimension     2
   boundary      o ao p
   seed          12345
   create_box    -1.0 0.85 0.0 3.25 -0.05 0.05
   create_grid   200 150 1
   balance_grid  rcb cell

   species       input/atoms.species W W+ W2+
   mixture       source W
   read_surf     input/wall.surf group wall
   surf_collide  wcoll diffuse 773.15 1.0
   surf_react    wpwi surface/pwi input/wall.recycle
   surf_modify   wall collide wcoll react wpwi

   fix           pd background file input/plasma.h5 static yes
   compute       cgeom nearest_surf/grid all wall dist surfid nx ny nz
   global        pusher plasma pd mode boris subcycles 2 &
                 sheath spatial geom cgeom mD_amu 2.0

   fix           fcoul coulomb/background 1 background pd 2.0 1.0
   fix           fth force/thermal 1 background pd ion_thermal yes elec_thermal yes
   fix           fcd cross_field_diffusion 1 background pd D_perp 0.3
   fix           fchem volume/chem/adas 1 W auto
   compute       csput surface/physical/sputter wall background pd &
                 target W projectiles O mass_amu 2.014 static yes erosion_flux
   fix           fsrc surface/emit/source source wall csput perspecies no &
                 nlaunch_total 6 model thompson 8.68 emax 80 nevery 1
   fix           fbal balance 200 1.1 rcb part

   timestep      2e-8
   stats         1000
   dump          d1 particle all 1000 output/particles.dump id type x y z
   run           100000

----------

.. _cmd_4:

3.4 Commands listed by category
-------------------------------

This section lists OpenEdge commands, grouped by category.  The
:ref:`next section <cmd_5>` lists all commands alphabetically.

Initialization:

:doc:`dimension <dimension>`, :doc:`package <package>`, :doc:`seed <seed>`,
:doc:`suffix <suffix>`, :doc:`units <units>`

Problem definition:

:doc:`boundary <boundary>`, :doc:`bound_modify <bound_modify>`,
:doc:`create_box <create_box>`, :doc:`create_grid <create_grid>`,
:doc:`create_particles <create_particles>`, :doc:`material <material>`,
:doc:`mixture <mixture>`, :doc:`read_grid <read_grid>`,
:doc:`read_isurf <read_isurf>`, :doc:`read_particles <read_particles>`,
:doc:`read_surf <read_surf>`, :doc:`read_restart <read_restart>`,
:doc:`species <species>`

Plasma background:

:doc:`fix background <fix_background>`, :doc:`compute plasma/fields <compute_plasma_fields>`,
:doc:`fix efield/particle <fix_efield_particle>`, :doc:`fix bfield/particle <fix_bfield_particle>`

Transport:

:doc:`global pusher <global>`, :doc:`fix coulomb/background <fix_coulomb_background>`,
:doc:`fix coulomb/binary <fix_coulomb_binary>`, :doc:`fix cross_field_diffusion <fix_cross_field_diffusion>`,
:doc:`fix force/thermal <fix_force_thermal>`, :doc:`fix core/transport <fix_core_transport>`,
:doc:`fix neutral/collide/background <fix_neutral_collide_background>`,
:doc:`fix reflect/psi <fix_reflect_psi>`, :doc:`fix particle/weight <fix_particle_weight>`,
:doc:`fix dt/monitor <fix_dt_monitor>`,
:doc:`fix particle/tag <fix_particle_tag>`, :doc:`fix store/force <fix_store_force>`

Atomic processes:

:doc:`fix volume/chem/adas <fix_volume_chem_adas>`,
:doc:`compute volume/emissivity/grid <compute_volume_emissivity_grid>`

Plasma-wall interaction:

:doc:`compute surface/physical/sputter <compute_surface_physical_sputter>`,
:doc:`surf_react surface/pwi <surf_react_surface_pwi>`,
:doc:`fix surface/emit/recycle <fix_surface_emit_recycle>`,
:doc:`fix surface/emit/source <fix_surface_emit_source>`,
:doc:`fix surface/emit/puff <fix_surface_emit_puff>`,
:doc:`fix surface/state/lm <fix_surface_state_lm>`,
:doc:`compute surface/chemical/evaporation <compute_surface_chemical_evaporation>`,
:doc:`compute surface/chemical/adatom <compute_surface_chemical_adatom>`,
:doc:`compute nearest_surf/grid <compute_nearest_surf_grid>`

Particulates:

:doc:`material <material>`, :doc:`fix particulate/emit <fix_particulate_emit>`,
:doc:`fix particulate/charge <fix_particulate_charge>`,
:doc:`fix particulate/drag <fix_particulate_drag>`,
:doc:`fix particulate/thermal <fix_particulate_thermal>`,
:doc:`surf_collide particulate/bounce <surf_collide_particulate_bounce>`

Settings:

:doc:`collide <collide>`, :doc:`collide_modify <collide_modify>`,
:doc:`compute <compute>`, :doc:`fix <fix>`, :doc:`global <global>`,
:doc:`react <react>`, :doc:`react_modify <react_modify>`,
:doc:`region <region>`, :doc:`surf_collide <surf_collide>`,
:doc:`surf_modify <surf_modify>`, :doc:`surf_react <surf_react>`,
:doc:`timestep <timestep>`, :doc:`uncompute <uncompute>`,
:doc:`unfix <unfix>`

Output:

:doc:`dump <dump>`, :doc:`dump_image <dump_image>`,
:doc:`dump_modify <dump_modify>`, :doc:`restart <restart>`,
:doc:`stats <stats>`, :doc:`stats_modify <stats_modify>`,
:doc:`stats_style <stats_style>`, :doc:`undump <undump>`,
:doc:`write_grid <write_grid>`, :doc:`write_isurf <write_isurf>`,
:doc:`write_surf <write_surf>`, :doc:`write_restart <write_restart>`

Actions:

:doc:`adapt_grid <adapt_grid>`, :doc:`balance_grid <balance_grid>`,
:doc:`run <run>`, :doc:`scale_particles <scale_particles>`

Miscellaneous:

:doc:`clear <clear>`, :doc:`echo <echo>`, :doc:`if <if>`,
:doc:`include <include>`, :doc:`jump <jump>`, :doc:`label <label>`,
:doc:`log <log>`, :doc:`next <next>`, :doc:`partition <partition>`,
:doc:`print <print>`, :doc:`quit <quit>`, :doc:`shell <shell>`,
:doc:`variable <variable>`

----------

.. _cmd_5:

.. _comm:

3.5 Individual commands
-----------------------

This section lists all OpenEdge commands alphabetically, with a
separate listing below of styles within certain commands.  The
:ref:`previous section <cmd_4>` lists many of the same commands,
grouped by category.

.. list-table::

   * - :doc:`adapt_grid <adapt_grid>`
     - :doc:`balance_grid <balance_grid>`
     - :doc:`boundary <boundary>`
     - :doc:`bound_modify <bound_modify>`
     - :doc:`clear <clear>`
     - :doc:`collide <collide>`
   * - :doc:`collide_modify <collide_modify>`
     - :doc:`compute <compute>`
     - :doc:`create_box <create_box>`
     - :doc:`create_grid <create_grid>`
     - :doc:`create_isurf <create_isurf>`
     - :doc:`create_particles <create_particles>`
   * - :doc:`custom <custom>`
     - :doc:`dimension <dimension>`
     - :doc:`dump <dump>`
     - :doc:`dump image <dump_image>`
     - :doc:`dump_modify <dump_modify>`
     - :doc:`dump movie <dump_image>`
   * - :doc:`echo <echo>`
     - :doc:`fix <fix>`
     - :doc:`global <global>`
     - :doc:`group <group>`
     - :doc:`if <if>`
     - :doc:`include <include>`
   * - :doc:`jump <jump>`
     - :doc:`label <label>`
     - :doc:`log <log>`
     - :doc:`material <material>`
     - :doc:`mixture <mixture>`
     - :doc:`move_surf <move_surf>`
   * - :doc:`next <next>`
     - :doc:`package <package>`
     - :doc:`partition <partition>`
     - :doc:`print <print>`
     - :doc:`quit <quit>`
     - :doc:`react <react>`
   * - :doc:`react_modify <react_modify>`
     - :doc:`read_grid <read_grid>`
     - :doc:`read_isurf <read_isurf>`
     - :doc:`read_particles <read_particles>`
     - :doc:`read_restart <read_restart>`
     - :doc:`read_surf <read_surf>`
   * - :doc:`region <region>`
     - :doc:`remove_surf <remove_surf>`
     - :doc:`reset_timestep <reset_timestep>`
     - :doc:`restart <restart>`
     - :doc:`run <run>`
     - :doc:`scale_particles <scale_particles>`
   * - :doc:`seed <seed>`
     - :doc:`shell <shell>`
     - :doc:`species <species>`
     - :doc:`species_modify <species_modify>`
     - :doc:`stats <stats>`
     - :doc:`stats_modify <stats_modify>`
   * - :doc:`stats_style <stats_style>`
     - :doc:`suffix <suffix>`
     - :doc:`surf_collide <surf_collide>`
     - :doc:`surf_modify <surf_modify>`
     - :doc:`surf_react <surf_react>`
     - :doc:`timestep <timestep>`
   * - :doc:`uncompute <uncompute>`
     - :doc:`undump <undump>`
     - :doc:`unfix <unfix>`
     - :doc:`units <units>`
     - :doc:`variable <variable>`
     - :doc:`write_grid <write_grid>`
   * - :doc:`write_isurf <write_isurf>`
     - :doc:`write_restart <write_restart>`
     - :doc:`write_surf <write_surf>`
     - 
     - 
     - 

----------

Fix styles
----------

See the :doc:`fix <fix>` command for one-line descriptions of each style
or click on the style itself for a full description.  Styles with a
Kokkos version are marked (k).

.. list-table::

   * - :doc:`ablate <fix_ablate>`
     - :doc:`adapt (k) <fix_adapt>`
     - :doc:`ambipolar (k) <fix_ambipolar>`
     - :doc:`ave/grid (k) <fix_ave_grid>`
     - :doc:`ave/histo (k) <fix_ave_histo>`
   * - :doc:`ave/histo/weight (k) <fix_ave_histo>`
     - :doc:`ave/surf <fix_ave_surf>`
     - :doc:`ave/time <fix_ave_time>`
     - :doc:`background <fix_background>`
     - :doc:`balance (k) <fix_balance>`
   * - :doc:`bfield/particle <fix_bfield_particle>`
     - :doc:`core/transport <fix_core_transport>`
     - :doc:`coulomb/background (k) <fix_coulomb_background>`
     - :doc:`coulomb/binary <fix_coulomb_binary>`
     - :doc:`cross_field_diffusion (k) <fix_cross_field_diffusion>`
   * - :doc:`dt/monitor <fix_dt_monitor>`
     - :doc:`dt/reset (k) <fix_dt_reset>`
     - :doc:`efield/particle <fix_efield_particle>`
     - :doc:`emit/face (k) <fix_emit_face>`
     - :doc:`emit/surf <fix_emit_surf>`
   * - :doc:`force/thermal (k) <fix_force_thermal>`
     - :doc:`grid/check (k) <fix_grid_check>`
     - :doc:`halt <fix_halt>`
     - :doc:`move/surf (k) <fix_move_surf>`
     - :doc:`neutral/collide/background <fix_neutral_collide_background>`
   * - :doc:`particle/tag <fix_particle_tag>`
     - :doc:`particle/weight <fix_particle_weight>`
     - :doc:`particulate/charge <fix_particulate_charge>`
     - :doc:`particulate/drag <fix_particulate_drag>`
     - :doc:`particulate/emit <fix_particulate_emit>`
   * - :doc:`particulate/thermal <fix_particulate_thermal>`
     - :doc:`print <fix_print>`
     - :doc:`reflect/psi <fix_reflect_psi>`
     - :doc:`store/force <fix_store_force>`
     - :doc:`surf/temp <fix_surf_temp>`
   * - :doc:`surface/emit/puff <fix_surface_emit_puff>`
     - :doc:`surface/emit/recycle <fix_surface_emit_recycle>`
     - :doc:`surface/emit/source (k) <fix_surface_emit_source>`
     - :doc:`surface/state/lm <fix_surface_state_lm>`
     - :doc:`temp/global/rescale <fix_temp_global_rescale>`
   * - :doc:`temp/rescale (k) <fix_temp_rescale>`
     - :doc:`vibmode (k) <fix_vibmode>`
     - :doc:`volume/chem/adas (k) <fix_volume_chem_adas>`
     -
     -

----------

Compute styles
--------------

See the :doc:`compute <compute>` command for one-line descriptions of
each style or click on the style itself for a full description.
Styles with a Kokkos version are marked (k).

.. list-table::

   * - :doc:`boundary (k) <compute_boundary>`
     - :doc:`count (k) <compute_count>`
     - :doc:`distsurf/grid (k) <compute_distsurf_grid>`
     - :doc:`dt/grid (k) <compute_dt_grid>`
     - :doc:`eflux/grid (k) <compute_eflux_grid>`
   * - :doc:`grid (k) <compute_grid>`
     - :doc:`grid/weighted (k) <compute_grid_weighted>`
     - :doc:`isurf/grid <compute_isurf_grid>`
     - :doc:`ke/particle (k) <compute_ke_particle>`
     - :doc:`lambda/grid (k) <compute_lambda_grid>`
   * - :doc:`nearest_surf/grid (k) <compute_nearest_surf_grid>`
     - :doc:`pflux/grid (k) <compute_pflux_grid>`
     - :doc:`plasma/fields (k) <compute_plasma_fields>`
     - :doc:`property/grid (k) <compute_property_grid>`
     - :doc:`property/surf <compute_property_surf>`
   * - :doc:`react/boundary <compute_react_boundary>`
     - :doc:`react/surf <compute_react_surf>`
     - :doc:`react/isurf/grid <compute_react_isurf_grid>`
     - :doc:`reduce <compute_reduce>`
     - :doc:`sonine/grid (k) <compute_sonine_grid>`
   * - :doc:`surf (k) <compute_surf>`
     - :doc:`surface/chemical/adatom <compute_surface_chemical_adatom>`
     - :doc:`surface/chemical/evaporation <compute_surface_chemical_evaporation>`
     - :doc:`surface/physical/sputter (k) <compute_surface_physical_sputter>`
     - :doc:`thermal/grid (k) <compute_thermal_grid>`
   * - :doc:`temp (k) <compute_temp>`
     - :doc:`tvib/grid <compute_tvib_grid>`
     - :doc:`volume/emissivity/grid <compute_volume_emissivity_grid>`
     - 
     - 

----------

Collide styles
--------------

See the :doc:`collide <collide>` command for details of each style.

.. list-table::

   * - :doc:`vss (k) <collide>`

----------

Surface collide styles
----------------------

See the :doc:`surf_collide <surf_collide>` command for details of each
style.

.. list-table::

   * - :doc:`adiabatic <surf_collide>`
     - :doc:`cll <surf_collide>`
     - :doc:`diffuse (k) <surf_collide>`
     - :doc:`impulsive <surf_collide>`
   * - :doc:`partial_recycle <surf_collide>`
     - :doc:`particulate/bounce <surf_collide_particulate_bounce>`
     - :doc:`piston (k) <surf_collide>`
     - :doc:`specular (k) <surf_collide>`
   * - :doc:`td <surf_collide>`
     - :doc:`toroidal (k) <surf_collide>`
     - :doc:`transparent <surf_collide>`
     - :doc:`vanish (k) <surf_collide>`

----------

Surface reaction styles
-----------------------

See the :doc:`surf_react <surf_react>` command for details of each
style.

.. list-table::

   * - :doc:`adsorb <surf_react_adsorb>`
     - :doc:`global (k) <surf_react>`
     - :doc:`prob (k) <surf_react>`
   * - :doc:`surface/pwi (k) <surf_react_surface_pwi>`
     - :doc:`vanish_track <surf_react>`
     -
